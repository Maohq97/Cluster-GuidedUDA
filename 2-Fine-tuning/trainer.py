import torch
import sys
import time
import os
import tqdm
import soundfile
import numpy
import torch.nn.functional as F
from tools import *
from wav_dset_2 import wav_dset_2
from sklearn import metrics
from sklearn.metrics import accuracy_score
from Bio.Cluster import * 


class Trainer:
    def __init__(
            self,
            cfg,
            model,
            Loss,
            Loss_cc,
            train_loader_1,
            optim,
            scheduler,
            start_epoch=0,
            ):
        self.cfg = cfg
        self.model = model.cuda()
        self.Loss = Loss
        self.Loss_cc = Loss_cc
        self.train_loader_1 = train_loader_1
        self.optim = optim
        self.scheduler = scheduler
        self.n_epoch = cfg.n_epoch
        self.start_epoch = start_epoch
        self.epoch = self.start_epoch
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.gamma = cfg.gamma
        self.test_step = cfg.test_step
        self.info = {'time':None, 
                     'tloss':None, 
                     'nloss':None, 
                     'kloss':None, 
                     'PS':None,
                     'NMI':None,
                     'CH': None,
                     'SC':None,
                     'lr':None, 
                     'acc':None, 
                     'EER':50.0}
        self.logger_save_path = os.path.join(cfg.save_path, 'logger')
        self.checkpoint_save_path = os.path.join(cfg.save_path, 'checkpoint')
        self.cluster_save_path = os.path.join(cfg.save_path, 'cluster')
        self.eer_threshod = cfg.eer_threshod
        self.eval_list = cfg.eval_list
        self.eval_path = cfg.eval_path
        self.cluster_K = cfg.cluster_K
        self.init_train_list = cfg.train_list
        self.cluster_dict = {}
        self.alpha = cfg.alpha
        self.criteria = torch.nn.CrossEntropyLoss()
        
        
    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch+self.n_epoch):
            self.epoch = epoch
            # Re-cluster
            if self.epoch == 0 or self.epoch % self.test_step == 1:
                if self.epoch != 1:
                    self._cluster()
            self._train()
            self.scheduler.step()
            self.log()
            if self.epoch % self.test_step == 0:
                self._eval()
    
    def _train(self):
        self.model.train()
        t_start = time.time()
        index, loss, n_loss_1, n_loss_2, k_loss = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, zip_data in enumerate(zip(self.train_loader_1, self.train_loader_2), start=1):
            self.optim.zero_grad()
            
            # source domain classification loss
            data_1, labels_1, data_2, labels_2 = zip_data[0][0], zip_data[0][1], zip_data[1][0], zip_data[1][1]
            labels_1 = torch.LongTensor(labels_1).cuda()
            nloss_1, prec_1 = self.model.forward(data_1.cuda(), labels_1, domain=1)
            
            # target domain contrastive loss
            data_2 = data_2.transpose(0,1)
            labels_2 = torch.LongTensor(labels_2).cuda()
            feat = []
            for inp in data_2:
                outp = self.model.forward(torch.FloatTensor(inp).cuda(), labels_2, domain=2)
                feat.append(outp)
            feat = torch.stack(feat,dim=1).squeeze()
            nloss_2 = self.Loss.forward(feat, None)
            
            # target domain contrastive center loss
            feat_mean = torch.mean(feat, 1)
            kloss = self.Loss_cc(feat_mean, self.centers, labels_2)
            
            # total loss
            nloss = self.alpha * nloss_1 + self.beta * nloss_2 + self.gamma * kloss
            
            nloss.backward()
            self.optim.step()
            index += len(labels_1)
            loss += nloss.detach().cpu().numpy()
            n_loss_1 += nloss_1.detach().cpu().numpy()
            n_loss_2 += nloss_2.detach().cpu().numpy()
            k_loss += kloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(self.epoch, lr, 100 * (num / self.train_loader_2.__len__())) + \
            " Loss: %.5f \r"        %(loss/(num)))
            sys.stderr.flush()
            sys.stdout.write("\n")

        # update training information
        self.info['time'] = time.time() - t_start
        self.info['n_loss_1'] = n_loss_1/num
        self.info['n_loss_2'] = n_loss_2/num
        self.info['kloss'] = k_loss/num
        self.info['lr'] = lr
        
    def log(self):
        message = (
            f"epoch {self.epoch} "
            f"time {self.info['time']:.0f} "
            f"lr {self.info['lr']:.5f} "
            f"nloss_1 {self.info['n_loss_1']:.3f} "
            f"nloss_2 {self.info['n_loss_2']:.3f} "
            f"kloss {self.info['kloss']:.3f} "
            f"PS {self.info['PS']:.3f} " 
            f"NMI {self.info['NMI']:.3f} "
            f"CH {self.info['CH']:.3f} "
            f"SC {self.info['SC']:.3f}\n"
            )
        with open(os.path.join(self.logger_save_path, 'training.log'), 'a') as f:
            f.write(message)
    
    def _eval(self):
        self.model.eval()
        files = []
        embeddings = {}
        lines = open(self.eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _  = soundfile.read(os.path.join(self.eval_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis = 0).astype(numpy.float)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.model.extract(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.model.extract(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        
        scores, labels  = [], []
        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))
        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        
        # Save evaluation results
        message = (
            f"epoch {self.epoch} "
            f"EER {EER:.2f} "
            f"minDCF {minDCF:.2f}\n"
            )
        with open(os.path.join(self.logger_save_path, 'evaluation.log'), 'a') as f:
            f.write(message)
        
        # save model checkpoing
        if EER < self.info['EER'] and EER < self.eer_threshod:
            save2path = os.path.join(self.checkpoint_save_path, f'model_{self.epoch}.pt')     
            torch.save(self.model.state_dict(), save2path)
            
            self.info['EER'] = EER
            
    @torch.no_grad()
    def _cluster(self):
        # perform clustering on target domain data 
        self.model.eval()
        save_path = os.path.join(self.cluster_save_path, f'train_list_cn1_{self.epoch}.txt')
        
        # step1: extract embeddings
        lines = open(self.init_train_list).read().splitlines()
        labels = []
        utters = []
        embeddings = torch.tensor([])
        for idx, line in tqdm.tqdm(enumerate(lines), total=len(lines)):
            label = line.split()[0]
            utter = line.split()[1]
            labels.append(label)
            utters.append(utter)
            utter_path = os.path.join(self.cfg.train_path, utter)
            audio, sr = soundfile.read(utter_path)
            audio = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
            embedding = self.model.extract(audio)
            embedding = embedding.detach().cpu()
            # normalization
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings = torch.cat((embeddings, embedding), dim=0)       
        
        # step2: k-means
        # cosine similarity metric
        embeddings_np = embeddings.numpy()
        pred_labels, cdata = KCluster(embeddings_np, self.cluster_K)
        pred_labels = ['id'+str(pre_label).zfill(5) for pre_label in pred_labels]
        
        # NMI, CH index, and Silhouette score
        self.info['NMI'] = metrics.normalized_mutual_info_score(labels, pred_labels)
        self.info['CH'] = metrics.calinski_harabasz_score(embeddings_np, pred_labels)
        self.info['SC'] = metrics.silhouette_score(embeddings_np, pred_labels, metric='euclidean')
        # Purity
        true_y = []
        pred_y = []
        for label in labels:
            true_y.append(int(label.split('id')[1]))
        for label in pred_labels:
            pred_y.append(int(label.split('id')[1]))
        pred_y = numpy.array(pred_y)
        true_y = numpy.array(true_y)
        self.info['PS'] = purity_score(true_y, pred_y)
        
        # step3: calculate cluster center
        self.centers = torch.from_numpy(cdata) # [K, 192]
        self.centers = self.centers.cuda()
        
        # step4: save clustering result
        for i, speaker_id in enumerate(pred_labels):
            with open(save_path, 'a') as f:
                f.write(f'{speaker_id} {utters[i]}\n')
                
        # create train loader
        self.cfg.train_list = save_path
        print(f'training list: {self.cfg.train_list}')
        self.train_dataset_2 = wav_dset_2(self.cfg)
        self.train_loader_2 = torch.utils.data.DataLoader(self.train_dataset_2,
                                                   batch_size=self.cfg.batch_size_2,
                                                   shuffle=True,
                                                   num_workers=self.cfg.n_thread,
                                                   drop_last=True,
                                                   )

def purity_score(y_true, y_pred):
    y_voted_labels = numpy.zeros(y_true.shape)
    labels = numpy.unique(y_true)
    ordered_labels = numpy.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = numpy.unique(y_true)
    bins = numpy.concatenate((labels, [numpy.max(labels)+1]), axis=0)

    for cluster in numpy.unique(y_pred):
        hist, _ = numpy.histogram(y_true[y_pred==cluster], bins=bins)
        winner = numpy.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

# K-means Clustering
def KCluster(data, K=2):
    clusterid, error, nfound = kcluster (data, nclusters=K, mask=None, weight=None, transpose=0, npass=1, method='a', dist='u', initialid=None)
    cdata, cmask = clustercentroids(data, mask=None, transpose=0, clusterid=clusterid, method='a')
    
    return clusterid, cdata

