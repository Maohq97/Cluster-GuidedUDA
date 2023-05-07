import torch
import sys
import time
import os
import tqdm
import soundfile
import numpy
import torch.nn.functional as F
from tools import *

class Trainer:
    def __init__(
            self,
            cfg,
            model,
            train_loader,
            optim,
            scheduler,
            start_epoch=0,
            ):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.optim = optim
        self.scheduler = scheduler
        self.n_epoch = cfg.n_epoch
        self.start_epoch = start_epoch
        self.epoch = self.start_epoch
        self.test_step = cfg.test_step
        self.info = {'time':None, 'loss':None, 'lr':None, 'acc':None, 'EER':50.0}
        self.logger_save_path = os.path.join(cfg.save_path, 'logger')
        self.checkpoint_save_path = os.path.join(cfg.save_path, 'checkpoint')
        self.eer_threshod = cfg.eer_threshod
        self.eval_list = cfg.eval_list
        self.eval_path = cfg.eval_path
        
    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch+self.n_epoch):
            self.epoch = epoch
            self._train()
            self.scheduler.step()
            self.log()
            if self.epoch % self.test_step == 0:
                self._eval()
    
    def _train(self):
        self.model.train()
        t_start = time.time()
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(self.train_loader, start=1):
            self.optim.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            nloss, prec = self.model.forward(data.cuda(), labels, aug=True)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(self.epoch, lr, 100 * (num / self.train_loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
            sys.stderr.flush()
            sys.stdout.write("\n")
        # update training information
        self.info['time'] = time.time() - t_start
        self.info['loss'] = loss/num
        self.info['lr'] = lr
        self.info['acc'] = float(top1/index*len(labels))
        
    def log(self):
        message = (
            f"epoch {self.epoch} "
            f"time {self.info['time']:.0f} "
            f"lr {self.info['lr']:.5f} "
            f"loss {self.info['loss']:.3f} "
            f"acc {self.info['acc']:.3f}\n"
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

