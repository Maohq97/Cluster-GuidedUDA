'''
Fine-tune the pre-trained model with the additional contrastive center loss
ECAPA-TDNN
Loss: AAM-softmax loss (source domain) + Contrastive loss (target domain) + Contrastive center loss (target domain)
'''

from omegaconf import OmegaConf
import os
from model import ECAPA_TDNN
from wav_dset_1 import wav_dset_1
import torch
from trainer import Trainer
from loss.ap_loss import LossFunction
from loss.ap_cc_loss import LossFunction_cc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load config file
cfg = OmegaConf.load('./config.yaml')
# create folders
checkpoint_save_path = os.path.join(cfg.save_path, 'checkpoint')
logger_save_path = os.path.join(cfg.save_path, 'logger')
cluster_save_path = os.path.join(cfg.save_path, 'cluster')
os.makedirs(checkpoint_save_path, exist_ok=True)
os.makedirs(logger_save_path, exist_ok=True)
os.makedirs(cluster_save_path, exist_ok=False)

# initialise
model= ECAPA_TDNN(C=1024, n_class=cfg.n_class, m=cfg.m, s=cfg.s)
Loss = LossFunction()
Loss_cc = LossFunction_cc()
train_dataset_1 = wav_dset_1(cfg.train_list_vox2, cfg.train_path_vox2, cfg)
train_loader_1 = torch.utils.data.DataLoader(train_dataset_1,
                                           batch_size=cfg.batch_size_1,
                                           shuffle=True,
                                           num_workers=cfg.n_thread,
                                           drop_last=True,
                                           )
parameters_m = model.parameters()
parameters_l = Loss.parameters()
parameters_cc = Loss_cc.parameters()
parameters = list(parameters_m) + list(parameters_l) + list(parameters_cc)
optim = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.lr_decay)

# load pre-trained model
checkpoint = torch.load(cfg.initial_model)
model.load_state_dict(checkpoint, strict=False)

trainer = Trainer(cfg,
                  model=model,
                  Loss=Loss,
                  Loss_cc=Loss_cc,
                  train_loader_1=train_loader_1,
                  optim=optim,
                  scheduler=scheduler,
                  )

# start training
trainer.train()
