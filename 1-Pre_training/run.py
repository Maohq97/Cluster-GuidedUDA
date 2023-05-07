'''
Pre-training
supervised training (source domain) + contrastive self supervised training (target domain)
ECAPA-TDNN
'''

from omegaconf import OmegaConf
import os
from model import ECAPA_TDNN
from wav_dset_1 import wav_dset_1
from wav_dset_2 import wav_dset_2
import torch
from trainer import Trainer
from loss.ap_loss import LossFunction
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load config file
cfg = OmegaConf.load('./config.yaml')

# create folders
checkpoint_save_path = os.path.join(cfg.save_path, 'checkpoint')
logger_save_path = os.path.join(cfg.save_path, 'logger')
os.makedirs(checkpoint_save_path, exist_ok=True)
os.makedirs(logger_save_path, exist_ok=True)

# initialise
model= ECAPA_TDNN(C=1024, n_class=cfg.n_class, m=cfg.m, s=cfg.s)
Loss = LossFunction(init_w=cfg.init_w, init_b=cfg.init_b)
train_dataset_1 = wav_dset_1(cfg.train_list_vox2, cfg.train_path_vox2, cfg)
train_loader_1 = torch.utils.data.DataLoader(train_dataset_1,
                                           batch_size=cfg.batch_size_1,
                                           shuffle=True,
                                           num_workers=cfg.n_thread,
                                           drop_last=True,
                                           )
train_dataset_2 = wav_dset_2(cfg)
train_loader_2 = torch.utils.data.DataLoader(train_dataset_2,
                                           batch_size=cfg.batch_size_2,
                                           shuffle=True,
                                           num_workers=cfg.n_thread,
                                           drop_last=True,
                                           )
parameters_m = model.parameters()
parameters_l = Loss.parameters()
parameters = list(parameters_m) + list(parameters_l)
optim = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.lr_decay)

trainer = Trainer(cfg,
                  model=model,
                  Loss=Loss,
                  train_loader_1=train_loader_1,
                  train_loader_2=train_loader_2,
                  optim=optim,
                  scheduler=scheduler,
                  )

# start training
trainer.train()