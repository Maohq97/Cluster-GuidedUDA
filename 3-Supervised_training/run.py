'''
Supervised training framework of speaker verification
Training data: labeled source domain data (Vox2) + pseudo-labeled target domain data (CN1)
ECAPA-TDNN
Loss: AAM-softmax
'''

from omegaconf import OmegaConf
import os
from model import ECAPA_TDNN
from wav_dset import wav_dset
import torch
from trainer import Trainer

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
train_dataset = wav_dset(cfg)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           shuffle=True,
                                           num_workers=cfg.n_thread,
                                           drop_last=True,
                                           )
optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.lr_decay)

trainer = Trainer(cfg,
                  model=model,
                  train_loader=train_loader,
                  optim=optim,
                  scheduler=scheduler,
                  )

# start training
trainer.train()
