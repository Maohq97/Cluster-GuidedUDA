'''
ref: https://github.com/joonson/voxceleb_unsupervised
AngleProto contrastive center Loss
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class LossFunction_cc(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction_cc, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto CC loss')

    def forward(self, feat_mean, centers, label):
        cos_sim_matrix = F.cosine_similarity(feat_mean.unsqueeze(-1), centers.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        loss = self.criterion(cos_sim_matrix, label)
        
        return loss