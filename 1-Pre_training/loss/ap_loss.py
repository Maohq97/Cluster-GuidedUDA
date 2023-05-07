'''
ref: https://github.com/joonson/voxceleb_unsupervised
AngleProto Loss
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class LossFunction(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, R=None):
        assert x.size()[1] >= 2
        out_positive = x[:,0,:]
        out_anchor = x[:,1,:]
        stepsize = out_anchor.size()[0]
        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6) 
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        loss = self.criterion(cos_sim_matrix, label)
        
        return loss