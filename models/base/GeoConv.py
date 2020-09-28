import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter

        self.build()

    def build(self):
        self.state_em = nn.Embedding(2, 2)   ### Optional, remove
        self.process_coords = nn.Linear(4, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj, config):
        lngs = torch.unsqueeze(traj['lngs'], dim = 2)
        lats = torch.unsqueeze(traj['lats'], dim = 2)
        states = self.state_em(traj['states'].long())   ### Optional, remove

        locs = torch.cat((lngs, lats, states), dim = 2)   ### Remove states 
        
        # map the coords into 16-dim vector
        locs = torch.tanh(self.process_coords(locs))   ### size [batch, max len of trajectory in batch, 16]
        locs = locs.permute(0, 2, 1)  

        conv_locs = F.elu(self.conv(locs))   ### size [batch, num_filter, max len of batch traj - kernel_size + 1]
        conv_locs = conv_locs.permute(0, 2, 1)   ### size [batch, max len - kernel_size + 1, num_filter]

        # calculate the dist for local paths
        local_dist = utils.get_local_seq(traj['dist_gap'], self.kernel_size, config['dist_gap_mean'], config['dist_gap_std'])
        local_dist = torch.unsqueeze(local_dist, dim = 2)   ### ### size [batch, max len - kernel_size + 1, 1]

        conv_locs = torch.cat((conv_locs, local_dist), dim = 2)   ### size [batch, max len - kernel_size + 1, num_filter + 1]

        return conv_locs

