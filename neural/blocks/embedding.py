""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural.blocks.layers import conv3x3


class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.batch_norm = batch_norm
        
        self.conv = conv3x3(self.in_channels, self.out_channels)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        
        return x

