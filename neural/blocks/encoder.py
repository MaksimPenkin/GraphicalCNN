""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural.blocks.layers import ResBlock, ConvBlock, conv3x3


class EncoderResBlock(nn.Module):
    def __init__(self, num_filters, num_blocks=4, batch_norm=False):
        super().__init__()

        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        
        for i in range(num_blocks):        
            self.add_module(f"resblock_{i+1}", ResBlock(self.num_filters * 2**i, batch_norm=self.batch_norm))
            self.add_module(f"conv2d_proj_{i+1}", conv3x3(self.num_filters * 2**i, self.num_filters * 2**(i+1), stride=(2,2)))
        self.add_module(f"bottleneck", ResBlock(self.num_filters * 2**self.num_blocks, batch_norm=self.batch_norm))

    def forward(self, x):
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f"resblock_{i+1}")(x)
            acts.append(x)
            x = self.__getattr__(f"conv2d_proj_{i+1}")(x)
        x = self.__getattr__(f"bottleneck")(x)
        acts.append(x)
        
        return acts


class EncoderConvBlock(nn.Module):
    def __init__(self, num_filters, num_blocks=4, batch_norm=False):
        super().__init__()

        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        
        for i in range(num_blocks):        
            self.add_module(f"convblock_{i+1}", ConvBlock(self.num_filters * 2**i, batch_norm=self.batch_norm))
            self.add_module(f"conv2d_proj_{i+1}", conv3x3(self.num_filters * 2**i, self.num_filters * 2**(i+1), stride=(2,2)))
        self.add_module(f"bottleneck", ConvBlock(self.num_filters * 2**self.num_blocks, batch_norm=self.batch_norm))

    def forward(self, x):
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f"convblock_{i+1}")(x)
            acts.append(x)
            x = self.__getattr__(f"conv2d_proj_{i+1}")(x)
        x = self.__getattr__(f"bottleneck")(x)
        acts.append(x)
        
        return acts

