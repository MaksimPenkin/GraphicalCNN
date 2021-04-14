""" 
 @author   Maksim Penkin @MaksimPenkin
 @author   Oleg Khokhlov @okhokhlov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural.blocks.layers import ResBlock, ConvBlock, UpSample


class DecoderResBlock(nn.Module):
    def __init__(self, num_filters, num_blocks=4, batch_norm=False):
        super().__init__()
        
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm

        for i in range(self.num_blocks, 0, -1):
            self.add_module(f"deconv_{i}", UpSample(self.num_filters * 2**i, 2))
            self.add_module(f"resblock_{i}", ResBlock(self.num_filters * 2**(i-1), batch_norm=self.batch_norm))

    def forward(self, acts):
        y = acts[-1]
        for i in range(self.num_blocks, 0, -1):
            left = acts[i-1]
            skip = self.__getattr__(f"deconv_{i}")(y, left)
            y = self.__getattr__(f"resblock_{i}")(skip)
        
        return y


class DecoderConvBlock(nn.Module):
    def __init__(self, num_filters, num_blocks=4, batch_norm=False):
        super().__init__()
        
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm

        for i in range(self.num_blocks, 0, -1):
            self.add_module(f"deconv_{i}", UpSample(self.num_filters * 2**i, 2))
            self.add_module(f"convblock_{i}", ConvBlock(self.num_filters * 2**(i-1), batch_norm=self.batch_norm))

    def forward(self, acts):
        y = acts[-1]
        for i in range(self.num_blocks, 0, -1):
            left = acts[i-1]
            skip = self.__getattr__(f"deconv_{i}")(y, left)
            y = self.__getattr__(f"convblock_{i}")(skip)
        
        return y


