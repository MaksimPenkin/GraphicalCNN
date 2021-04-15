# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import torch.nn as nn
from neural.blocks.embedding import Embedding
from neural.blocks.encoder import EncoderResBlock, EncoderConvBlock
from neural.blocks.decoder import DecoderResBlock, DecoderConvBlock
from neural.blocks.restoration import Restoration


class Generator(nn.Module):
    """A class to represent a generator CNN."""

    def __init__(self, num_filters, num_blocks=4, batch_norm=False, arch='ResBlock'):
        """Constructor method."""
        super().__init__()

        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        self.arch = arch

        self.emb = Embedding(3, self.num_filters, self.batch_norm)
        if self.arch == 'ResBlock':
            self.encoder = EncoderResBlock(self.num_filters, self.num_blocks, self.batch_norm)
            self.decoder = DecoderResBlock(self.num_filters, self.num_blocks, self.batch_norm)
        elif self.arch == 'ConvBlock':
            self.encoder = EncoderConvBlock(self.num_filters, self.num_blocks, self.batch_norm)
            self.decoder = DecoderConvBlock(self.num_filters, self.num_blocks, self.batch_norm)
        self.restore = Restoration(self.num_filters, 2, self.batch_norm)

    def forward(self, x):
        """Method for forward pass.

        :param x: input tensor
        :return y: output tensor
        """
        x_emb = self.emb(x)
        encoder_acts = self.encoder(x_emb)
        y = self.decoder(encoder_acts)
        y = self.restore(y)

        return y
