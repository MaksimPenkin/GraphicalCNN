# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import torch.nn as nn
from neural.blocks.layers import ResBlock, ConvBlock, conv3x3


class EncoderResBlock(nn.Module):
    """A class to represent a residual encoder."""

    def __init__(self, num_filters, num_blocks=4, batch_norm=False):
        """Constructor method."""
        super().__init__()

        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm

        for i in range(num_blocks):
            self.add_module(f"resblock_{i+1}",
                            ResBlock(self.num_filters * 2**i, batch_norm=self.batch_norm))
            self.add_module(f"conv2d_proj_{i+1}",
                            conv3x3(self.num_filters * 2**i, self.num_filters * 2**(i+1), stride=(2, 2)))
        self.add_module("bottleneck",
                        ResBlock(self.num_filters * 2**self.num_blocks, batch_norm=self.batch_norm))

    def forward(self, x):
        """Method for forward pass.

        :param x: input tensor
        :return acts: list of encoder activations
        """
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f"resblock_{i+1}")(x)
            acts.append(x)
            x = self.__getattr__(f"conv2d_proj_{i+1}")(x)
        x = self.__getattr__("bottleneck")(x)
        acts.append(x)

        return acts


class EncoderConvBlock(nn.Module):
    """A class to represent a convolutional encoder."""

    def __init__(self, num_filters, num_blocks=4, batch_norm=False):
        """Constructor method."""
        super().__init__()

        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm

        for i in range(num_blocks):
            self.add_module(f"convblock_{i+1}",
                            ConvBlock(self.num_filters * 2**i, batch_norm=self.batch_norm))
            self.add_module(f"conv2d_proj_{i+1}",
                            conv3x3(self.num_filters * 2**i, self.num_filters * 2**(i+1), stride=(2, 2)))
        self.add_module("bottleneck",
                        ConvBlock(self.num_filters * 2**self.num_blocks, batch_norm=self.batch_norm))

    def forward(self, x):
        """Method for forward pass.

        :param x: input tensor
        :return acts: list of encoder activations
        """
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f"convblock_{i+1}")(x)
            acts.append(x)
            x = self.__getattr__(f"conv2d_proj_{i+1}")(x)
        x = self.__getattr__("bottleneck")(x)
        acts.append(x)

        return acts
