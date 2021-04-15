# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, padding=1, stride=1):
    """Method for convolution with 3x3 kernel.

    :param in_channels: input number of channels
    :param out_channels: resulting number of channels
    :param padding: number of pixels to pad tensor
    :param stride: stride of convolutional kernel
    :return y: output tensor
    """
    return nn.Conv2d(in_channels, out_channels, 3, padding=padding, stride=stride)


def conv5x5(in_channels, out_channels, padding=2, stride=1):
    """Method for convolution with 5x5 kernel.

    :param in_channels: input number of channels
    :param out_channels: resulting number of channels
    :param padding: number of pixels to pad tensor
    :param stride: stride of convolutional kernel
    :return y: output tensor
    """
    return nn.Conv2d(in_channels, out_channels, 5, padding=padding, stride=stride)


def conv7x7(in_channels, out_channels, padding=3, stride=1):
    """Method for convolution with 7x7 kernel.

    :param in_channels: input number of channels
    :param out_channels: resulting number of channels
    :param padding: number of pixels to pad tensor
    :param stride: stride of convolutional kernel
    :return y: output tensor
    """
    return nn.Conv2d(in_channels, out_channels, 7, padding=padding, stride=stride)


class ResBlock(nn.Module):
    """A class to represent a residual block."""

    def __init__(self, num_channels, batch_norm=False):
        """Constructor method."""
        super().__init__()

        self.num_channels = num_channels
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = conv3x3(self.num_channels, self.num_channels)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(self.num_channels, self.num_channels)

    def forward(self, x):
        """Method for forward pass.

        :param x: input tensor
        :return y: output tensor
        """
        original_x = x

        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        return x + original_x


class ConvBlock(nn.Module):
    """A class to represent a convolutional block."""

    def __init__(self, num_channels, ksize=3, batch_norm=False):
        """Constructor method."""
        super().__init__()

        self.num_channels = num_channels
        self.batch_norm = batch_norm
        self.ksize = ksize
        assert self.ksize in [3, 5, 7]

        if self.batch_norm:
            self.bn = nn.BatchNorm2d(self.num_channels)
        self.relu = nn.ReLU()

        if self.ksize == 3:
            self.conv = conv3x3(self.num_channels, self.num_channels)
        elif self.ksize == 5:
            self.conv = conv5x5(self.num_channels, self.num_channels)
        elif self.ksize == 7:
            self.conv = conv7x7(self.num_channels, self.num_channels)

    def forward(self, x):
        """Method for forward pass.

        :param x: input tensor
        :return y: output tensor
        """
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        return x


class UpSample(nn.Module):
    """A class to represent a upsample block."""

    def __init__(self, num_channels, scale_factor=2):
        """Constructor method."""
        super().__init__()
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        self.up = torch.nn.Upsample(scale_factor=self.scale_factor)
        self.conv = conv3x3(self.num_channels, self.num_channels//2)

    def forward(self, x1, x2):
        """Method for forward pass.

        :param x1: input tensor to be upsampled
        :param x2: input tensor to be summed
        :return y: output tensor
        """
        x1 = self.conv(self.up(x1))
        x = x1+x2

        return x
