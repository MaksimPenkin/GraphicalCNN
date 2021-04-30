import unittest
from neural.blocks.layers import conv3x3, ResBlock, ConvBlock, UpSample, conv5x5
import torch


class TestNeuralBlocks(unittest.TestCase):
    """base test class"""

    def test_conv_3x3_bloc(self):
        """Test conv block"""
        block = conv3x3(3, 64)
        tensor = torch.randn(8, 3, 224, 224)

        self.assertSequenceEqual(block(tensor).shape, (8, 64, 224, 224))

    def test_res_bloc(self):
        """Test residual block"""
        block = ResBlock(16)
        tensor = torch.randn(8, 16, 224, 224)

        self.assertSequenceEqual(block(tensor).shape, (8, 16, 224, 224))

    def test_conv_5x5_bloc(self):
        """Test conv block"""
        block = conv5x5(3, 64)
        tensor = torch.randn(8, 3, 224, 224)

        self.assertSequenceEqual(block(tensor).shape, (8, 64, 224, 224))

    def test_conv_bloc(self):
        """Test conv block"""
        block = ConvBlock(16)
        tensor = torch.randn(8, 16, 224, 224)

        self.assertSequenceEqual(block(tensor).shape, (8, 16, 224, 224))

    def test_up_sample_bloc(self):
        """Test conv block"""
        block = UpSample(3)
        tensor_1 = torch.randn(8, 3, 112, 112)
        tensor_2 = torch.randn(8, 3, 224, 224)

        self.assertSequenceEqual(block(tensor_1, tensor_2).shape, (8, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
