# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import numpy as np
import cv2
import torch


class Resize(object):
    """A class to represent a resize augmentation."""

    def __init__(self, h, w):
        """Constructor method."""
        self.h = h
        self.w = w

    def __call__(self, sample):
        """Call method.

        :param sample: a dictionary {'image': img, 'mask': mask}
        :return sample: a dictionary with resized {'image': img, 'mask': mask}
        """
        image, mask = sample['image'], sample['mask']

        image_out = cv2.resize(image, (self.w, self.h)).copy()
        mask_out = cv2.resize(mask,   (self.w, self.h))[..., np.newaxis].copy()
        return {'image': image_out, 'mask': mask_out}


class HorizontalFlipRandom(object):
    """A class to represent a horizontal flip augmentation."""

    def __call__(self, sample):
        """Call method.

        :param sample: a dictionary {'image': img, 'mask': mask}
        :return sample: a dictionary with flipped {'image': img, 'mask': mask}
        """
        self.mode = np.random.randint(low=0, high=2)

        if self.mode:
            image, mask = sample['image'], sample['mask']
            return {'image': np.flip(image, axis=0).copy(), 'mask': np.flip(mask, axis=0).copy()}
        else:
            return sample


class VerticalFlipRandom(object):
    """A class to represent a vertical flip augmentation."""

    def __call__(self, sample):
        """Call method.

        :param sample: a dictionary {'image': img, 'mask': mask}
        :return sample: a dictionary with flopped {'image': img, 'mask': mask}
        """
        self.mode = np.random.randint(low=0, high=2)

        if self.mode:
            image, mask = sample['image'], sample['mask']
            return {'image': np.flip(image, axis=1).copy(), 'mask': np.flip(mask, axis=1).copy()}
        else:
            return sample


class Rot90Random(object):
    """A class to represent a rotate90 augmentation."""

    def __call__(self, sample):
        """Call method.

        :param sample: a dictionary {'image': img, 'mask': mask}
        :return sample: a dictionary with rotated {'image': img, 'mask': mask}
        """
        self.mode = np.random.randint(low=0, high=4)
        if self.mode:
            image, mask = sample['image'], sample['mask']
            return {'image': np.rot90(image, k=self.mode).copy(), 'mask': np.rot90(mask, k=self.mode).copy()}
        else:
            return sample


class ScaleRotate(object):
    """A class to represent a scale&rotate augmentation."""

    def __init__(self, s_min=1., s_max=1., a_min=-5., a_max=5.):
        """Constructor method."""
        self.s_min = s_min
        self.s_max = s_max
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, sample):
        """Call method.

        :param sample: a dictionary {'image': img, 'mask': mask}
        :return sample: a dictionary with scaled and rotated {'image': img, 'mask': mask}
        """
        scale = np.random.uniform(self.s_min, self.s_max)  # random scaling
        angle = np.random.uniform(self.a_min, self.a_max)  # random rotation degrees
        image, mask = sample['image'], sample['mask']

        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), angle, scale)
        image_out = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_NEAREST).copy()
        mask_out = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST)[..., np.newaxis].copy()
        mask_out = mask_out * (mask_out > 0)
        return {'image': image_out, 'mask': mask_out}


class ToTensor(object):
    """A class to represent a ToTensor augmentation."""

    def __call__(self, sample):
        """Call method.

        :param sample: a dictionary {'image': img, 'mask': mask}
        :return sample: a dictionary with tensors {'image': img, 'mask': mask}
        """
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        mask = np.concatenate((1 - mask, mask), axis=0)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}
