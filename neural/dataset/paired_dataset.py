# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import os
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset


class SegmDataset(Dataset):
    """A class to represent a segmentation dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """Constructor method."""
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Method for getting number of image-pairs in dataset."""
        return len(self.csv_data)

    def __getitem__(self, idx):
        """Method for getting current desired image-pair.

        :param idx: index of the desired image-pair
        :return sample: a dictionary {'image': img, 'mask': mask}
        """
        img_name = os.path.join(self.root_dir,
                                self.csv_data.iloc[idx, 0])
        mask_name = os.path.join(self.root_dir,
                                 self.csv_data.iloc[idx, 1])

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.

        mask = cv2.imread(mask_name, 0).astype(np.float32) / 255.
        mask = mask[..., np.newaxis]

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
