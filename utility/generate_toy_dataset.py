# """
# @author   Maksim Penkin @MaksimPenkin
# @author   Oleg Khokhlov @okhokhlov
# """

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2


def check_positive_int(value):
    """Method for checking value to be positive integer.

    :param value: input value
    :raises TypeError: exception is raised if value is not positive
    :return ivalue: resulting value
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('%s is an invalid positive int value' % value)
    return ivalue


def parse_args():
    """Method for parsing args."""
    parser = argparse.ArgumentParser(description='Toy Circle arguments', usage='%(prog)s [-h]')

    parser.add_argument('--num_images', type=check_positive_int, default=1000,
                        help='number of images to generate', metavar='')

    parser.add_argument('--height', type=check_positive_int, default=128,
                        help='image height', metavar='')
    parser.add_argument('--width', type=check_positive_int, default=128,
                        help='image width', metavar='')

    parser.add_argument('--min_rad', type=check_positive_int, default=5,
                        help='minimal radius of generated circles on each image', metavar='')
    parser.add_argument('--max_rad', type=check_positive_int, default=25,
                        help='maximal radius of generated circles on each image', metavar='')

    parser.add_argument('--min_num_circles', type=check_positive_int, default=3,
                        help='minimal number of generated circles on each image', metavar='')
    parser.add_argument('--max_num_circles', type=check_positive_int, default=10,
                        help='maximal number of generated circles on each image', metavar='')

    parser.add_argument('--path_out', type=str, default=r'E:\GraphicalCNN\data\train',
                        help='path save generated images', metavar='')
    parser.add_argument('--force', default=False, action='store_true',
                        help='if set, overwrite the existed path_out dir')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import utils as utils

    print('Creating dataset...')
    args = parse_args()

    num_train = args.num_images
    H, W = args.height, args.width
    min_rad = args.min_rad
    max_rad = args.max_rad
    min_num_circles = args.min_num_circles
    max_num_circles = args.max_num_circles
    path_experiment = args.path_out
    utils.create_folder(path_experiment, force=args.force)

    image_names = []
    mask_names = []

    for num in tqdm(range(num_train)):
        img = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        img[:, :, 0] += np.random.randint(low=0, high=256)
        img[:, :, 1] += np.random.randint(low=0, high=256)
        img[:, :, 2] += np.random.randint(low=0, high=256)

        mask = np.ones(shape=(H, W), dtype=np.uint8)

        for i in range(np.random.randint(low=min_num_circles, high=max_num_circles + 1)):
            center_coordinates = (np.random.randint(low=max_rad, high=W - max_rad),
                                  np.random.randint(low=max_rad, high=H - max_rad))
            radius = np.random.randint(low=min_rad, high=max_rad)
            color = (np.random.randint(low=20, high=256),
                     np.random.randint(low=20, high=256),
                     np.random.randint(low=20, high=256))
            thickness = -1

            img = cv2.circle(img, center_coordinates, radius, color, thickness)
            mask = cv2.circle(mask, center_coordinates, radius, (0, 0, 0), thickness)

        img = img.astype(np.float32) + np.random.normal(loc=0., scale=15., size=(H, W, 3)).astype(np.float32)
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

        cv2.imwrite(os.path.join(path_experiment, 'img{}.png').format(num), (img * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(path_experiment, 'mask{}.png').format(num), (mask * 255).astype(np.uint8))
        image_names.append('img{}.png'.format(num))
        mask_names.append('mask{}.png'.format(num))

    df = pd.DataFrame({'image': image_names, 'mask': mask_names})
    df.to_csv(os.path.join(os.path.split(path_experiment)[0], os.path.split(path_experiment)[-1]+'.csv'), index=False)
    print('Finished!')
