from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
from lpips_criterion import Lpips


class RGBD_Dataset(Dataset):

    def __init__(self, data_path, init_num, num):
        self.data_path = data_path
        self.num = num
        self.init_num = init_num

    def __getitem__(self, index):
        im_rgb_path = self.data_path + 'im_rgb/'
        im_depth_path = self.data_path + 'im_depth/'
        im_blur_path = self.data_path + 'im_blur/'

        im_rgb = cv2.imread(im_rgb_path + str(self.init_num + index) + '.png').transpose([2, 0, 1]).astype(np.float32) / 255.
        im_depth = cv2.imread(im_depth_path + str(self.init_num + index) + '.png', 0).astype(np.float32)    # range:36-200mm
        im_blur = cv2.imread(im_blur_path + str(self.init_num + index) + '.png').transpose([2, 0, 1]).astype(np.float32) / 255.

        im_rgb = torch.as_tensor(im_rgb, dtype=torch.float32)
        im_depth = torch.as_tensor(im_depth, dtype=torch.float32)
        im_blur = torch.as_tensor(im_blur, dtype=torch.float32)

        return im_rgb, im_depth, im_blur

    def __len__(self):
        return self.num


def load_data(data_path, total_num=52132, batch_size=20):
    train_set = RGBD_Dataset(data_path, int(0.2 * total_num), int(0.8 * total_num))
    val_set = RGBD_Dataset(data_path, int(0.1 * total_num), int(0.1 * total_num))
    test_set = RGBD_Dataset(data_path, 0, int(0.1 * total_num))
    train_data = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, pin_memory=True, shuffle=True)
    val_data = DataLoader(dataset=val_set, num_workers=4, batch_size=1, pin_memory=True, shuffle=False)
    test_data = DataLoader(dataset=test_set, num_workers=4, batch_size=1, pin_memory=True, shuffle=False)
    return train_data, val_data, test_data


