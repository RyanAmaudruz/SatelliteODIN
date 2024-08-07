import cv2
import pandas as pd
import pytorch_lightning as pl
import random
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

# from data.imagenet import logger
import numpy as np
import h5py
import torchvision
from torchvision.transforms.functional import adjust_hue, InterpolationMode, to_pil_image
from cvtorchvision import cvtransforms
from torchvision import transforms
import time

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]

S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]

S2C_MEAN_NEW = [x / 10000.0 * 255.0 for x in S2C_MEAN]

S2C_STD_NEW = [x / 10000.0 * 255.0 for x in S2C_STD]

S2C_MEAN_NEW_NP = np.array(S2C_MEAN_NEW).reshape(1, -1, 1, 1)

S2C_STD_NEW_NP = np.array(S2C_STD_NEW).reshape(1, -1, 1, 1)

S2C_MEAN_NEW_NP_F = torch.from_numpy(S2C_MEAN_NEW_NP / 255).to('cuda', torch.float32)

S2C_STD_NEW_NP_F = torch.from_numpy(S2C_STD_NEW_NP / 255).to('cuda', torch.float32)

class S2cDataModule(pl.LightningDataModule):

    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 meta_df: pd.DataFrame,
                 val_transforms = None,
                 size_val_set: int = 10):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.size_val_set = size_val_set
        self.meta_df = meta_df
        self.patch_id_list = self.meta_df['patch_id'].unique().tolist()
        self.num_images = len(self.patch_id_list)
        # self.num_images = num_images
        self.val_transforms = val_transforms
        self.im_train = None
        self.im_val = None
        self.file_list = meta_df['file_name'].tolist()
        random.shuffle(self.file_list)
        # logger.info(f"Found {len(self.file_list)} many images")
        print(f"Found {len(self.file_list)} many images")

    def __len__(self):
        # return len(self.file_list)
        return len(self.patch_id_list)

    def setup(self, stage: Optional[str] = None):
        # Split test set in val an test
        if stage == 'fit' or stage is None:
            self.im_train = UnlabelledSc2(self.patch_id_list)
            assert len(self.im_train) == self.num_images
            print(f"Train size {len(self.im_train)}")
        else:
            raise NotImplementedError("There is no dedicated val/test set.")
        # logger.info(f"Data Module setup at stage {stage}")
        print(f"Data Module setup at stage {stage}")

    def train_dataloader(self):
        return DataLoader(self.im_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)


class UnlabelledSc2(Dataset):

    def __init__(self, file_list):
        self.file_names = file_list
        # global_crops_scale = (0.6, 1.0)
        global_crops_scale = (0.3, 1.0)
        # self.resize_trans = cvtransforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC)
        self.resize_trans = cvtransforms.RandomResizedCrop(448, scale=global_crops_scale, interpolation='BICUBIC')
        self.to_tensor = cvtransforms.ToTensor()
        flip_and_color_jitter = cvtransforms.Compose([
            cvtransforms.RandomApply([
                RandomBrightness(0.4),
                RandomContrast(0.4),
                RandomSaturation(0.2),
                RandomHue(0.1)
            ], p=0.8),
            cvtransforms.RandomApply([ToGray(13)], p=0.2),
        ])

        DEFAULT_AUG = cvtransforms.Compose([
            flip_and_color_jitter,
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            ])
        self.augment1 = DEFAULT_AUG
        self.augment2 = cvtransforms.Compose([DEFAULT_AUG, cvtransforms.RandomApply([Solarize()], p=0.2)])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        patch_id = self.file_names[idx]
        with h5py.File('/gpfs/scratch1/shared/ramaudruz/s2c_un/s2c_264_light_new.h5', 'r') as f:
            data = np.array(f.get(patch_id))

        img_raw = self.to_tensor(
            self.resize_trans(np.transpose(data[np.random.choice([0,1,2,3]),:,:,:], (1, 2, 0)))
        )

        return img_raw


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class RandomBrightness(object):
    """ Random Brightness """

    def __init__(self, brightness=0.4):
        self.brightness = brightness

    def __call__(self, sample):
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        img = sample * s

        return img.clip(0, 1)

class RandomContrast(object):
    """ Random Contrast """

    def __init__(self, contrast=0.4):
        self.contrast = contrast

    def __call__(self, sample):
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        mean = sample.mean(axis=(2, 3))[:, :, None, None]
        return ((sample - mean) * s + mean).clip(0, 1)


class RandomSaturation(object):
    """ Random Contrast """

    def __init__(self, saturation=0.4):
        self.saturation = saturation

    def __call__(self, sample):

        s = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        mean = sample.mean(axis=1)[:, None, :, :]
        return ((sample - mean) * s + mean).clip(0, 1)


class RandomHue(object):
    """ Random Contrast """
    def __init__(self, hue=0.1):
        self.hue = hue

    def __call__(self, sample):
        rgb_channels = sample[:, 1:4, :, :].flip(1)
        h = np.random.uniform(0 - self.hue, self.hue)
        rgb_channels_hue_mod = adjust_hue(rgb_channels, hue_factor=h)
        sample[:, 1:4, :, :] = rgb_channels_hue_mod.flip(1)
        return sample

class ToGray(object):
    def __init__(self, out_channels):
        self.out_channels = out_channels
    def __call__(self,sample):
        nc = sample.shape[1]
        return sample.mean(axis=1)[:, None, :, :].expand(-1, nc, -1, -1)

class RandomChannelDrop(object):
    """ Random Channel Drop """

    def __init__(self, min_n_drop=1, max_n_drop=8):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def __call__(self, sample):
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)

        for c in channels:
            sample[c, :, :] = 0
        return sample


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
        self.transform = None

    def __call__(self, x):
        if self.transform is None:
            img_size = x.shape[-1]
            kernel_size = int(img_size * 0.1)
            # Make kernel size odd
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
            self.transform = torchvision.transforms.GaussianBlur(kernel_size, self.sigma)
        return self.transform(x)



class Solarize(object):

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        x1 = x.clone()
        one = torch.ones(x.shape, device='cuda')
        bool_check = x > self.threshold
        x1[bool_check] = one[bool_check] - x[bool_check]
        return x1


