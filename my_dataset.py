import logging
import os
import glob

import cv2
from tqdm import tqdm
import torch.utils.data as data
from PIL import Image
import numpy as np


class MyDataset(data.Dataset):
    def __init__(self, is_train, data_root, transforms=None):
        super(MyDataset, self).__init__()
        self.data_root = data_root
        self.is_train = is_train
        if self.is_train:
            image_path = os.path.join(self.data_root, 'train', 'image')
            mask_path = os.path.join(self.data_root, 'train', 'mask')
        else:
            image_path = os.path.join(self.data_root, 'val', 'image')
            mask_path = os.path.join(self.data_root, 'val', 'mask')
        name = [i.split('/')[-1].split('.jpg')[0] for i in glob.glob(image_path + '/*.jpg')]
        self.images = [os.path.join(image_path, n + '.jpg') for n in name]
        self.masks = [os.path.join(mask_path, n + '.png') for n in name]
        assert len(self.images) == len(self.masks)

        self.transforms = transforms
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.images)

    def check_num_classes(self):
        mask_values = []

        pbar = tqdm(self.masks, ncols=50)
        for i in pbar:
            mask = cv2.imread(self.masks[i])
            masks = np.append(masks, mask)
            if mask.ndim == 2:
                mask_values.append(np.unique(mask))
            elif mask.ndim == 3:
                mask_values.append(np.unique(mask.reshape(-1, mask.shape[-1]), axis=0))
        mask_values = np.array(mask_values)
        mask_values = mask_values.reshape(-1, mask_values.shape[-1])

        return mask_values

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets
    
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images), ) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
