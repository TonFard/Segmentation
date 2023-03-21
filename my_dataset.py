import logging
import os
import glob

import cv2
from tqdm import tqdm
import torch.utils.data as data
from PIL import Image
import numpy as np

# 建立数据集
class MyDataset(data.Dataset):
    def __init__(self, is_train, data_root, transforms=None):
        super(MyDataset, self).__init__()
        self.data_root = data_root # 数据集的根目录
        self.is_train = is_train # 是否为训练集
        if self.is_train:
            # 训练集的图像和掩码目录
            image_path = os.path.join(self.data_root, 'train', 'image')
            mask_path = os.path.join(self.data_root, 'train', 'mask')
        else:
            # 验证集的图像和掩码目录
            image_path = os.path.join(self.data_root, 'val', 'image')
            mask_path = os.path.join(self.data_root, 'val', 'mask')
        
        # 获取图像和掩码的文件名
        name = [i.split('/')[-1].split('.jpg')[0] for i in glob.glob(image_path + '/*.jpg')]
        
        # 获取图像和掩码的文件路径
        self.images = [os.path.join(image_path, n + '.jpg') for n in name]
        self.masks = [os.path.join(mask_path, n + '.png') for n in name]
        assert len(self.images) == len(self.masks)

        self.transforms = transforms
        
    def __getitem__(self, index):
        # 读取图像和掩码
        img = Image.open(self.images[index]).convert('RGB') # 读取图像并转换为RGB格式
        target = Image.open(self.masks[index]) # 读取掩码
        
        if self.transforms is not None:
            # 进行图像和掩码的变换
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        # 返回数据集大小
        return len(self.images)
    
    # 检查数据集中的类别个数
    def check_num_classes(self):
        mask_values = []

        # 遍历所有掩码，获取掩码值
        pbar = tqdm(self.masks, ncols=50) # 在进度条中显示掩码的处理进度
        for i in pbar:
            mask = cv2.imread(self.masks[i]) # 读取掩码
            masks = np.append(masks, mask) # 将掩码添加到列表中
            if mask.ndim == 2:
                mask_values.append(np.unique(mask)) # 将掩码的唯一值添加到列表中
            elif mask.ndim == 3:
                mask_values.append(np.unique(mask.reshape(-1, mask.shape[-1]), axis=0))
        mask_values = np.array(mask_values)
        mask_values = mask_values.reshape(-1, mask_values.shape[-1])

        return mask_values

    @staticmethod
    def collate_fn(batch):
        # 将图像和掩码进行批处理
        images, targets = list(zip(*batch)) # 解压缩成两个列表
        batched_imgs = cat_list(images, fill_value=0) # 将图像进行批处理
        batched_targets = cat_list(targets, fill_value=255) # 将掩码进行批处理
        return batched_imgs, batched_targets