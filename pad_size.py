import glob
import os.path
import cv2

from tqdm import tqdm
import numpy as np

# 运行该脚本之前请对原来的数据进行备份
# 该脚本会把图片进行尺度不变的变换，默认把图片和mask转换成（512，512）的大小

img_path = r'/home/wg/PythonProject/PycharmProjects/medicine/dataset/lab_1/2ch_ed/val/image'
mask_path = r'/home/wg/PythonProject/PycharmProjects/medicine/dataset/lab_1/2ch_ed/val/mask'

for i in tqdm(glob.glob(img_path + '/*.jpg')):
    name = i.split('/')[-1].split('.jpg')[0]
    mask_n = os.path.join(mask_path, name + '.png')
    img = cv2.imread(i)
    mask = cv2.imread(mask_n)
    h, w, _ = img.shape
    t = max(h, w)
    pad_h = t - h
    pad_w = t - w
    # print(h, w, pad_h, pad_w)
    if pad_h != 0:
        pad_img = np.zeros((pad_h, w, 3), dtype=np.uint8)
        img = np.concatenate([img, pad_img], axis=0)
        mask = np.concatenate([mask, pad_img], axis=0)
    if pad_w != 0:
        pad_img = np.zeros((h, pad_w, 3), dtype=np.uint8)
        img = np.concatenate([img, pad_img], axis=1)
        mask = np.concatenate([mask, pad_img], axis=1)
    img = cv2.resize(img, (512, 512))
    mask = cv2.resize(mask, (512, 512))
    cv2.imwrite(i, img)
    cv2.imwrite(mask_n, mask)
