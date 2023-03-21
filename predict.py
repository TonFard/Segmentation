import glob
import os
from pathlib import Path
import json
import time

import torch
import numpy as np
from model import lraspp_mobilenetv3_large
from utils import increment_path
from PIL import Image
from torchvision import transforms


def run(args):
    source, weights, num_classes, palette_path, device, with_color = args.source, args.weights, \
        args.num_classes, args.palette, args.device, args.with_color
    # 设置保存预测结果的目录
    base_path = Path('runs/predict/exp')
    save_dir = increment_path(base_path)
    save_img_dir = save_dir / 'output'
    save_img_dir.mkdir(parents=True, exist_ok=True)

    # 判断输入源是一个文件还是目录
    source = Path(source)
    imgs_path = None
    if source.is_file():
        imgs_path = [source]

    if source.is_dir():
        imgs_path = glob.glob(str(source) + '/*.png') + glob.glob(str(source) + '/*.jpg')

    # 加载调色板
    with open(palette_path, 'rb') as f:
        palette_dict = json.load(f)
        palette = []
        for v in palette_dict.values():
            palette += v

    # 打印当前使用的设备类型
    print("using {} device.".format(device))

    # 加载模型并设置输出类别数
    num_classes = num_classes + 1
    model = lraspp_mobilenetv3_large(num_classes=num_classes)

    # 加载预训练权重
    weights_dict = torch.load(weights, map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    model.to(device)

    # 对每张输入图像进行预测
    for img_path in imgs_path:
        # 打开图像文件并进行数据增强和归一化处理
        image = Image.open(img_path).convert('RGB')
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        img = data_transform(image)
        img = torch.unsqueeze(img, dim=0)

        # 对图像进行推理
        model.eval()
        with torch.no_grad():
            im_h, im_w = img.shape[-2:]
            init_im = torch.zeros((1, 3, im_h, im_w), device=device)
            model(init_im)

            t_start = time.time()
            output = model(img.to(device))
            t_end = time.time()
            print('推理时间: {}'.format(t_end - t_start))

            # 获取预测结果并保存
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to('cpu').numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            img_name = img_path.split('/')[-1]

            # 是否需要对预测结果进行调色
            if with_color:
                mask.putpalette(palette)

            if mask.mode == 'P':
                mask = mask.convert('RGB')
            mask.save(os.path.join(save_img_dir, img_name))

    # 所有图像的预测都完成
    print('predict over!!!')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch lraspp training")

    parser.add_argument("--source", default="/home/wg/PythonProject/PycharmProjects/medicine/dataset/1-422/val/image",
                        help="image source (file or dir)")
    parser.add_argument("--weights", default='/home/wg/下载/segmentation/runs/train/exp2/weights/best.pt', help='model weights path')
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--palette", default="./palette.json", help='color for output')
    parser.add_argument("--with_color", default=True, help='whether or need color')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    run(args)