import glob
import os
import json
import time

import torch
import numpy as np
from model import lraspp_mobilenetv3_large
from PIL import Image
from torchvision import transforms


def main(img_path):
    classes = 4
    weights_path = "./save_weights/model_26.pth"
    palette_path = "./palette.json"
    with open(palette_path, 'rb') as f:
        palette_dict = json.load(palette_path)
        palette = []
        for v in palette_dict.values():
            palette += v
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = lraspp_mobilenetv3_large(num_classes=classes)

    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    model.to(device)

    image = Image.open(img_path)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])
    img = data_transform(image)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        im_h, im_w = img.shape[-2:]
        init_im = torch.zeros((1, 3, im_h, im_w), device=device)
        model(init_im)

        t_start = time.time()
        output = model(img.to(device))
        t_end = time.time()
        print('inference time: {}'.format(t_end - t_start))

        prediction = output['output'].argmax(1).squeeze(0)
        prediction = prediction.to('cpu').numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(palette)
        img_name = img_path.split('/')[-1]
        mask.save(os.path.join('/home/wg/下载/segmentation/inference_result', img_name))


if __name__ == '__main__':
    img_path = ''
    main(img_path)



