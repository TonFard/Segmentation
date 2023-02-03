import glob
import logging
import re
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import warnings
warnings.filterwarnings('ignore')

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        
    def update(self, a, b):
        """
            a: label
            b: pred
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        
        with torch.inference_mode():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
        
    def reset(self):
        self.mat.zero_()
        
    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iou
    
    def __str__(self):
        acc_global, acc, iou = self.compute()
        # return ("global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}").format(
        #     acc_global.item() * 100,
        #     [f"{i:.1f}" for i in (acc * 100).tolist()],
        #     [f"{i:.1f}" for i in (iou * 100).tolist()],
        #     iou.mean().item() * 100
        # )

        return ("IoU: {}\nmean IoU: {:.1f}").format(
            [f"{i:.1f}" for i in (iou * 100).tolist()],
            iou.mean().item() * 100
        )

def create_lr_scheduler(optimizer, num_step, epochs, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0
    
    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)    


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def EvaluateImageSegmentationScores(target, pred, num_classes):

    target = target.clone().cpu().numpy()
    pred = pred.clone().cpu().numpy()

    def f(X, Y):
        assert X.shape == Y.shape, 'image shape not matching'
        sumindex = X + Y
        TP = np.sum(sumindex == 2)
        TN = np.sum(sumindex == 0)
        substractindex = X - Y
        FP = np.sum(substractindex == -1)
        FN = np.sum(substractindex == 1)
        Accuracy = (TP + TN) / (FN + FP + TP + TN)
        Sensitivity = TP / (TP + FN)
        Precision = TP / (TP + FP)
        Fmeasure = 2 * TP / (2 * TP + FP + FN)
        A = np.sqrt((TP + FP) * (TP + FN))
        B = np.sqrt((TN + FP) * (TN + FN))
        MCC = (TP * TN - FP * FN) / (A * B)
        Dice = 2 * TP / (2 * TP + FP + FN)
        Jaccard = Dice / (2 - Dice)
        IOU = Jaccard
        return np.array([IOU, Dice, Accuracy, Sensitivity, Precision, Fmeasure, MCC])

    scores = []
    for i in range(1, num_classes):
        target[target != i] = 0
        target[target == i] = 1
        pred[pred != i] = 0
        pred[pred == i] = 1
        result = f(target, pred)
        result = np.nan_to_num(result)
        scores.append({'class{}'.format(i): result})
    scores = np.array(scores).reshape(num_classes - 1, -1)
    return scores


def get_image_score(target, pred, num_classes):
    scoresMat = pd.DataFrame(columns=['IOU', 'Dice', 'Accuracy', 'Sensitivity', 'Precision', 'Fmeasure', 'MCC'])
    cls_scores = np.zeros((num_classes - 1, 7))
    scores = EvaluateImageSegmentationScores(target, pred.argmax(1), num_classes)
    for i in range(num_classes - 1):
        cls_scores[i] += scores[i][0]['class{}'.format(i + 1)]

    per_img_score = pd.DataFrame(cls_scores, index=['class{}'.format(i) for i in range(1, num_classes)],
                                 columns=['IOU', 'Dice', 'Accuracy', 'Sensitivity', 'Precision', 'Fmeasure', 'MCC'])

    for i in range(num_classes - 1):
        scoresMat = scoresMat.append(per_img_score.iloc[i, ...])

    return scoresMat


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        data_bar = tqdm(data_loader, ncols=80)

        for image, target in data_bar:

            data_bar.set_description(desc='Validation')
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, scaler=None, epochs=None):
    model.train()
    losses = 0.0
    data_bar = tqdm(data_loader, ncols=80)
    for image, target in data_bar:
        data_bar.set_description(desc=f'Epoch [{epoch}/{epochs - 1}]')
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        losses += loss.item()
        data_bar.set_postfix(**{'loss (batch)': '{:.4f}'.format(loss.item())})

    return losses / len(data_loader), lr


def print_env_info(device):
    s = f'Segmentation env Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')
    devices = device.split(',') if device else '0'
    space = ' ' * (len(s) + 1)
    for i, d in enumerate(devices):
        p = torch.cuda.get_device_properties(i)
        s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"  # bytes to MB
    logging.info(s)


def CV2toPIL(img):
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img_PIL

def PILtoCV2(img):
    img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img_cv


def increment_path(base_path):
    """
        根据文件夹中已有的文件名，自动获得新路径或文件名
        Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
        使用pathlib处理文件
    """
    base_path = Path(base_path)
    # 如果根文件夹已经存在,则不再重新创建
    if base_path.exists():

        dirs = glob.glob(f'{base_path}*')
        matches = [re.search(rf'%s(\d+)' % base_path.stem, d) for d in dirs]

        # groups对应re匹配到的内容
        i = [int(m.groups()[0]) for m in matches if m]

        n = max(i) + 1 if i else 2
        base_path = Path(f'{base_path}{n}')

    base_path.mkdir(parents=True, exist_ok=True)

    return base_path


def plot_results(results_file, save_dir):
    from matplotlib.ticker import MaxNLocator

    result_dict = {'mean IoU': [], 'train_loss': [], 'lr': []}
    with open(results_file, 'r') as f:
        for i in f.readlines():
            s_ = i.strip().split(': ')
            if len(s_) > 1:
                n = s_[0]
                if n in result_dict.keys():
                    v = float(s_[1])
                    result_dict[n].append(v)

    plt.figure(dpi=200, figsize=(10, 5))

    for idx, (key, value) in enumerate(result_dict.items()):

        x = range(len(value))
        y = value
        plt.subplot(1, 3, idx + 1)
        plt.title(key, fontsize=12)
        plt.plot(x, y, marker='.', linewidth=2, markersize=8)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

    plt.savefig(Path(save_dir) / 'results.png')

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # test print_env_info
    # print_env_info(device)

    # test increment_path
    # base_path = Path('runs/train/exp')

    # test increment_path
    # save_dir = increment_path(base_path)

    # test plot_results
    results_path = r'/home/wg/下载/segmentation/runs/train/exp5/results20230203-160025.txt'
    plot_results(results_path, save_dir='./')

