import os
from pathlib import Path

import pandas as pd
from utils import get_image_score, increment_path
import cv2


def run(args):
    base_path = Path('runs/eval/exp')
    save_dir = increment_path(base_path)

    target, pred, num_classes = args.target, args.pred, args.num_classes
    target, pred = Path(target), Path(pred)
    targets = []  # 用于存放target，target为array
    preds = []    # 用于存放pred，pred为array
    if target.is_file():
        assert pred.is_file(), f'pred need a image'
        mask_t = cv2.imread(str(target))
        mask_p = cv2.imread(str(pred))
        targets = [mask_t]
        preds = [mask_p]

    names = None   # 图片名称
    if target.is_dir():
        assert pred.is_dir(), f'pred need a dir'
        names = os.listdir(str(target))
        for i in names:
            # target_mask_path
            mask_t_p = os.path.join(str(target), i)
            # pred_mask_path
            mask_p_p = os.path.join(str(pred), i)
            # target_mask
            mask_t = cv2.imread(mask_t_p)
            # pred_mask
            mask_p = cv2.imread(mask_p_p)
            targets.append(mask_t)
            preds.append(mask_p)

    # 用于存放所有图片的结果
    scoresMat = pd.DataFrame(columns=['img_name', 'IoU/Jaccard', 'Dice', 'Accuracy', 'Sensitivity', 'Precision', 'Fmeasure', 'MCC'])

    for idx, (t, p) in enumerate(zip(targets, preds)):
        scoreMat = get_image_score(t, p, num_classes + 1)
        scoreMat.assign(img_name=names[idx])   # 增加img_name列名
        scoresMat = scoresMat.append(scoreMat)
    scoresMat.to_csv(save_dir / 'result.csv')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch lraspp training")

    parser.add_argument("--target", default="/home/wg/PythonProject/PycharmProjects/medicine/dataset/1-422/val/mask",
                        help="label mask")
    parser.add_argument("--pred", default="/home/wg/下载/segmentation/runs/predict/exp3/output",
                        help="pred mask")
    parser.add_argument("--num_classes", default=4, type=int)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)


