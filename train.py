import os
import time
from pathlib import Path
import datetime
import logging

import torch

# 导入定义的模型
from model import lraspp_mobilenetv3_large, Unet, deeplabv3_resnet50

# 导入自定义的工具函数
from utils import train_one_epoch, evaluate, create_lr_scheduler, criterion, \
    print_env_info, increment_path, plot_results

# 导入自定义的数据集类
from my_dataset import MyDataset

# 导入自定义的数据处理函数
import transforms as T


# 定义训练集数据预处理类
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 数据增强方式包括：随机缩放、随机水平翻转、随机裁剪、归一化
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


# 定义验证集数据预处理类
class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 数据预处理方式包括：随机缩放、归一化
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


# 获取数据预处理方式
def get_transform(train):
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


# 创建模型
def create_model(num_classes, pretrain=False):
    model = lraspp_mobilenetv3_large(num_classes=num_classes)

    if pretrain:
        # 加载预训练模型参数
        weights_dict = torch.load("./lraspp_mobilenet_v3_large.pth", map_location='cpu')

        # 如果不是21类别的数据集，则删除和类别相关的权重
        if num_classes != 21:
            for k in list(weights_dict.keys()):
                if "low_classifier" in k or "high_classifier" in k:
                    del weights_dict[k]

        # 加载模型参数
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


# 主函数
def main(args):
    # 配置logging模块
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 定义存储模型权重、训练结果等信息的路径
    base_path = Path('runs/train/exp')
    save_dir = increment_path(base_path=base_path)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results{}.txt'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 判断设备类型，并将模型放到相应的设备上
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print_env_info(device)

    # 定义每个batch的大小、类别数量等参数
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 定义训练集和验证集的数据集，并进行数据预处理
    train_dataset = MyDataset(True, args.data_path, transforms=get_transform(train=True))
    val_dataset = MyDataset(False, args.data_path, transforms=get_transform(train=False))

    # 输出训练集和验证集的图像数量
    logging.info('train image: {}, val image: {}'.format(len(train_dataset), len(val_dataset)))

    # 定义num_classes是否匹配的检查
    # logging.info('Scanning mask files to determine unique values')
    # train_num_classes = train_dataset.check_num_classes()
    # logging.info(f'Train Datasets Unique mask values: {train_num_classes}')
    #
    # val_num_classes = val_dataset.check_num_classes()
    # logging.info(f'Train Datasets Unique mask values: {val_num_classes}')
    # assert

    # 定义num_workers的数量
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 定义训练集和验证集的数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    model = create_model(num_classes=num_classes)  # 创建模型
    model.to(device)  # 将模型移动到指定设备上（GPU或CPU）

    optimizer = torch.optim.SGD(  # 定义优化器，使用随机梯度下降算法
        model.parameters(),  # 优化器需要优化的参数
        lr=args.lr,  # 学习率
        momentum=args.momentum,  # 动量参数
        weight_decay=args.weight_decay  # L2正则化参数
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None  # 如果使用混合精度训练，则创建GradScaler实例，否则为None

    # 定义学习率更新策略，这里采用每个step更新一次（不是每个epoch）
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:  # 如果指定了断点续训
        checkpoint = torch.load(args.resume, map_location='cpu')  # 加载断点
        model.load_state_dict(checkpoint['model'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])  # 加载学习率更新策略状态
        args.start_epoch = checkpoint['epoch'] + 1  # 设置当前epoch
        if args.amp:  # 如果使用混合精度训练
            scaler.load_state_dict(checkpoint["scaler"])  # 加载GradScaler状态

    start_time = time.time()  # 记录开始时间
    loss_list = []  # 保存每个epoch的平均损失
    miou_list = []  # 保存每个epoch的平均IoU
    best_miou = 0.0  # 保存最佳平均IoU

    # 训练多个epoch
    for epoch in range(args.start_epoch, args.epochs):

        # 训练一个epoch
        mean_loss, lr = train_one_epoch(model, criterion, optimizer, train_loader, device=device, epoch=epoch,
                                        lr_scheduler=lr_scheduler, scaler=scaler, epochs=args.epochs)

        # 在验证集上评估模型并计算混淆矩阵
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)

        # 将训练和验证信息写入txt文件
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        # 保存模型参数和优化器状态
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        # 如果使用混合精度训练，则保存scaler状态
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # 获取验证集结果的最后一项，并转化为float类型
        miou = float(val_info.split('\n')[-1].split(' ')[-1])

        # 如果当前miou值比历史最佳值高，则更新历史最佳值并保存模型
        if best_miou < miou:
            best_miou = miou
            torch.save(save_file, best)
        # 保存最终的模型
        torch.save(save_file, last)

    # 训练时间计算
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    # 绘制结果并保存训练日志
    plot_results(results_file, save_dir)
    logging.info('weights and results save in {}'.format(save_dir))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch lraspp training")

    parser.add_argument("--data-path", default="/home/wg/PythonProject/PycharmProjects/medicine/dataset/1-422/", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=4, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)