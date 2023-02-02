from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
            x.shape -> [B, C, H, W]
        """
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
            x.shape -> [B, in_channels, H, W]
        """
        return self.maxpool_conv(x)  # x.shape -> [B, out_channels, H // 2, W // 2]


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        """
            上采样模块: 上采样，跳跃连接，卷积层
        """
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
            x1, x2 -> [B, C1, H1, W1], [B, C2, H2, W2]
        """
        # 上采样
        x1 = self.up(x1)

        diffH = torch.tensor([x2.size()[2] - x1.size()[2]])   # H2 - H1
        diffW = torch.tensor([x2.size()[3] - x1.size()[3]])   # W2 - W1
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])  # [H1, W1] = [H2, W2]

        # 跳跃连接
        x = torch.cat([x2, x1], dim=1)   # [B, C1 + C2, H, W]

        # 卷积层
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, num_classes, in_channels=3, bilinear=True):
        super(Unet, self).__init__()
        """      
             ->x1 ------------cat------------> x9->
                ↓->x2 --------cat--------> x8->↑
                    ↓->x3 ----cat----> x7->↑          
                        ↓->x4 cat> x6->↑
                            ↓->x5->↑ 
        """
        self.in_channels = in_channels    # 输入通道数
        self.num_classes = num_classes    # 输出类别数
        self.bilinear = bilinear          # 上采样方式
        factor = 2 if bilinear else 1
        self.head = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.out = OutConv(64, self.num_classes)

    def forward(self, x):
        """
            for example: x.shape -> [1, 3, 224, 224]
        """
        x1 = self.head(x)         # [1, 64, 224, 224]
        x2 = self.down1(x1)       # [1, 128, 112, 112]
        x3 = self.down2(x2)       # [1, 256, 56, 56]
        x4 = self.down3(x3)       # [1, 512, 28, 28]
        x5 = self.down4(x4)       # [1, 512, 14, 14]

        x = self.up1(x5, x4)      # [1, 256, 28, 28]
        x = self.up2(x, x3)       # [1, 128, 56, 56]
        x = self.up3(x, x2)       # [1, 64, 112, 112]
        x = self.up4(x, x1)       # [1, 64, 224, 224]
        logits = self.out(x)      # [1, num_classes, 224, 224]
        result = OrderedDict()
        result['out'] = logits
        return result


if __name__ == '__main__':
    img = torch.rand((1, 3, 224, 224))
    model = Unet(in_channels=3, num_classes=20)
    output = model(img)
    print(output.shape)



