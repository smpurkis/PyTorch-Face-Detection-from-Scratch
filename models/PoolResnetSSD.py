import os

import torch
import torch.nn as nn
from torchvision.transforms import transforms

from models.SSDBaseModel import SSDBaseModel


class ResidualBlock(nn.Module):
    def __init__(self, filters, dropout=0.35, apply_max_pool: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            # padding="same"
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            # padding="same",
            padding=1
        )
        # self.bn1 = nn.BatchNorm2d(filters)
        # self.bn2 = nn.BatchNorm2d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.apply_max_pool = apply_max_pool
        self.dropout2d = nn.Dropout2d(dropout)
        if self.apply_max_pool:
            self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_x = x
        x = self.conv1(x)
        x = self.leaky_relu(x)
        # x = self.bn1(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        # x = self.bn2(x)
        x = self.dropout2d(x)
        x = x + skip_x
        if self.apply_max_pool:
            x = self.max_pool(x)
        return x

class DepthwiseResidualBlock(nn.Module):
    def __init__(self, filters, dropout=0.35, bias=True, apply_max_pool: bool = False):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            padding=1,
            groups=filters,
            bias=bias
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(1, 1),
            padding=0,
            bias=bias
        )
        # self.bn1 = nn.BatchNorm2d(filters)
        # self.bn2 = nn.BatchNorm2d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.apply_max_pool = apply_max_pool
        self.dropout2d = nn.Dropout2d(dropout)
        if self.apply_max_pool:
            self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_x = x
        x = self.depthwise_conv(x)
        x = self.leaky_relu(x)
        # x = self.bn1(x)
        x = self.pointwise_conv(x)
        x = self.leaky_relu(x)
        # x = self.bn2(x)
        x = self.dropout2d(x)
        x = x + skip_x
        if self.apply_max_pool:
            x = self.max_pool(x)
        return x


class PoolResnetSSD(SSDBaseModel):
    def __init__(self, filters, input_shape, num_of_patches, num_of_residual_blocks=10, probability_threshold=0.5,
                 iou_threshold=0.5, pretrained=False, input_kernel_size=6, input_stride=4, output_kernel_size=3,
                 output_padding=1):
        super().__init__(filters, input_shape, num_of_patches=num_of_patches,
                         probability_threshold=probability_threshold, iou_threshold=iou_threshold)
        self.pretrained = pretrained
        self.dropout2d = nn.Dropout2d(0.5)
        self.num_of_patches = num_of_patches
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=filters,
            kernel_size=(input_kernel_size, input_kernel_size),
            stride=(input_stride, input_stride),
            padding=input_kernel_size - input_stride,
        )
        self.feature_extractor = nn.Sequential(
            *[
                ResidualBlock(filters=filters, apply_max_pool=False),
                ResidualBlock(filters=filters, apply_max_pool=False),
                ResidualBlock(filters=filters, apply_max_pool=True),
                ResidualBlock(filters=filters, apply_max_pool=False),
                ResidualBlock(filters=filters, apply_max_pool=False),
                ResidualBlock(filters=filters, apply_max_pool=False),
                ResidualBlock(filters=filters, apply_max_pool=False),
                ResidualBlock(filters=filters, apply_max_pool=False),
            ]
        )
        self.residual_blocks = nn.ModuleList([nn.Sequential(
            *[
                ResidualBlock(filters=filters, apply_max_pool=False),
                ResidualBlock(filters=filters, apply_max_pool=True),
            ]
        ) for _ in range(len(self.num_of_patches))])
        self.output_blocks = nn.ModuleList([nn.Sequential(
            *[
                ResidualBlock(filters=filters, apply_max_pool=False),
                nn.Sequential(nn.Conv2d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=(3, 3),
                    padding=1,
                    groups=filters,
                ), nn.Conv2d(
                    in_channels=filters,
                    out_channels=5,
                    kernel_size=(1, 1),
                    padding=0,
                ))
            ]
        ) for _ in range(len(self.num_of_patches))])
        self.sigmoid = nn.Sigmoid()
        self.resize = transforms.Resize(size=self.input_shape[1:])

    def forward(
            self, x: torch.Tensor,
            predict: torch.Tensor = torch.tensor(0)
    ):
        if predict == 1:
            x = self.resize(x) / 255.
            if len(x.shape) == 3:
                x = torch.unsqueeze(x, 0)
        x = self.conv1(x)
        x = self.feature_extractor(x)
        outs = []
        for i in range(len(self.num_of_patches)):
            x = self.residual_blocks[i](x)
            z = self.output_blocks[i](x)
            z = self.sigmoid(z)
            outs.append(z)
        if predict == 1:
            outs = self.single_non_max_suppression(outs)
        return outs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    input_shape = (480, 480)
    bm = PoolResnetSSD(
        filters=64,
        input_shape=(3, *input_shape),
        num_of_patches=(30, 15, 7),
        num_of_residual_blocks=10
    ).cpu()
    bm.eval()
    bm.summary()
    t = torch.rand(10, *(3, *input_shape)).cpu()
    # bm = torch.jit.trace(bm, t)

    from time import time

    with torch.no_grad():
        s = time()
        p = bm(t)
        s = time() - s
    fps = 1 / s
    print(s, fps)
    # print(p.shape)
    # bm.summary()
    # i = 0
