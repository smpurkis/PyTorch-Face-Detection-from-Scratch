import torch
import torch.nn as nn
from torchvision.transforms import transforms

from models import BaseModel


class ResidualBlock(nn.Module):
    def __init__(self, filters, num_of_patches, dropout=0.25):
        super().__init__()
        self.num_of_patches = num_of_patches
        self.conv1 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(3, 3),
            padding="same"
        )
        self.max_pool = nn.MaxPool2d(2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout2d = nn.Dropout2d(dropout)

    def forward(self, x):
        skip_x = x
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout2d(x)
        x = x + skip_x
        if x.shape[2] > self.num_of_patches:
            x = self.max_pool(x)
        return x


class Resnet(BaseModel):
    def __init__(self, filters, input_shape, num_of_patches=16, num_of_residual_blocks=10, probability_threshold=0.5,
                 iou_threshold=0.5, pretrained=False):
        super().__init__(filters, input_shape, num_of_patches=16, probability_threshold=0.5, iou_threshold=0.5)
        self.pretrained = pretrained
        self.dropout2d = nn.Dropout2d(0.5)
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=filters,
            kernel_size=(10, 10),
            stride=(4, 4),
            padding=5,
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(
                filters=filters,
                num_of_patches=self.num_of_patches
            ) for _ in range(num_of_residual_blocks)]
        )
        self.out = nn.Conv2d(
            in_channels=filters,
            out_channels=5,
            stride=(1, 1),
            kernel_size=(6, 6),
            padding=0
        )
        self.sigmoid = nn.Sigmoid()
        self.resize = transforms.Resize(size=self.input_shape[1:])

    def forward(self, x, predict=False):
        if predict:
            x = self.resize(x) / 255.
            if len(x.shape) == 3:
                x = torch.unsqueeze(x, 0)
        x = self.conv1(x)
        x = self.residual_blocks(x)
        x = self.dropout2d(x)
        x = self.out(x)
        x = self.sigmoid(x)
        if predict:
            x = self.non_max_suppression(x)
        return x


if __name__ == '__main__':
    input_shape = (480, 480)
    bm = Resnet(
        filters=64,
        input_shape=(3, *input_shape),
        num_of_patches=10,
        num_of_residual_blocks=10
    ).cuda()
    bm.eval()
    bm.summary()
    t = torch.rand(1, *(3, *input_shape)).cuda()
    from time import time

    s = time()
    p = bm(t)
    s = time() - s
    fps = 1 / s
    print(s, fps)
    print(p.shape)
    # bm.summary()
    # i = 0
