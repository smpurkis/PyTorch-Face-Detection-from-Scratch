import math

import torch
import torch.nn as nn
from torchinfo import summary
from torchvision.ops import nms


class ResidualBlock(nn.Module):
    def __init__(self, filters, num_of_patches):
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

    def forward(self, x):
        skip_x = x
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = x + skip_x
        if x.shape[2] > self.num_of_patches:
            x = self.max_pool(x)
        return x


class ReduceBoundingBoxes(nn.Module):
    def __init__(self, probability_threshold: float = 0.9, iou_threshold: float = 0.5, input_shape=(3, 320, 240),
                 num_of_patches=40):
        super().__init__()
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape
        _, self.width, self.height = input_shape
        self.x_patch_size = math.floor(self.width / num_of_patches)
        self.y_patch_size = math.floor(self.height / num_of_patches)

    def remove_low_probabilty_bbx(self, x):
        i, j = torch.where(x[0] > self.probability_threshold)
        if len(i) == 0 and len(j) == 0:
            return torch.empty([0]), False
        bbx = x[:, i, j].permute(1, 0)
        # bbx[:, 2] = bbx[:, 0] + bbx[:, 2]
        # bbx[:, 3] = bbx[:, 1] + bbx[:, 3]
        return bbx, True

    def convert_batch_bbx_to_xyxy_scaled(self, x):
        i, j = torch.where(x[0] > self.probability_threshold)
        x[3, i, j] = (x[3, i, j] - x[1, i, j]) * self.width + i * self.x_patch_size
        x[4, i, j] = (x[4, i, j] - x[2, i, j]) * self.height + j * self.y_patch_size
        x[1, i, j] = x[1, i, j] * self.width + i * self.x_patch_size
        x[2, i, j] = x[2, i, j] * self.height + j * self.y_patch_size
        return x

    def convert_batch_to_xywh(self, x):
        x[:, 3] = x[:, 3] - x[:, 1]
        x[:, 4] = x[:, 4] - x[:, 2]
        return x

    def forward(self, x):
        x = self.convert_batch_bbx_to_xyxy_scaled(x)
        x, boxes_exist = self.remove_low_probabilty_bbx(x)
        if boxes_exist:
            bbx = x[:, 1:]
            scores = x[:, 0]
            bbxis = nms(boxes=bbx, scores=scores, iou_threshold=self.iou_threshold)
            out = self.convert_batch_to_xywh(x[bbxis])
            return out
        else:
            return torch.empty(0)


class BaseModel(nn.Module):
    def __init__(self, filters, input_shape, num_of_patches=16, num_of_residual_blocks=10, probability_threshold=0.5,
                 iou_threshold=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.num_of_patches = num_of_patches
        assert input_shape[1] % num_of_patches == 0 and input_shape[2] % num_of_patches == 0, \
            f"Input shape {input_shape} cannot be divided into {num_of_patches} patches"
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=filters,
            kernel_size=(3, 3),
            padding="same"
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
            kernel_size=(3, 3),
            padding="same"
        )
        self.reduce_bounding_boxes = ReduceBoundingBoxes(
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
            input_shape=self.input_shape
        )

    def summary(self):
        if self.input_shape is None:
            raise Exception("Please set 'self.input_shape'")
        else:
            print(summary(self, (1, *self.input_shape)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_blocks(x)
        x = self.out(x)
        x = nn.Sigmoid()(x)
        # s = [self.reduce_bounding_boxes(xi) for xi in x[:]]
        return x


if __name__ == '__main__':
    input_shape = (3, 640, 480)
    bm = BaseModel(
        filters=64,
        input_shape=input_shape,
        num_of_patches=20,
        num_of_residual_blocks=10
    ).cuda()
    t = torch.cat([torch.rand(1, *input_shape) for _ in range(10)], 0).cuda()
    from time import time

    s = time()
    p = bm(t)
    print(time() - s)
    print(p.shape)
    # bm.summary()
    # i = 0
