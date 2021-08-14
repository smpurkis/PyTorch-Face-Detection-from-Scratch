import torch
import torch.nn as nn
from torchinfo import summary
from torchvision.ops import nms


class ResidualBlock(nn.Module):
    def __init__(self, filters, patch_size):
        super().__init__()
        self.patch_size = patch_size
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
        if x.shape[2]-1 > self.patch_size:
            x = self.max_pool(x)
        return x


class ReduceBoundingBoxes(nn.Module):
    def __init__(self, probability_threshold: float = 0.9, iou_threshold: float = 0.5):
        super().__init__()
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold

    def remove_low_probabilty_bbx(self, x):
        above_threshold_indices = ati = torch.where(x[0] > self.probability_threshold)
        if len(ati[0]) == 0:
            return torch.empty([0]), False
        bbx = x[:, ati[0], ati[1]].permute(1, 0)
        bbx[:, 2] = bbx[:, 0] + bbx[:, 2]
        bbx[:, 3] = bbx[:, 1] + bbx[:, 3]
        return bbx, True

    def forward(self, x):
        x, boxes_exist = self.remove_low_probabilty_bbx(x)
        if boxes_exist:
            bbx = x[:, 1:]
            scores = x[:, 0]
            bbxis = nms(boxes=bbx, scores=scores, iou_threshold=self.iou_threshold)
            return x[bbxis]
        else:
            return torch.empty(0)


class BaseModel(nn.Module):
    def __init__(self, filters, input_shape, num_of_patches=16, num_of_residual_blocks=10, probability_threshold=0.5,
                 iou_threshold=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.num_of_patches = num_of_patches
        assert input_shape[
                   2] % num_of_patches == 0, f"Input shape {input_shape} cannot be divided into {num_of_patches} patches"
        self.patch_size = input_shape[2] // num_of_patches
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=filters,
            kernel_size=(3, 3),
            padding="same"
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(filters=filters, patch_size=self.patch_size) for _ in range(num_of_residual_blocks)]
        )
        self.out = nn.Conv2d(
            in_channels=filters,
            out_channels=5,
            kernel_size=(3, 3),
            padding="same"
        )
        self.reduce_bounding_boxes = ReduceBoundingBoxes(
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold
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
        s = [self.reduce_bounding_boxes(xi) for xi in x[:]]
        return s


if __name__ == '__main__':
    bm = BaseModel(64, (3, 320, 240), 10, 10).cuda()
    t = torch.cat([torch.rand(1, 3, 320, 240) for _ in range(10)], 0).cuda()
    from time import time
    s = time()
    p = bm(t)
    print(time() - s)
    # print(p)
    # bm.summary()
    # i = 0
