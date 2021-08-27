import torch
import torch.nn as nn
import torchvision.models
from torchinfo import summary

from datasets.utils import ReduceBoundingBoxes


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


class BaseModel(nn.Module):
    def __init__(self, filters, input_shape, num_of_patches=16, num_of_residual_blocks=10, probability_threshold=0.1,
                 iou_threshold=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.num_of_patches = num_of_patches
        assert input_shape[1] % num_of_patches == 0 and input_shape[2] % num_of_patches == 0, \
            f"Input shape {input_shape} cannot be divided into {num_of_patches} patches"
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        # self.conv1 = nn.Conv2d(
        #     in_channels=input_shape[0],
        #     out_channels=filters,
        #     kernel_size=(3, 3),
        #     padding="same"
        # )
        # self.residual_blocks = nn.Sequential(
        #     *[ResidualBlock(
        #         filters=filters,
        #         num_of_patches=self.num_of_patches
        #     ) for _ in range(num_of_residual_blocks)]
        # )
        # self.out = nn.Conv2d(
        #     in_channels=filters,
        #     out_channels=5,
        #     kernel_size=(3, 3),
        #     padding="same"
        # )
        self.out = nn.LazyConv2d(
            out_channels=5 * filters,
            stride=2,
            kernel_size=(5, 5),
            padding=2
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        # self.linear1 = nn.LazyLinear(5 * num_of_patches ** 2)
        self.linear = nn.LazyLinear(5 * num_of_patches ** 2)
        self.reduce_bounding_boxes = ReduceBoundingBoxes(
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
            input_shape=self.input_shape,
            num_of_patches=self.num_of_patches
        )
        self.feature_extractor = nn.Sequential(*[l for l in list(torchvision.models.resnet18(pretrained=True).children())[:-2]])
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        self.dropout = nn.Dropout(0.75)
        self.dropout2d = nn.Dropout2d(0.75)

    def summary(self):
        if self.input_shape is None:
            raise Exception("Please set 'self.input_shape'")
        else:
            self(torch.rand((1, *self.input_shape)).cuda())
            print(summary(self, (1, *self.input_shape)))

    def non_max_suppression(self, x):
        if len(x.shape) == 4:
            return [self.reduce_bounding_boxes(xi) for xi in x[:]]
        else:
            return self.reduce_bounding_boxes(x)

    def forward(self, x):
        bs = x.size(0)
        # x = self.conv1(x)
        # x = self.residual_blocks(x)
        x = self.feature_extractor(x)
        x = self.dropout2d(x)
        x = self.out(x)
        x = self.leaky_relu(x)
        x = self.dropout2d(x)
        # x = nn.Sigmoid()(x)
        x = nn.Flatten()(x)
        # x = self.linear1(x)
        # x = self.dropout(x)
        x = self.linear(x).reshape(bs, 5, self.num_of_patches, self.num_of_patches)
        x = nn.Sigmoid()(x)
        # x = torch.abs(x)
        # s = [self.reduce_bounding_boxes(xi) for xi in x[:]]
        return x


if __name__ == '__main__':
    input_shape = (320, 320)
    bm = BaseModel(
        filters=64,
        input_shape=(3, *input_shape),
        num_of_patches=2,
        num_of_residual_blocks=10
    ).cuda()
    bm.eval()
    bm.summary()
    t = torch.rand(1, *(3, *input_shape)).cuda()
    from time import time

    s = time()
    p = bm(t)
    s = time() - s
    fps = 1/s
    print(s, fps)
    print(p.shape)
    # bm.summary()
    # i = 0
