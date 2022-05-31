import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from datasets.utils import ReduceSSDBoundingBoxes
from models.BaseSSDModel import BaseSSDModel

torch.set_printoptions(sci_mode=False)


class SeparableResidualBlock(nn.Module):
    def __init__(
        self, in_filters, out_filters, dropout=0.25, use_max_pool=False, bias=True
    ):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.filters_equal = in_filters == out_filters
        self.allow_groups = in_filters % out_filters == 0
        if not self.filters_equal:
            self.pointwise_conv_skip = nn.Conv2d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=(1, 1),
                padding=0,
                bias=bias,
            )
        # self.depthwise_conv = nn.Conv2d(
        #     in_channels=in_filters,
        #     out_channels=out_filters,
        #     kernel_size=(3, 3),
        #     padding=1,
        #     groups=in_filters,
        #     bias=bias
        # )
        # self.pointwise_conv = nn.Conv2d(
        #     in_channels=out_filters,
        #     out_channels=out_filters,
        #     kernel_size=(1, 1),
        #     padding=0,
        #     bias=bias
        # )
        self.conv1 = nn.Conv2d(
            in_channels=in_filters,
            out_channels=out_filters,
            kernel_size=(3, 3),
            padding=1,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_filters,
            out_channels=out_filters,
            kernel_size=(3, 3),
            padding=1,
            bias=bias,
        )
        self.use_max_pool = use_max_pool
        self.max_pool = nn.MaxPool2d(2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout2d = nn.Dropout2d(dropout)

    def forward(self, x):
        if self.filters_equal:
            skip_x = x
        else:
            skip_x = self.pointwise_conv_skip(x)
        # x = self.depthwise_conv(x)
        # x = self.leaky_relu(x)
        # x = self.pointwise_conv(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout2d(x)
        x = x + skip_x
        if self.use_max_pool:
            x = self.max_pool(x)
        return x


class SSD(BaseSSDModel):
    def __init__(
        self,
        filters,
        input_shape,
        probability_threshold=0.5,
        iou_threshold=0.5,
        priors=None,
    ):
        super().__init__(
            filters,
            input_shape,
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
        )
        self.patch_sizes = (60, 30, 15, 7)
        # self.patch_sizes = (2,)
        _, self.height, self.width = input_shape

        # regular functions
        self.dropout2d = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.resize = transforms.Resize(size=self.input_shape[1:])
        self.min_filters = filters
        self.max_filters = 16 * filters
        self.multiply_priors = torch.unsqueeze(
            torch.cat(
                [torch.tensor(1 / ps).repeat(ps * ps) for ps in self.patch_sizes]
            ),
            dim=1,
        )
        if priors:
            self.priors = priors
        else:
            self.priors = self.calculate_priors()
        self.reduce_bounding_boxes = ReduceSSDBoundingBoxes(
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
            input_shape=self.input_shape,
            patch_sizes=self.patch_sizes,
            priors=self.priors,
        )

        self.input_normalizer = nn.Conv2d(
            in_channels=3,
            out_channels=filters,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=True,
        )
        self.feature_extractor = nn.Sequential(
            SeparableResidualBlock(
                in_filters=filters, out_filters=2 * filters, use_max_pool=True
            ),
            SeparableResidualBlock(
                in_filters=2 * filters, out_filters=2 * filters, use_max_pool=True
            ),
            SeparableResidualBlock(
                in_filters=2 * filters, out_filters=2 * filters, use_max_pool=False
            ),
            SeparableResidualBlock(
                in_filters=2 * filters, out_filters=2 * filters, use_max_pool=False
            ),
            SeparableResidualBlock(
                in_filters=2 * filters, out_filters=2 * filters, use_max_pool=False
            ),
            SeparableResidualBlock(
                in_filters=2 * filters, out_filters=2 * filters, use_max_pool=False
            ),
            SeparableResidualBlock(
                in_filters=2 * filters, out_filters=2 * filters, use_max_pool=False
            ),
            SeparableResidualBlock(
                in_filters=2 * filters, out_filters=2 * filters, use_max_pool=False
            ),
            SeparableResidualBlock(
                in_filters=2 * filters, out_filters=4 * filters, use_max_pool=False
            ),
        )
        continue_layers = []
        extracting_layers = []
        for i, ps in enumerate(self.patch_sizes):
            in_filters = 4 * filters * (2**i)
            in_filters = (
                in_filters if in_filters <= self.max_filters else self.max_filters
            )
            out_filters = 2 * in_filters
            out_filters = (
                out_filters if out_filters <= self.max_filters else self.max_filters
            )

            continue_layer = nn.Sequential(
                SeparableResidualBlock(
                    in_filters=in_filters,
                    out_filters=out_filters,
                    use_max_pool=False if i == 0 else True,
                )
            )
            extracting_layer = nn.Sequential(
                nn.Linear(in_features=out_filters, out_features=5),
            )
            continue_layers.append(continue_layer)
            extracting_layers.append(extracting_layer)
        self.continue_layers = nn.ModuleList(continue_layers)
        self.extracting_layers = nn.ModuleList(extracting_layers)
        self.avg_pooling = nn.AdaptiveAvgPool2d(2)

    def calculate_priors(self):
        priors_list = []
        for ps in self.patch_sizes:
            priors = torch.zeros((4, ps, ps))
            i, j = torch.where(priors[0] >= 0)
            priors[0, i, j] = priors[0, i, j] + 1 / ps * i
            priors[1, i, j] = priors[1, i, j] + 1 / ps * j
            priors[2, i, j] = priors[2, i, j]
            priors[3, i, j] = priors[3, i, j]
            priors = priors.permute(1, 2, 0).reshape(ps * ps, 4)
            priors_list.append(priors)
        priors = torch.cat(priors_list, dim=0)
        return priors

    def apply_priors(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        self.multiply_priors = self.multiply_priors.to(x.device)
        self.priors = self.priors.to(x.device)
        scaled_x = torch.clone(x).float()
        scaled_x[..., 1:2] = scaled_x[..., 1:2] * self.multiply_priors.repeat(
            repeats=(bs, 1, 1)
        )
        scaled_x[..., 2:3] = scaled_x[..., 2:3] * self.multiply_priors.repeat(
            repeats=(bs, 1, 1)
        )
        scaled_x[..., 1:5] = scaled_x[..., 1:5] + self.priors.repeat(repeats=(bs, 1, 1))
        # scaled_x[..., [1, 3]] = scaled_x[..., [1, 3]] * self.width
        # scaled_x[..., [2, 4]] = scaled_x[..., [2, 4]] * self.height
        return scaled_x

    def forward(self, x: torch.Tensor, predict: torch.Tensor = torch.tensor(0)):
        bs = x.size(0)
        if predict == 1:
            x = self.resize(x) / 255.0
            if len(x.shape) == 3:
                x = torch.unsqueeze(x, 0)
        x = self.input_normalizer(x)
        if torch.any(torch.isnan(x)):
            p = 0
        x = self.feature_extractor(x)
        if torch.any(torch.isnan(x)):
            p = 0
        scores, bbxs = [], []
        for i in range(len(self.continue_layers)):
            x = self.continue_layers[i](x)
            # x = self.avg_pooling(x)
            if torch.any(torch.isnan(x)):
                p = 0
            z = self.extracting_layers[i](x.permute(0, 2, 3, 1).contiguous())
            z = z.reshape(bs, -1, 5)
            # z = z[:, :4, :]
            scores.append(z[..., :1])
            bbxs.append(z[..., 1:5])
        scores = self.sigmoid(torch.cat(scores, dim=1))
        # bbxs = self.sigmoid(torch.cat(bbxs, dim=1))
        bbxs = torch.cat(bbxs, dim=1)
        x = torch.cat([scores, bbxs], dim=2)
        x = self.apply_priors(x)
        # print("\n", scores.min(), scores.mean(), scores.max())
        if torch.any(torch.isnan(x)):
            p = 0
        if predict == 1:
            x = self.non_max_suppression(x)
        return x


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    input_shape = (480, 480)
    bm = SSD(
        filters=16,
        input_shape=(3, *input_shape),
    ).cpu()
    bm.eval()
    bm.summary()
    t = torch.rand(1, *(3, *input_shape)).cpu()
    from time import time

    s = time()
    p = bm(t)
    s = time() - s
    fps = 1 / s
    print(s, fps)
    # print(p.shape)
    # bm.summary()
    # i = 0
