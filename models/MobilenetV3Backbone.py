import os

import timm
import torch
import torch.nn as nn
from torchvision.transforms import transforms

from models import BaseModel


class MobilenetV3Backbone(BaseModel):
    def __init__(
        self,
        filters,
        input_shape,
        num_of_patches,
        probability_threshold=0.5,
        iou_threshold=0.5,
        pretrained=True,
        input_kernel_size=10,
        input_stride=8,
        output_kernel_size=3,
        output_padding=0,
    ):
        super().__init__(
            filters,
            input_shape,
            num_of_patches=num_of_patches,
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
        )
        self.pretrained = pretrained
        self.feature_extractor = torch.nn.Sequential(
            *list(
                timm.create_model(
                    "tf_mobilenetv3_small_100", pretrained=pretrained
                ).children()
            )[:-5]
        )
        self.out = nn.Conv2d(
            in_channels=576,
            out_channels=5,
            stride=(1, 1),
            kernel_size=(output_kernel_size, output_kernel_size),
            padding=1,
        )
        self.sigmoid = nn.Sigmoid()
        self.resize = transforms.Resize(size=self.input_shape[1:])

    def forward(self, x: torch.Tensor, predict: torch.Tensor = torch.tensor(0)):
        if predict == 1:
            x = self.resize(x) / 255.0
            if len(x.shape) == 3:
                x = torch.unsqueeze(x, 0)
        x = self.feature_extractor(x)
        x = self.out(x)
        x = self.sigmoid(x)
        if predict == 1:
            x = self.single_non_max_suppression(x[0])
        return x


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    input_shape = (480, 480)
    bm = MobilenetV3Backbone(
        filters=64,
        input_shape=(3, *input_shape),
        num_of_patches=15,
        num_of_residual_blocks=10,
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
    print(p.shape)
    # bm.summary()
    # i = 0
