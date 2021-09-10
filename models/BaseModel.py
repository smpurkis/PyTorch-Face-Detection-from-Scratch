import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
from torchinfo import summary
from torchvision.transforms import transforms

from datasets.utils import ReduceBoundingBoxes

class BaseModel(nn.Module):
    def __init__(self, filters, input_shape, num_of_patches=16, probability_threshold=0.5, iou_threshold=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.num_of_patches = num_of_patches
        assert input_shape[1] % num_of_patches == 0 and input_shape[2] % num_of_patches == 0, \
            f"Input shape {input_shape} cannot be divided into {num_of_patches} patches"
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.reduce_bounding_boxes = ReduceBoundingBoxes(
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
            input_shape=self.input_shape,
            num_of_patches=self.num_of_patches
        )

    def summary(self):
        if self.input_shape is None:
            raise Exception("Please set 'input_shape'")
        else:
            self(torch.rand((1, *self.input_shape)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            print(summary(self, (1, *self.input_shape)))

    def non_max_suppression(self, x):
        if len(x.shape) == 4:
            return tuple([self.reduce_bounding_boxes(x[i]) for i in range(x.shape[0])])
        else:
            return self.reduce_bounding_boxes(x)

    def single_non_max_suppression(self, x):
        return self.reduce_bounding_boxes(x)

    @torch.no_grad()
    def predict(self, x, probability_threshold=0.5, iou_threshold=0.5):
        self.reduce_bounding_boxes = ReduceBoundingBoxes(
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
            input_shape=self.input_shape,
            num_of_patches=self.num_of_patches
        )
        x = transforms.Resize(size=self.input_shape[1:])(x)
        x = x / 255.
        image = x
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)
        x = self(x)
        bbxs = self.non_max_suppression(x)
        return image, bbxs[0]
