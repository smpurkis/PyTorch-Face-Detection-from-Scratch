import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
from ptflops import get_model_complexity_info
from torchinfo import summary
from torchvision.transforms import transforms

from datasets.utils import ReduceBoundingBoxes, SSDReduceBoundingBoxes


class SSDBaseModel(nn.Module):
    def __init__(self, filters, input_shape, num_of_patches, probability_threshold=0.5, iou_threshold=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.num_of_patches = num_of_patches
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.reduce_bounding_boxes = SSDReduceBoundingBoxes(
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
            input_shape=self.input_shape,
        )

    def summary(self, *args, **kwargs):
        if self.input_shape is None:
            raise Exception("Please set 'input_shape'")
        else:
            self(torch.rand((1, *self.input_shape)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            print(summary(self, (1, *self.input_shape), *args, **kwargs))
        macs, params = get_model_complexity_info(self, self.input_shape, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    def non_max_suppression(self, x):
        fm_batch_split = []
        bs = x[0].size(0)
        for i in range(bs):
            z = []
            for j in range(len(x)):
                z.append(x[j][i])
            fm_batch_split.append(z)
        filtered_bbxs = []
        for fm_set in fm_batch_split:
            bbxs = self.reduce_bounding_boxes(fm_set, self.num_of_patches)
            filtered_bbxs.append(bbxs)
        return filtered_bbxs

    def single_non_max_suppression(self, x):
        return self.reduce_bounding_boxes([z[0] for z in x], self.num_of_patches)

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
