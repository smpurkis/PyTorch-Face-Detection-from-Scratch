import torch
import torch.nn as nn
from torchinfo import summary
from torchvision.transforms import transforms

from datasets.utils import ReduceBoundingBoxes, ReduceSSDBoundingBoxes
from ptflops import get_model_complexity_info



class BaseSSDModel(nn.Module):
    def __init__(self, filters, input_shape, probability_threshold=0.5, iou_threshold=0.5, priors=None):
        super().__init__()
        self.input_shape = input_shape
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold

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
        if len(x.shape) == 3:
            return tuple([self.reduce_bounding_boxes(x[i]) for i in range(x.shape[0])])
        else:
            return self.reduce_bounding_boxes(x)

    def single_non_max_suppression(self, x):
        return self.reduce_bounding_boxes(x)

    @torch.no_grad()
    def predict(self, x, probability_threshold=0.5, iou_threshold=0.5):
        self.reduce_bounding_boxes = ReduceSSDBoundingBoxes(
            probability_threshold=probability_threshold,
            iou_threshold=iou_threshold,
            input_shape=self.input_shape,
            patch_sizes=self.patch_sizes
        )
        x = transforms.Resize(size=self.input_shape[1:])(x)
        x = x / 255.
        image = x
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)
        x = self(x)
        bbxs = self.non_max_suppression(x)
        return image, bbxs[0]
