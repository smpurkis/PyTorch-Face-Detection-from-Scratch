import math

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import draw_bbx
from models.BaseModel import ReduceBoundingBoxes
from torchvision.transforms import transforms
torch.set_printoptions(sci_mode=False)


class WIDERFaceDataset(Dataset):
    def __init__(self, data_dir, num_of_patches, input_shape, targets=None, split: str = "train", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.targets = targets
        self.num_of_patches = num_of_patches
        self.input_shape = input_shape

    def __len__(self):
        # return 2
        return len(self.targets)//4

    def convert_bbx_to_feature_map(self, bbx, img_size):
        feature_map = fm = torch.zeros((5, self.num_of_patches, self.num_of_patches))
        width, height = img_size

        # Calculate the number of pixels in height and width to get the desired number of patches
        x_patch_size, y_patch_size = width / self.num_of_patches, height / self.num_of_patches
        for k, bx in enumerate(bbx):
            # Get position indices to place the bbx at
            i, j = math.floor(bx[1] / x_patch_size), math.floor(bx[2] / y_patch_size)

            # Calculate the x0, y0, x1, y1 coordinates relative to the top left corner of its containing ij box
            # normalized_bx = self.convert_to_xyxy(bx)
            normalized_bx = torch.clone(bx)

            # normalized_bx[[1, 3]] = normalized_bx[[1, 3]] - i * x_patch_size
            # normalized_bx[[2, 4]] = normalized_bx[[2, 4]] - j * y_patch_size
            normalized_bx[1] = normalized_bx[1] - i * x_patch_size
            normalized_bx[2] = normalized_bx[2] - j * y_patch_size

            # Normalize by the size of the image
            normalized_bx[1] = normalized_bx[1] / x_patch_size
            normalized_bx[2] = normalized_bx[2] / y_patch_size

            normalized_bx[3] = normalized_bx[3] / width
            normalized_bx[4] = normalized_bx[4] / height
            # normalized_bx[3] = normalized_bx[3] / x_patch_size
            # normalized_bx[4] = normalized_bx[4] / y_patch_size

            i = min(max(i, 0), self.num_of_patches - 1)
            j = min(max(j, 0), self.num_of_patches - 1)
            fm[:, i, j] = normalized_bx
        return feature_map

    def convert_to_xyxy(self, x):
        if len(x.shape) == 2:
            x[:, 3] = x[:, 3] + x[:, 1]
            x[:, 4] = x[:, 4] + x[:, 2]
        else:
            x = torch.tensor([x[0], x[1], x[2], x[3] + x[1], x[4] + x[2]])
        return x

    def convert_bbx_to_transform_format(self, bbx):
        bbx_formatted = []
        for b in bbx:
            bbx_formatted.append([*b[1:].tolist(), "face"])
        return bbx_formatted

    def convert_transform_format_to_bbx(self, bbx_formatted):
        if len(bbx_formatted) > 0:
            bbx = torch.vstack([torch.round(torch.tensor([1.0, *b[:-1]], dtype=torch.float32)) for b in bbx_formatted])
        else:
            bbx = torch.tensor([])
        return bbx

    def __getitem__(self, index):
        target = self.targets[index]
        bbx_og = target["bbx"]
        if len(torch.where(bbx_og == 0)[0]) == 4:
            target = self.targets[index-1]
            bbx_og = target["bbx"]
        img_path = target["img_path"]
        img_og = Image.open(img_path)
        original_img_size = img_og.size

        if self.transform:
            transformed_data = self.transform(image=np.array(img_og), bboxes=self.convert_bbx_to_transform_format(bbx_og))
        bbx = self.convert_transform_format_to_bbx(transformed_data["bboxes"])
        img = transformed_data["image"]
        # draw_bbx(img, bbx, show=True)

        # bbx = self.convert_batch_to_xywh(target["bbx"])
        # bbx = target["bbx"]
        # if index == 0:
        # bbx2 = torch.clone(bbx)
        # bbx2[:, [1, 3]] = torch.round(bbx2[:, [1, 3]] * self.input_shape[0] / original_img_size[0])
        # bbx2[:, [2, 4]] = torch.round(bbx2[:, [2, 4]] * self.input_shape[1] / original_img_size[1])
        # draw_bbx(img, bbx2, self.input_shape, show=True)
        # draw_bbx(img_og, bbx, original_img_size)

        fm = self.convert_bbx_to_feature_map(bbx, original_img_size)
        # draw_bbx(img_og, fm, original_img_size, show=True)

        # reduce_bounding_boxes = ReduceBoundingBoxes(0.5, 0.1, (3, *original_img_size), self.num_of_patches)
        # s = reduce_bounding_boxes(torch.clone(fm))
        # # draw_bbx(img, s, self.input_shape, show=True)
        #
        # b = torch.sort(bbx, dim=0)
        # bb = torch.sort(s, dim=0)
        #
        # try:
        #     torch.all(b.values == bb.values)
        # except Exception as e:
        #     print(index, len(bbx), len(s), len(bbx) - len(s))
        #
        # assert torch.all(b.values == bb.values)


        # normalization_transform = transforms.Compose([
        #     transforms.Normalize(mean=0.5, std=0.25)
        # ])
        # img = normalization_transform(img/255)

        img = img/255
        return img, fm, bbx