import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import ReduceSSDBoundingBoxes

torch.set_printoptions(sci_mode=False)


class WIDERFaceDatasetSSD(Dataset):
    def __init__(
            self,
            data_dir,
            num_of_patches,
            input_shape,
            targets=None,
            split: str = "train",
            transform=None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.targets = targets
        self.num_of_patches = num_of_patches
        self.input_shape = input_shape
        # self.patch_sizes = (60, 30, 15, 7)
        self.patch_sizes = (2,)

    def __len__(self):
        # return 1
        return len(self.targets)//8

    def convert_bbx_to_feature_map(self, bbx, img_size, patch_size):
        feature_map = fm = torch.zeros((5, patch_size, patch_size))
        width, height = img_size
        if len(bbx.shape) == 1:
            return feature_map
        bbx = torch.clone(bbx)
        bbx[:, [1, 3]] = bbx[:, [1, 3]] / width
        bbx[:, [2, 4]] = bbx[:, [2, 4]] / height

        # Calculate the number of pixels in height and width to get the desired number of patches
        x_patch_size, y_patch_size = (
            1 / patch_size,
            1 / patch_size,
        )
        for k, bx in enumerate(bbx):
            # Get position indices to place the bbx at
            i, j = math.floor(bx[1] / x_patch_size), math.floor(bx[2] / y_patch_size)

            # Calculate the x0, y0, x1, y1 coordinates relative to the top left corner of its containing ij box
            # normalized_bx = self.convert_to_xyxy(bx)
            normalized_bx = torch.clone(bx)

            # Ensure that the smaller bounding boxes have the highest score
            normalized_bx[0] = normalized_bx[0] - 0.001 * patch_size

            # normalized_bx[[1, 3]] = normalized_bx[[1, 3]] - i * x_patch_size
            # normalized_bx[[2, 4]] = normalized_bx[[2, 4]] - j * y_patch_size
            normalized_bx[1] = normalized_bx[1] - i * x_patch_size
            normalized_bx[2] = normalized_bx[2] - j * y_patch_size

            # Normalize by the size of the image
            normalized_bx[1] = normalized_bx[1] / x_patch_size
            normalized_bx[2] = normalized_bx[2] / y_patch_size

            # normalized_bx[3] = normalized_bx[3] / width
            # normalized_bx[4] = normalized_bx[4] / height

            i = min(max(i, 0), patch_size - 1)
            j = min(max(j, 0), patch_size - 1)
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
            box = b[1:].tolist()
            bbx_formatted.append([*box, "face"])
        return bbx_formatted

    def convert_transform_format_to_bbx(self, bbx_formatted):
        if len(bbx_formatted) > 0:
            bbx = torch.vstack(
                [
                    torch.round(torch.tensor([1.0, *b[:-1]], dtype=torch.float32))
                    for b in bbx_formatted
                ]
            )
        else:
            bbx = torch.tensor([])
        return bbx

    def __getitem__(self, index):
        try:
            target = self.targets[index]
            bbx_og = target["bbx"]
            if len(torch.where(bbx_og == 0)[0]) == 4:
                target = self.targets[index - 1]
                bbx_og = target["bbx"]
            img_path = target["img_path"]
            # print(img_path)
            img_og = Image.open(img_path)
            original_img_size = img_og.size

            if self.transform:
                transformed_data = self.transform(
                    image=np.array(img_og),
                    bboxes=self.convert_bbx_to_transform_format(bbx_og),
                )
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
            feature_maps = []
            for patch_size in self.patch_sizes:
                fm = self.convert_bbx_to_feature_map(bbx, self.input_shape, patch_size)
                fm = fm.permute(1, 2, 0).reshape(-1, 5)
                feature_maps.append(fm)
            feature_map = torch.cat(feature_maps, dim=0)
            # draw_bbx(img_og, fm, original_img_size, show=True)

            reduce_bounding_boxes = ReduceSSDBoundingBoxes(
                0.5, 0.5, (3, *self.input_shape), self.patch_sizes, with_priors=True
            )
            s = reduce_bounding_boxes(torch.clone(feature_map))
            # # draw_bbx(img, s, self.input_shape, show=True)
            #
            b = torch.sort(bbx, dim=0)
            bb = torch.sort(s, dim=0)
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

            img = img / 255
            return img, feature_map, bbx
        except Exception as e:
            Path("incorrect_indices.log").open("a").write(f"{index}, {img_path}\n")
            return self[index - 1 if index != 0 else 0]
