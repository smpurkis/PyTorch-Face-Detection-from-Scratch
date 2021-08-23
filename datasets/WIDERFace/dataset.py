import math

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import draw_bbx
from models.BaseModel import ReduceBoundingBoxes

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
            normalized_bx = torch.tensor([
                bx[0],
                bx[1],
                bx[2],
                (bx[1] + bx[3]),
                (bx[2] + bx[4]),
            ])

            normalized_bx[[1, 3]] = normalized_bx[[1, 3]] - i * x_patch_size
            normalized_bx[[2, 4]] = normalized_bx[[2, 4]] - j * y_patch_size

            # Normalize by the size of the image
            normalized_bx[1] = normalized_bx[1] / x_patch_size
            normalized_bx[2] = normalized_bx[2] / y_patch_size

            normalized_bx[3] = normalized_bx[3] / width
            normalized_bx[4] = normalized_bx[4] / height

            i = min(max(i, 0), self.num_of_patches-1)
            j = min(max(j, 0), self.num_of_patches-1)
            fm[:, i, j] = normalized_bx
        return feature_map

    def __getitem__(self, index):
        target = self.targets[index]
        img_path = target["img_path"]

        img_og = Image.open(img_path)
        original_img_size = img_og.size
        if self.transform:
            img = self.transform(img_og)

        bbx = target["bbx"]
        # if index == 0:
        bbx2 = torch.clone(bbx)
        bbx2[:, [1, 3]] = torch.round(bbx2[:, [1, 3]] * self.input_shape[0] / original_img_size[0])
        bbx2[:, [2, 4]] = torch.round(bbx2[:, [2, 4]] * self.input_shape[1] / original_img_size[1])
        # print(bbx2)
        # draw_bbx(img, bbx2, self.input_shape)
        # draw_bbx(img_og, bbx, original_img_size)

        fm = self.convert_bbx_to_feature_map(bbx, original_img_size)
        # draw_bbx(img_og, fm, original_img_size)

        # reduce_bounding_boxes = ReduceBoundingBoxes(0.5, 0.1, (3, *self.input_shape), self.num_of_patches)
        # s = reduce_bounding_boxes(torch.clone(fm))
        # draw_bbx(img, s, self.input_shape)

        # b = torch.sort(bbx, dim=0)
        # bb = torch.sort(s, dim=0)
        #
        # try:
        #     torch.all(b.values == bb.values)
        # except Exception as e:
        #     print(index, len(bbx), len(s), len(bbx) - len(s))
        #
        # assert torch.all(b.values == bb.values)

        return img, fm, bbx2
