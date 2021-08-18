import math

import torch
from PIL import Image
from torch.utils.data import Dataset

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
        return len(self.targets)

    def convert_bbx_to_feature_map(self, bbx, img_size):
        feature_map = fm = torch.zeros((5, self.num_of_patches, self.num_of_patches))
        width, height = img_size

        # Calculate the number of pixels in height and width to get the desired number of patches
        x_patch_size, y_patch_size = math.floor(width / self.num_of_patches), math.floor(height / self.num_of_patches)
        for bx in bbx:
            # Get position indices to place the bbx at
            i, j = math.floor(bx[1] / x_patch_size), math.floor(bx[2] / y_patch_size)

            # Calculate the x0, y0, x1, y1 coordinates relative to the top left corner of its containing ij box
            normalized_bx = torch.tensor([
                bx[0],
                bx[1] - i * x_patch_size,
                bx[2] - j * y_patch_size,
                (bx[1] + bx[3]) - i * x_patch_size,
                (bx[2] + bx[4]) - j * y_patch_size,
            ])

            # Normalize by the size of the image
            normalized_bx[[1, 3]] = normalized_bx[[1, 3]] / width
            normalized_bx[[2, 4]] = normalized_bx[[2, 4]] / height

            fm[:, i, j] = normalized_bx
        return feature_map

    def __getitem__(self, index):
        target = self.targets[index]
        img_path = target["img_path"]

        img = Image.open(img_path)
        original_img_size = img.size
        if self.transform:
            img = self.transform(img)

        bbx = target["bbx"]
        fm = self.convert_bbx_to_feature_map(bbx, original_img_size)
        # reduce_bounding_boxes = ReduceBoundingBoxes(0.9, 0.5, (3, *original_img_size), self.num_of_patches)
        # s = reduce_bounding_boxes(fm)
        # b = torch.sort(bbx, dim=0)
        # bb = torch.sort(s, dim=0)
        return img, bbx
