from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torch
torch.set_printoptions(sci_mode=False)

class WIDERFaceDataset(Dataset):
    def __init__(self, data_dir, num_of_patches, targets=None, split: str = "train", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.targets = targets
        self.num_of_patches = num_of_patches

    def __len__(self):
        return len(self.targets)

    # def convert_bbx_to_feature_map(self, bbx, img_size):
    #     feature_map = fm = torch.empty((5, self.num_of_patches, self.num_of_patches))
    #     i = 0
    #     return feature_map

    def __getitem__(self, index):
        target = self.targets[index]
        img_path = target["img_path"]

        img = Image.open(img_path)
        original_img_size = img.size
        if self.transform:
            img = self.transform(img)

        bbx = target["bbx"]
        # fm = self.convert_bbx_to_feature_map(bbx, original_img_size)
        bbx[:, [1, 3]] = bbx[:, [1, 3]]/original_img_size[0]
        bbx[:, [2, 4]] = bbx[:, [2, 4]]/original_img_size[1]

        bbx[:, 3] = bbx[:, 1] + bbx[:, 3]
        bbx[:, 4] = bbx[:, 2] + bbx[:, 4]

        return img, bbx
