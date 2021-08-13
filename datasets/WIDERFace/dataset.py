from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class WIDERFaceDataset(Dataset):
    def __init__(self, data_dir, targets=None, split: str = "train", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        bbx = target["bbx"]
        img_path = target["img_path"]

        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        return img, bbx
