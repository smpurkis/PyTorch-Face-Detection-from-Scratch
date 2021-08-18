from multiprocessing import cpu_count
from pathlib import Path

import gdown
import pytorch_lightning as pl
import torch
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.WIDERFace.dataset import WIDERFaceDataset

dataset_links = {
    "train": {
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=AB-4&id=0B6eKvaijfFUDQUUwd21EckhUbWs",
        "output": "WIDER_train.zip"
    },
    "val": {
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=aVur&id=0B6eKvaijfFUDd3dIRmpvSk8tLUk",
        "output": "WIDER_val.zip"
    },
    "test": {
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=7vAN&id=0B6eKvaijfFUDbW4tdGpaYjgzZkU",
        "output": "WIDER_test.zip"
    },
    "target": {
        "url": "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip",
        "output": "WIDER_train.zip"
    },
}


class WIDERFaceDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str = "./", input_shape=(320, 320), num_of_patches: int = 20, batch_size: int = 8,
                 shuffle: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.data_dir = Path(self.root_dir, "data")
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_of_patches = num_of_patches

    def data_folder(self, output, zipfile: bool = False):
        return Path(self.data_dir, output if zipfile else Path(output).stem)

    def check_folder_and_zip_exist(self, output):
        check_zip = self.data_folder(output, zipfile=True).exists()
        check_folder = self.data_folder(output).exists()
        return check_zip or check_folder

    def download_dataset_files(self):
        for split in dataset_links.keys():
            if not self.check_folder_and_zip_exist(dataset_links[split]["output"]):
                gdown.cached_download(
                    url=dataset_links[split]["url"],
                    path=self.data_folder(dataset_links[split]["output"]),
                    postprocess=gdown.extractall
                )

    @staticmethod
    def convert_bbx_to_xyxy(bbx):
        return bbx[1], bbx[2], bbx[1] + bbx[3], bbx[2] + bbx[4]

    def draw_bbx(self, img, bbx):
        draw = ImageDraw.Draw(img)
        for b in bbx:
            draw.rectangle(WIDERFaceDataModule().convert_bbx_to_xyxy(b), outline="blue", width=2)
        return draw

    def get_targets(self, split: str = "train"):
        assert self.check_folder_and_zip_exist(
            dataset_links["target"]["output"]), "Target files/folder is missing, please download it!"
        lines = Path(self.data_dir, f"wider_face_split/wider_face_{split}_bbx_gt.txt").read_text().split("\n")
        targets = []
        target = {}
        for line_no, line in enumerate(lines):
            if len(line) == 0:
                continue
            if line[-3:] == "jpg":
                if line_no > 1:
                    targets.append(target)
                img_path = Path(self.data_dir, f"WIDER_{split}", "images", line)
                assert img_path.exists(), "Image for this target does not exist, please download it!"
                target = {"img_path": img_path,
                          "number_faces": 0,
                          "bbx": []}
            else:
                if len(line.split()) == 1:
                    target["number_faces"] = int(line)
                else:
                    target["bbx"].append([float(l) for l in (1, *line.split()[:4])])
        targets.append(target)
        for target in targets:
            target["bbx"] = torch.tensor(target["bbx"])
        return targets

    def default_transform(self):
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=self.input_shape)
        ])
        return default_transform

    def setup(self, stage=None):
        self.train_dataset = WIDERFaceDataset(
            data_dir=self.data_dir,
            split="train",
            num_of_patches=self.num_of_patches,
            transform=self.default_transform(),
            input_shape=self.input_shape,
            targets=self.get_targets(split="train")
        )
        self.val_dataset = WIDERFaceDataset(
            data_dir=self.data_dir,
            split="val",
            num_of_patches=self.num_of_patches,
            transform=self.default_transform(),
            input_shape=self.input_shape,
            targets=self.get_targets(split="val")
        )
        self.test_dataset = WIDERFaceDataset(
            data_dir=self.data_dir,
            split="test",
            num_of_patches=self.num_of_patches,
            input_shape=self.input_shape,
            transform=self.default_transform(),
        )

    def my_collate(self, batch):
        data = torch.stack([item[0] for item in batch])
        target = [item[1] for item in batch]
        return [data, target]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.my_collate,
            num_workers=cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.my_collate,
            num_workers=cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.my_collate,
            num_workers=cpu_count()
        )

    def teardown(self, stage=None):
        pass


if __name__ == '__main__':
    dm = WIDERFaceDataModule(
        "/home/sam/PycharmProjects/python/PyTorch-Face-Detection-from-Scratch",
        num_of_patches=300
    )
    dm.setup()
    x, y = dm.train_dataset[2]
    draw = dm.draw_bbx(x, y)
    x.show()
    i = 0
