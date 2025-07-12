import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.io import read_image
import pandas as pd
import os
from typing import Optional
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, img_dir: str, annotation_file: str, img_shape, wide_range=False):
        self.img_dir = img_dir
        self.img_shape = img_shape
        self.files = pd.read_csv(annotation_file)
        if wide_range:
            self.transform = transforms.Compose([
                transforms.Resize(img_shape[1:]),
                transforms.Lambda(self.normalize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_shape[1:]),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files.iloc[idx][0])
        image = Image.open(img_path)
        if self.img_shape[0] == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        image = self.transform(image)
        return image

    def normalize(self, x):
        x = np.array(x)
        x = (x - 127.5) / 127.5
        return x.astype(np.float32)

class ImageDataModule(pl.LightningDataModule):
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ImageDataModule")
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--data_path", type=str, default="./datasets")
        parser.add_argument("--annotation_path", type=str, default="./datasets/train_.txt")
        parser.add_argument("--num_workers", type=int, default=12)
        parser.add_argument("--wide_range", action='store_true')
        return parent_parser

    def __init__(self, data_path: str, annotation_path: str, batch_size, num_workers, img_shape, wide_range):
    # def __init__(self, args):
        super().__init__()
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wide_range = wide_range

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            full_data = ImageDataset(self.data_path, self.annotation_path, img_shape=self.img_shape, wide_range=self.wide_range)
            self.full_size = len(full_data)
            self.train_size = int(self.full_size * 0.8)
            self.valid_size = self.full_size - self.train_size
            self.train_data, self.valid_data = random_split(full_data, [self.train_size, self.valid_size], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers)
