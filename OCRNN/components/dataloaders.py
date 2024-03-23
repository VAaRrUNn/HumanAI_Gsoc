import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
# Load model directly
from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


class LoadData(Dataset):
    def __init__(self,
                 images,
                 labels,
                 transforms=None,
                 tokenizer=None):
        super().__init__()
        self.images = images
        self.labels = labels

        self.transforms = transforms
        self.tokenizer = tokenizer

    def __len__(self,) -> int:
        return len(self.images)

    def __getitem__(self,
                    idx) -> Tuple:
        img, lab = Image.open(self.images[idx]), self.labels[idx]
        if self.transforms:
            img = self.transforms(img)

        # The padding will be done in coll_fn in dataloader
        lab = self.tokenizer(lab)

        return (img, lab)


class OCRDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 csv_file,
                 transforms,
                 batch_size=5,
                 tokenizer=None,
                 data_collator=None):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.csv_file = csv_file

        self.transforms = transforms
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    def prepare_data(self):
        """
        Some Preprocessing, currently not implemented fully yet
        """
        images_name, self.labels = [], []
        with open("dataset.csv", 'r') as f:
            content = csv.reader(f)

            # Skipping 1st row
            next(content)

            for row in content:
                images_name.append(row[0])
                self.labels.append(row[1])

        # Getting absolute path for images
        self.images = [
            self.data_dir / Path("images") / x for x in images_name
        ]

    def setup(self,
              stage: str = "train"):
        if stage == "fit" or "validate":
            self.data = LoadData(images=self.images,
                                 labels=self.labels,
                                 transforms=self.transforms,
                                 tokenizer=self.tokenizer)

        if stage == "test":
            """
            Currently not implemented, because the (current) dataset size is very small
            """
            pass

    def __collate_fn(self,
                     samples) -> Tuple:
        images, labels = [], []
        for (img, lab) in samples:
            images.append(img)
            labels.append(lab)
        labels = self.data_collator(labels)
        return (images, labels)

    def train_dataloader(self):
        return DataLoader(dataset=self.data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.__collate_fn)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


# Loading spanish tokenizer

tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = OCRDataModule(data_dir=".",
                        csv_file="dataset.csv",
                        transforms=train_transform,
                        tokenizer=tokenizer,
                        data_collator=data_collator)
