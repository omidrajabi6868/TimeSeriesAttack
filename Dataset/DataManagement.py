from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from PIL import Image
from PIL.Image import Resampling
import torch
import numpy as np
import os
 
class ImageDataSet(TorchDataset):
    def __init__(self, label_path, transform=None, image_size=None):
        self.label_path = label_path
        self.transform = transform
        self.image_size = self._validate_image_size(image_size)
        self.image_paths, self.labels = self.solve_paths(self.label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.image_size is not None:
            image = image.resize(self.image_size, Resampling.BILINEAR)

        image = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label
    
    def train_val_test_loader(self,
                              train_ratio=0.7,
                              val_ratio=0.15,
                              test_ratio=0.15,
                              batch_size=32,
                              shuffle_train=True,
                              num_workers=0,
                              seed=42):
        if not torch.isclose(torch.tensor(train_ratio + val_ratio + test_ratio), torch.tensor(1.0)):
            raise ValueError('train_ratio, val_ratio and test_ratio must sum to 1.')

        dataset_size = len(self.image_paths)
        if dataset_size == 0:
            raise ValueError('No image paths were loaded from label_path.')

        train_size = int(dataset_size * train_ratio)
        val_size = int(dataset_size * val_ratio)
        test_size = dataset_size - train_size - val_size

        generator = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(
            self,
            [train_size, val_size, test_size],
            generator=generator
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return train_loader, val_loader, test_loader

    @staticmethod
    def solve_paths(label_path):
        paths = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                splited_line = line.strip().split(',')
                if len(splited_line) < 3:
                    continue
                if splited_line[1].lower() not in ['good', 'bad'] or splited_line[2].lower() not in ['good', 'bad']:
                    pass
                else:
                    first_label = 1 if splited_line[1].lower()=='good' else 0
                    second_label = 1 if splited_line[2].lower()=='good' else 0
                    labels.append([first_label, second_label])
                    image_path = splited_line[0]
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(os.path.dirname(label_path), image_path)
                    paths.append(image_path)
        return paths, labels

    @staticmethod
    def _validate_image_size(image_size):
        if image_size is None:
            return None

        if isinstance(image_size, int):
            if image_size <= 0:
                raise ValueError('image_size must be a positive integer or a tuple of two positive integers.')
            return (image_size, image_size)

        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            width, height = image_size
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                raise ValueError('image_size tuple values must be positive integers.')
            return (width, height)

        raise ValueError('image_size must be None, a positive integer, or a (width, height) tuple.')

    