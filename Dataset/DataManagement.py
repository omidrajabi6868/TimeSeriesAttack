from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os

class ImageDataSet(TorchDataset):
    def __init__(self, label_path, trasform=None):
        self.label_path = label_path
        self.image_paths, self.labels = self.solve_paths(self.label_path)

        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')/255.
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        if self.trasform:
            img = self.trasform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label
    
    def train_val_test_loader(self):

        
        return

    @staticmethod
    def solve_paths(label_path):
        paths = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                splited_line = line.split(',')
                if splited_line[1].lower() not in ['good', 'bad'] or splited_line[2].lower() not in ['good', 'bad']:
                    pass
                else:
                    first_label = 1 if splited_line[1].lower()=='good' else 0
                    second_label = 1 if splited_line[2].lower()=='good' else 0
                    labels.append([first_label, second_label])
                    paths.append(splited_line[0])
        return paths, labels

    