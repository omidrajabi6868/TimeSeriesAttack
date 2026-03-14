from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from PIL import Image
import torch

class ImageDataSet(TorchDataset):
    def __init__(self, image_dir, label_dir, trasform=None):
        self.image_dir = images_dir
        self.label_dir = label_dir

        self.image_paths = solve_image_path(self.image_dir)
        self.labels = solve_label_path(self.label_dir)
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.trasform:
            img = self.trasform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label
    
    def train_val_loader(self):
        return

    def test_loader(self):
        return

    
    