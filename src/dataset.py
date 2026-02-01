# dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
import torch

class TrafficImageDataset(Dataset):
    def __init__(self, tensor_dir, label):
        self.tensor_dir = tensor_dir
        self.files = [f for f in os.listdir(tensor_dir) if f.endswith('.npy')]
        self.label = int(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.tensor_dir, self.files[idx])
        img = np.load(path).astype(np.float32) / 255.0   # Normalize to [0, 1]
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = torch.tensor(img).permute(2, 0, 1)         # (3, H, W)
        label = torch.tensor(self.label, dtype=torch.long)
        return img, label