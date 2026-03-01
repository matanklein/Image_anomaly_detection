# dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import config as Config

class TrafficImageDataset(Dataset):
    def __init__(self, tensor_dir, label):
        self.tensor_dir = tensor_dir
        self.label = int(label)
        self.files = self._collect_files()

    def _collect_files(self):
        label_keyword = None
        if self.label == Config.BENIGN_LABEL:
            label_keyword = 'benign'
        elif self.label == Config.MALICIOUS_LABEL:
            label_keyword = 'malicious'

        candidate_roots = []
        if label_keyword is None:
            candidate_roots = [self.tensor_dir]
        else:
            for root, _, _ in os.walk(self.tensor_dir):
                if os.path.basename(root).lower() == label_keyword:
                    candidate_roots.append(root)

            if not candidate_roots:
                if os.path.basename(self.tensor_dir).lower() == label_keyword:
                    candidate_roots = [self.tensor_dir]
                else:
                    candidate_roots = [self.tensor_dir]

        files = []
        for root_dir in candidate_roots:
            for root, _, filenames in os.walk(root_dir):
                for name in filenames:
                    if name.endswith('.npy'):
                        files.append(os.path.join(root, name))

        return sorted(set(files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = np.load(path).astype(np.float32) / 255.0   # Normalize to [0, 1]
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = torch.tensor(img)
        if img.dim() == 2:
            img = img.unsqueeze(0)  # (1, H, W)
        else:
            img = img.permute(2, 0, 1)  # (C, H, W)
        label = torch.tensor(self.label, dtype=torch.long)
        return img, label