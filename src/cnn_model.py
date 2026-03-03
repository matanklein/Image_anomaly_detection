import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FlowPicCNN(nn.Module):
    def __init__(self, input_dim=1500, latent_dim=64):
        super(FlowPicCNN, self).__init__()

        # Deep SVDD-friendly topology:
        # - bias=False everywhere
        # - no BatchNorm
        # - no Dropout
        self.conv1 = nn.Conv2d(1, 10, kernel_size=10, stride=5, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=10, stride=5, padding=3, bias=False)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(4500, latent_dim)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        embedding = self.fc1(x)
        return embedding