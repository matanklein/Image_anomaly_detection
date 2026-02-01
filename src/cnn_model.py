import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FlowPicCNN(nn.Module):
    def __init__(self, input_dim=1500, num_classes=1, dropout_rate=0.5):
        super(FlowPicCNN, self).__init__()
        
        # Layer 1: Conv 10 filters, kernel 10, stride 5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=10, stride=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(2)
        
        # Layer 2: Conv 20 filters, kernel 10, stride 5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=10, stride=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2)
        
        # Calculate Flatten dimension dynamically
        # Input: 1500 -> Conv1(s5) -> 299 -> Pool(2) -> 149
        # 149 -> Conv2(s5) -> 28 -> Pool(2) -> 14
        # 14 * 14 * 20 = 3920 (Approx, logic needs to be exact)
        self._to_linear = None
        self._check_dim(input_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _check_dim(self, dim):
        with torch.no_grad():
            x = torch.zeros(1, 1, dim, dim)
            x = self.pool1(self.bn1(self.conv1(x)))
            x = self.pool2(self.bn2(self.conv2(x)))
            self._to_linear = int(np.prod(x.shape[1:]))

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1) # Flatten
        embedding = F.relu(self.fc1(x)) # Embedding vector
        
        out = self.fc2(self.dropout(embedding))
        
        # Return both logits and embedding (useful for OE)
        return out, embedding