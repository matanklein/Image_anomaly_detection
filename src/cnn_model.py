import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FlowPicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FlowPicCNN, self).__init__()
        
        # Layer 1: Conv 10 filters, kernel 10, stride 5 
        # Input 1500x1500 -> Output 300x300
        self.conv1 = nn.Conv2d(1, 10, kernel_size=10, padding=3, stride=5)
        self.pool1 = nn.MaxPool2d(2) # Output 150x150 
        
        # Layer 2: Conv 20 filters, kernel 10, stride 5 
        # Input 150x150 -> Output 30x30
        self.conv2 = nn.Conv2d(10, 20, kernel_size=10, padding=3, stride=5)
        self.pool2 = nn.MaxPool2d(2) # Output 15x15 
        
        # Specified Dropout Rates 
        self.dropout_conv2 = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        
        # Flattened dimension: 20 maps * 15 * 15 = 4500 
        self.fc1 = nn.Linear(4500, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x)) # 
        x = self.pool1(x)
        
        # Block 2 with 0.25 Dropout 
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout_conv2(x)
        x = self.pool2(x)
        
        # Flatten for Dense Layers
        x = x.view(x.size(0), -1) 
        
        # Hidden Fully-Connected Layer with 0.5 Dropout 
        embedding = F.relu(self.fc1(x))
        x = self.dropout_fc(embedding)
        
        # Final Logits (Softmax omitted per request)
        out = self.fc2(x)
        
        # Return logits and embedding vector 
        return out, embedding