import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        # Sử dụng AdaptiveAvgPool2d để out cố định 4x1x1 cho mọi size ảnh
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 3:
            # (Batch, H, W) -> (Batch, 1, H, W)
            x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x))
