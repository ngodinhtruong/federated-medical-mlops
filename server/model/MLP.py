import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))
