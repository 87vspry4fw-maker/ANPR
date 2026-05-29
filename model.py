import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, NumClasses):
        super().__init__()
        self.Convolutional1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.Convolutional2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.Pool = nn.MaxPool2d(2, 2)
        self.FC1 = nn.Linear(64 * 16 * 16, 128)
        self.FC2 = nn.Linear(128, NumClasses)
    
    def forward(self, x):
        self.dropout = nn.Dropout(0.5)
        x = self.Pool(F.relu(self.Convolutional1(x)))
        x = self.Pool(F.relu(self.Convolutional2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.FC1(x))
        x = self.dropout(x)
        x = self.FC2(x)
        return x