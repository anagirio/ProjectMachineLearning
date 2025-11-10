import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Bloco 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Bloco 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloco 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        # the dropout layer avoids overfitting by randomly setting a fraction of input units to 0 during training
        self.dropout = nn.Dropout(0.5)
        
        # 36→18→9→4, então 128*4*4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Bloco 1: Conv → BN → ReLU → Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Bloco 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Bloco 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Classificador
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x