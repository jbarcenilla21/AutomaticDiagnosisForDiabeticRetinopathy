"""Custom CNN built from scratch for binary DR classification.

No pretrained weights are used — compliant with the Custom track rules.
Input: (N, 3, 224, 224) float32 tensor.
Output: (N, 1) float32 tensor of DR scores in [0, 1].
"""

import torch.nn as nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    """4-block Conv-ReLU-Pool backbone followed by a 3-layer MLP head.

    Spatial flow on 224x224 input (conv 5x5, pool 2x2 at each block):
        224 -> 220 -> 110 -> 106 -> 53 -> 49 -> 24 -> 20 -> 10
    Flattened feature vector: 64 * 10 * 10 = 6400
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  6,  5)
        self.conv2 = nn.Conv2d(6,  16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(6400, 120)
        self.fc2   = nn.Linear(120,  84)
        self.fc3   = nn.Linear(84,   1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
