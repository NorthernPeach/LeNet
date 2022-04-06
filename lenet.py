# -*- coding: utf-8 -*-
"""LeNet.py
"""

import torch
import torch.nn as nn

class LeNEt(nn.Module):
  def __init__(self):
    super(LeNEt, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
    self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
    self.linear1 = nn.Linear(120, 84)
    self.linear2 = nn.Linear(84, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.pool(x)
    x = self.relu(self.conv2(x))
    x = self.pool(x)
    x = self.relu(self.conv3(x)) # 120 x 1 x 1
    x = x.reshape(x.shape[0], -1)
    x = self.relu(self.linear1(x))
    out = self.linear2(x)
    return out

x = torch.randn(64, 1, 32, 32)
model = LeNEt()
outputs = model(x)
print(outputs.shape)

