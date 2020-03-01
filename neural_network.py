import torch
import torch.nn as nn
import os
import torchaudio
import torchvision
import numpy
import pandas
import console
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv1d(201, 8, 21001)
        self.bn1 = nn.BatchNorm1d(8)
        # # self.pool1 = nn.MaxPool1d(4)
        # self.conv2 = nn.Conv1d(8, 8, 1)
        # self.bn2 = nn.BatchNorm1d(8)
        # # self.pool2 = nn.MaxPool1d(4)
        # self.conv3 = nn.Conv1d(128, 256, 3)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.pool3 = nn.MaxPool1d(4)
        # self.conv4 = nn.Conv1d(256, 512, 3)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.pool4 = nn.MaxPool1d(4)
        # self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(1, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        # # x = self.pool1(x)
        # x = self.conv2(x)
        # x = F.relu(self.bn2(x))
        # # x = self.pool2(x)
        # x = self.conv3(x)
        # x = F.relu(self.bn3(x))
        # # x = self.pool3(x)
        # x = self.conv4(x)
        # x = F.relu(self.bn4(x))
        # # x = self.pool4(x)
        # x = self.avgPool(x)
        # x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features