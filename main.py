import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dataset as d



def setupDevice():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

def processDataset():
    dataset = d.Dataset()

def main():
    setupDevice()
    processDataset()

if __name__ == "__main__":
    main()

