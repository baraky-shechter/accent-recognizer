import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import accent_dataset as d
import neural_network

def setupDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def processDataset():
    dataset = d.AccentDataset()
    loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
    print(loader)
    return loader

def createNeuralNetwork(dataloader):
    return neural_network.NN()

def train(network):
    raise NotImplementedError

def test(network):
    raise NotImplementedError

def main():
    device = setupDevice()
    dataloader = processDataset()
    network = createNeuralNetwork(dataloader)

if __name__ == "__main__":
    main()

