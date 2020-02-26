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
    return d.AccentDataset()

def createNeuralNetwork():
    return neural_network.NN()

def train(network):
    raise NotImplementedError

def test(network):
    raise NotImplementedError

def main():
    device = setupDevice()
    dataset = processDataset()
    network = createNeuralNetwork()
    train(network)
    test(network)

if __name__ == "__main__":
    main()

