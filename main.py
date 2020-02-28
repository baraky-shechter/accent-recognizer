import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import accent_dataset as d
import neural_network
import console

def processDataset():
    dataset = d.AccentDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return loader

def train(epochs, dataloader, network, optimizer, criterion):
    loss_values = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader,0):
            inputs, labels = data

            if (torch.cuda.is_available()):
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                loss_values.append(running_loss / len(dataloader.dataset))
                running_loss = 0.0

def test(network):
    raise NotImplementedError

def main():
    dataloader = processDataset()
    network = neural_network.NN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    train(10, dataloader, network, optimizer, criterion)

if __name__ == "__main__":
    main()

