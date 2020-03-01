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

log_interval = 20

def processDataset():
    dataset = d.AccentDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    return loader

def train(epochs, dataloader, network, optimizer):
    network.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        for epoch in range(epochs):
            optimizer.zero_grad()
            data = data.requires_grad_()  # set requires_grad to True for training
            output = network(data)
            print(output)
            output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
            loss = F.nll_loss(output[0], target)  # the loss functions expects a batchSizex10 input
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:  # print training stats
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader), loss))
    # optimizer    network.train()
#     loss_values = []
#     print('training...')
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for i, data in enumerate(dataloader,0):
#             optimizer.zero_grad()
#             inputs, labels = data
#
#             output = input.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
#             loss = F.nll_loss(output[0], labels)  # the loss functions expects a batchSizex10 input
#
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             print("calculating loss")
#             if i % 200 == 0:
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 200))
#                 loss_values.append(running_loss / len(dataloader.dataset))
#                 running_loss = 0.0



def test(network):
    raise NotImplementedError

def main():
    dataloader = processDataset()
    network = neural_network.NN()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    train(10, dataloader, network, optimizer)

if __name__ == "__main__":
    main()

