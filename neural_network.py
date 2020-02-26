import torch
import torch.nn as nn
import os
import torchaudio
import torchvision
import numpy
import pandas
import console

class NN(nn.Module):

    def __init__(self):
        console.log('Creating Neural Network')
        super(NN, self).__init__()

        # TODO: Setup network