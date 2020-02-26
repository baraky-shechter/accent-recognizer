import os
import torchaudio
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import console
import glob
from torchaudio.datasets.utils import walk_files

dataset_pathname = r'./dataset/'
speakers_csv_pathname = os.path.join(dataset_pathname, r'test.csv')
dataset_recordings_folder_pathname = os.path.join(dataset_pathname, r'recordings/recordings/')


csv = np.genfromtxt(speakers_csv_pathname, delimiter=',')
column = csv[:,1]
print(column)