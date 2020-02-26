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
speakers_csv_pathname = os.path.join(dataset_pathname, r'speakers_all.csv')
dataset_recordings_folder_pathname = os.path.join(dataset_pathname, r'recordings/recordings/')

classes = ['age', 'age_onset', 'birthplace', 'filename',
           'native_language', 'sex', 'speakerid', 'country']


def load_recording(header, filename, path, ext):
    csv = np.genfromtxt(speakers_csv_pathname, delimiter=',')
    column = csv[classes.index('filename')]
    print(column)

    row = np.where(csv == filename + ext)
    line = row[0]
    
    line = csv
    file_audio = os.path.join(path, filename + ext)
    waveform, sample_rate = torchaudio(file_audio)
    return waveform, sample_rate


class AccentDataset(Dataset):
    ext = '.mp3'

    def __init__(self):
        walker = walk_files(
            dataset_recordings_folder_pathname, suffix = self.ext, prefix=False, remove_suffix=True
        )
        self._header = next(walker)
        self._walker = list(walker)


    def __len__(self):
        return len(self._walker)

    def __getitem__(self, n):
        filename = self._walker[n]
        return load_recording(filename, dataset_recordings_folder_pathname, self.ext)


