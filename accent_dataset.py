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
import librosa
import warnings


warnings.filterwarnings('ignore')

dataset_pathname = r'dataset/'
speakers_csv_pathname = os.path.join(dataset_pathname, r'speakers_all.csv')
dataset_recordings_folder_pathname = os.path.join(dataset_pathname, r'recordings/recordings/')

headernames = ['age', 'age_onset', 'filename',
           'native_language', 'sex', 'speakerid', 'country']


class AccentDataset(Dataset):
    ext = '.mp3'

    def load_recording(self, filename, row, path, ext):
        csv = np.genfromtxt(speakers_csv_pathname, delimiter=',')
        print(csv)
        speakerid = csv[row]
        # labels = dict(zip(headers, line))
        file_audio = os.path.join(path, filename + ext)
        sound_data, sample_size = torchaudio.load(file_audio)
        temp_data = torch.zeros([4200000])
        print(sound_data, sound_data.numel())
        if sound_data.numel() < 4200000:
            temp_data[:sound_data.numel()] = sound_data[:]
        # print(temp_data, temp_data.numel())
        sound_data = temp_data
        formatted_sound = torchaudio.transforms.Spectrogram()(sound_data)

        print(formatted_sound)
        # labels = torch.from_numpy(csv[row])
        # print(labels)
        # print(labels.size())
        labels = torch.tensor(self.labels)
        return formatted_sound

    def __init__(self):
        csv = np.genfromtxt(speakers_csv_pathname, delimiter=',')
        self.labels = []
        for i in range(1,len(csv[:])):
            self.labels.append(csv[i,5])


        print(self.labels)

        walker = walk_files(
            dataset_recordings_folder_pathname, suffix = self.ext, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __len__(self):
        return len(self._walker)

    def __getitem__(self, n):
        filename = self._walker[n]
        return self.load_recording(filename, n, dataset_recordings_folder_pathname, self.ext), self.labels[n]


