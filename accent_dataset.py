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

dataset_pathname = r'dataset/'
speakers_csv_pathname = os.path.join(dataset_pathname, r'speakers_all.csv')
dataset_recordings_folder_pathname = os.path.join(dataset_pathname, r'recordings/recordings/')

headernames = ['age', 'age_onset', 'filename',
           'native_language', 'sex', 'speakerid', 'country']


def load_recording(row, filename, path, ext):
    csv = pd.read_csv(speakers_csv_pathname)
    csv = csv.to_numpy(dtype=object)
    labels = csv[row]
    file_audio = os.path.join(path, filename + ext)
    audio_tensor, sample_rate = torchaudio.load(file_audio)
    filtered_tensor = librosa.util.fix_length(audio_tensor, 15000000)
    return {'tensor' : filtered_tensor,'sample_rate' : sample_rate, 'labels' : labels}

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
        return load_recording(n, filename, dataset_recordings_folder_pathname, self.ext)


