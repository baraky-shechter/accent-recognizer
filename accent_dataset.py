import os
import torchaudio
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import console

dataset_pathname = './dataset/'
speakers_csv_pathname = dataset_pathname + 'speakers_all.csv'
dataset_csv_pathname = dataset_pathname + 'speakers_all.csv'
dataset_recordings_folder_pathname = dataset_pathname + 'recordings/'

class AccentDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.demographics_frame = pd.read_csv(speakers_csv_pathname)
        print(self.demographics_frame)


        for recording in os.listdir(dataset_recordings_folder_pathname):
            self.samples.append(recording)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]



        # console.log(dataset_pathname + "\n" + dataset_csv_pathname + "\n" + dataset_recordings_folder_pathname)
        # filename = dataset_recordings_folder_pathname + 'recordings/afrikaans1.mp3'
        #
        # waveform, sample_rate = torchaudio.load(filename)
        #
        # console.log("Shape of waveform: {}".format(waveform.size()))
        # console.log("Sample rate of waveform: {}".format(sample_rate))
        #
        # plt.figure()
        # plt.plot(waveform.t().numpy())
        # plt.show()


