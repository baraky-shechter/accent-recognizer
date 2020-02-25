import os
import torchaudio
import torch
import pandas as pd
import numpy as np

print("worked")

class Dataset:

    def __init__(self):
        dataset_pathname = './dataset/'
        dataset_csv_pathname = dataset_pathname + 'speakers_all.csv'
        dataset_recordings_folder_pathname = dataset_pathname + 'recordings/'

        print(dataset_pathname + "\n" + dataset_csv_pathname + "\n" + dataset_recordings_folder_pathname)


