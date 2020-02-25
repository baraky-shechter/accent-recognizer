import os
import torchaudio
import torch
import pandas as pd
import numpy as np

dataset_pathname = './dataset/'
dataset_csv_pathname = dataset_pathname + 'speakers_all.csv'
dataset_recordings_folder_pathname = dataset_pathname + 'recordings/'

