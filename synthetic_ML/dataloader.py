import os
import glob
import numpy as np
import pandas as pd
import pickle
import sys
import random
import io

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache"
os.environ["USER"] = "researcher"
os.environ["LOGNAME"] = "researcher"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from PIL import Image
from pathlib import Path


def load_metadata(meta_dict):
    """
    Helper function to load the meta data
    Returns a list of all parameter values
    """
    # seed = meta_dict.get("seed") #not used

    G_align = meta_dict.get("G_align")
    G_density = meta_dict.get("G_density")
    G_curve = meta_dict.get("G_curve")
    G_conn = meta_dict.get("G_conn")

    L_align = meta_dict.get("L_align")
    L_density = meta_dict.get("L_density")
    L_conn = meta_dict.get("L_conn")
    L_curve = meta_dict.get("L_curve")

    spline_length = meta_dict.get("spline_length")
    spline_num = meta_dict.get("spline_num")

    wave_amplitude_px = meta_dict.get("wave_amplitude_px")
    wave_wavelength_px = meta_dict.get("wave_wavelength_px")  # Can be None
    L_wave_freq = meta_dict.get("L_wave_freq")

    return [
        G_align,
        G_density,
        G_curve,
        G_conn,
        L_align,
        L_density,
        L_conn,
        L_curve,
        spline_length,
        spline_num,
        wave_amplitude_px,
        wave_wavelength_px,
        L_wave_freq,
    ]

class ImageDataset(Dataset):
    def __init__(self, img_path):
        self.pkl_paths = list(Path(img_path).glob("*pkl"))

    def __len(self):
        return len(self.pkl_path)
    
    def __getitem__(self, idx):
        # Get path
        pkl_path = self.pkl_paths[idx]

        # Load file
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)

        img = torch.tensor(data['img']) # Is a np.array
        density = torch.tensor(data['density'])
        vector = torch.tensor(data['vector'])

        meta = data['meta'] # List of metadata
        
        return str(pkl_path), img, meta, density, vector

