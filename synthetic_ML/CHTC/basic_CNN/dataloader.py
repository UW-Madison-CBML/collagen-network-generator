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


# def load_metadata(meta_dict):
#     """
#     Helper function to load the meta data
#     Returns a list of all parameter values
#     """
#     # seed = meta_dict.get("seed") #not used

#     G_align = meta_dict.get("G_align")
#     G_density = meta_dict.get("G_density")
#     G_curve = meta_dict.get("G_curve")
#     G_conn = meta_dict.get("G_conn")

#     L_align = meta_dict.get("L_align")
#     L_density = meta_dict.get("L_density")
#     L_conn = meta_dict.get("L_conn")
#     L_curve = meta_dict.get("L_curve")

#     spline_length = meta_dict.get("spline_length")
#     spline_num = meta_dict.get("spline_num")

#     wave_amplitude_px = meta_dict.get("wave_amplitude_px")
#     wave_wavelength_px = meta_dict.get("wave_wavelength_px")  # Can be None
#     L_wave_freq = meta_dict.get("L_wave_freq")

#     return [
#         G_align,
#         G_density,
#         G_curve,
#         G_conn,
#         L_align,
#         L_density,
#         L_conn,
#         L_curve,
#         spline_length,
#         spline_num,
#         wave_amplitude_px,
#         wave_wavelength_px,
#         L_wave_freq,
#     ]

META_MINS = torch.tensor([0.005, 0.005, 0.0, 0.0, 0.005, 0.005, 0.0, 0.0, 50.0, 100.0, 0.0, 10.0, 0.0, 30.0], dtype=torch.float32)
META_MAXS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 500.0, 1000.0, 10.0, 70.0, 1.0, 1000.0], dtype=torch.float32)

class ImageDataset(Dataset):
    def __init__(self, img_path):
        self.pkl_paths = list(Path(img_path).glob("*pkl"))

    def __len__(self):
        return len(self.pkl_paths)
    
    def __getitem__(self, idx):
        # Get path
        pkl_path = self.pkl_paths[idx]

        # Load file
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)

        # [0,1]
        img = torch.from_numpy(data['image']).float() #.unsqueeze(0) # (B, C, 512, 512)
        img = img / 255
        img = img.unsqueeze(0)

        # [0,1]
        density = torch.from_numpy(data['D']).float() #.unsqueeze(0)
        density = density / density.max()
        density = density.unsqueeze(0)

        # [-1,1]
        vector = torch.from_numpy(data['theta']).float() #.unsqueeze(0)
        data_min = vector.min()
        data_max = vector.max()
        vector = 2 * (vector - data_min) / (data_max - data_min) - 1
        vector = vector.unsqueeze(0)

        meta_list = data['meta'] # List of metadata
        meta = torch.tensor(meta_list, dtype=torch.float32) # convert to tensor to use
        meta_norm = (meta - META_MINS) / (META_MAXS - META_MINS)
        meta_norm = torch.clamp(meta_norm, 0.0, 1.0)
        
        return str(pkl_path), img, meta_norm, density, vector


    
    # return {
    #     "Qx": Qx,
    #     "Qy": Qy,
    #     "D": D,
    #     "theta": theta,
    #     "seeds": seeds,
    #     "fibers": fibers,
    #     "splines": splines,
    #     "image": img,
    #     "wells": wells,
    #     "well_influence": influence,
    #     "aux_L_curve": aux_L_curve,
    #     "aux_L_conn": aux_L_conn,
    #     "aux_L_wave_freq": aux_L_wave_freq,
    #     "aux_susceptibility": aux_susceptibility,
    #     "opacity_table": opacity_table,
    #     "spline_length": spline_length,
    #     # res['meta'] = meta_data
    # }