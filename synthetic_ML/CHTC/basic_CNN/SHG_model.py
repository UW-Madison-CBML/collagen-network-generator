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


class ImageCNN(nn.Module):
    def __init__(self, 
                 input_img_size=512,
                 num_filters=16, # for a 512 image
                 in_channels=1,  # n_channels, 3d will have more...
                 out_channels=1,
                 num_params=12):
        super(ImageCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            #(B, 1, 512, 512) --> (B, 16, 256, 256)
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.GELU(),

            #(B, 16, 256, 256) --> (B, 32, 128, 128)
            nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.GELU(), 
        )

        self.density_head = nn.Sequential(
            #(B, 32, 128, 128) --> (B, 16, 256, 256)
            nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=2, stride=2),
            nn.GELU(),

            #(B, 16, 256, 256) --> (B, 1, 512, 512)
            nn.ConvTranspose2d(num_filters, out_channels, kernel_size=2, stride=2),
            nn.GELU(),

            #(B, 1, 512, 512) --> (B, 1, 1024, 1024)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Sigmoid() #[0,1]
        )

        self.vector_head = nn.Sequential(
            #(B, 32, 128, 128) --> (B, 16, 256, 256)
            nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=2, stride=2),
            nn.GELU(),

            #(B, 16, 256, 256) --> (B, 1, 512, 512)
            nn.ConvTranspose2d(num_filters, out_channels, kernel_size=2, stride=2),
            nn.GELU(),

            #(B, 1, 512, 512) --> (B, 1, 1024, 1024)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Tanh() #[-1,1]
        )

        flat_size = (input_img_size // 4)**2 * (num_filters*2) #img_size * num_filters

        self.meta_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_params),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc = self.encoder(x)
        # print(f"Encoder output: {enc.shape}")

        # SEPARATE heads
        density = self.density_head(enc)
        # print(f"Density output: {density.shape}")

        vector = self.vector_head(enc)
        # print(f"Vector output: {vector.shape}")

        meta = self.meta_head(enc)
        # print(f"Meta output: {len(meta[1,:])}")

        return density, vector, meta 