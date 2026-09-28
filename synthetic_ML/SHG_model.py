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


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        skip_features = F.relu(self.conv2(x)) # We want this for the skip connection
        pool_out = self.pool(skip_features)   # This goes to the next encoder layer
        return skip_features, pool_out
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, num_filters, skip_channels):
        super(DecoderBlock, self).__init__()

        self.convT = nn.ConvTranspose2d(in_channels, num_filters, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(num_filters + skip_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

    def forward(self, x, skip_features):
        x = self.convT(x)
        # Skip features
        if skip_features.shape[2:] != x.shape[2:]:
            skip_features = TF.resize(skip_features, size=x.shape[2:])
        
        # Concate skips
        x = torch.cat([x, skip_features], dim=1)

        # ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class UNET(nn.Module):
    """
    This UNET architecture is adapted from https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/

    Added a vectorfield and density field head
    Added a parameter estimator

    Input: 
        img, fields, metadata

    Outputs: 
        reconstructed shg, meta, density
        param vector
    """
    def __init__(self, 
                 input_img_size=512,
                 num_filters=64, # for a 512 image
                 in_channels=1,  # n_channels, 3d will have more...
                 num_classes=1,   # grayscale images, numclasses=1
                 num_params=12):
        super(UNET, self).__init__()

        self.enc1 = EncoderBlock(in_channels, num_filters)
        self.enc2 = EncoderBlock(num_filters, num_filters*2)
        self.enc3 = EncoderBlock(num_filters*2, num_filters*4)
        self.enc4 = EncoderBlock(num_filters*4, num_filters*8)

        self.bottleneck_conv1 = nn.Conv2d(num_filters*8, num_filters*16, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(num_filters*16, num_filters*16, kernel_size=3, padding=1)

        self.dec1 = DecoderBlock(num_filters*16, num_filters*8, num_filters*8) #input, skip, filters
        self.dec2 = DecoderBlock(num_filters*8, num_filters*4, num_filters*4)
        self.dec3 = DecoderBlock(num_filters*4, num_filters*2, num_filters*2)
        self.dec4 = DecoderBlock(num_filters*2, num_filters, num_filters)

        self.final_conv = nn.Conv2d(num_filters, num_classes, kernel_size=1) #Replace with the classifiers

        self.density_head = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters // 2, 1, kernel_size=1),
            nn.Sigmoid() #[0,1]
        )

        self.vector_head = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters // 2, 1, kernel_size=1),
            nn.Sigmoid() #[0,1], maybe tan
        )

        flat_size = (input_img_size // 16)**2 * (num_filters*16) #img_size * num_filters
        self.meta_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_params)
        )

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b1 = self.bottleneck_conv1(p4)
        b2 = self.bottleneck_conv2(b1)

        meta = self.meta_head(b2)

        d1 = self.dec1(b2, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        shg = self.final_conv(d4)
        density = self.density_head(d4)
        vector = self.vector_head(d4)

        return shg, meta, density, vector #self.final_conv(d4)
