import os
import glob
import numpy as np
import pandas as pd
import pickle
import sys
import random
import io
import argparse

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache"
os.environ["USER"] = "researcher"
os.environ["LOGNAME"] = "researcher"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
import yaml
import wandb

from SHG_model import *
from dataloader import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_cpu(item):
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy()
    elif isinstance(item, list):
        return [to_cpu(i) for i in item]
    return item

def main(args):
    
    print(args.seed)

    torch.cuda.empty_cache()

    # Get config info
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_tag = args.run_name or "shg_unet"

    save_name = run_tag
    
    # Load data
    img_path = cfg["img_path"]
    print(f"Loading data from: {img_path}")

    dataset = ImageDataset(img_path)
    print(f"Loaded dataset of size: {len(dataset)}")

    dataloader = DataLoader(dataset)

    # Model + params
    model = UNET(
        input_img_size=cfg['img_size'], # 512
        num_filters=64,                 # for a 512 image
        in_channels=1,                  # n_channels, 3d will have more...
        num_classes=1,                  # grayscale images, numclasses=1
        num_params=12
    )

    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr']) # Can change this...

    # Load checkpoint
    model_path = "/staging/s/svaren/synthetic_ML/results/ML_test_2k.pth"
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()

    counter = 1

    with torch.no_grad():
        for pkl_path, shg_true, meta_true, density_true, vector_true in dataloader:
            if counter > 50:
                break

            shg_true = shg_true.to(DEVICE)
        
            shg_pred, density_pred, vector_pred = model(shg_true)

            res = {
                "pkl_path": pkl_path,
                "shg_true": to_cpu(shg_true),
                "meta_true": to_cpu(meta_true),
                "density_true": to_cpu(density_true), 
                "vector_true": to_cpu(vector_true),
                "shg_pred": to_cpu(shg_pred),
                "density_pred": to_cpu(density_pred),
                "vector_pred": to_cpu(vector_pred)
            }

            save_name = os.path.basename(pkl_path[0])
            print(f"Save sample result: {pkl_path}")
            with open(f"{args.save_dir}/final_result_{args.run_name}_{save_name}", "wb") as f:
                pickle.dump(res, f)

            counter += 1

            



        
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",        required=True)
    p.add_argument("--save_dir",      required=True)
    p.add_argument("--run_name",      required=True) 
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_team",    default=None)
    p.add_argument("--wandb_dir",     default=None)
    p.add_argument("--seed",          default=777, type=int)


    args = p.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    main(args)
