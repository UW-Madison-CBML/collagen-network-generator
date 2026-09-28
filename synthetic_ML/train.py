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

def main(args):
    torch.cuda.empty_cache()

    # Get config info
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_tag = args.run_name or "shg_unet"

    save_name = run_tag

    # Set up W and B
    wandb.init(
        name=args.run_name or f"shg_unet",
        entity=args.wandb_team    or cfg["team"],
        project=args.wandb_project or cfg["project"],
        dir=args.wandb_dir         or cfg["dir"],
        config=cfg,
    )
    
    # Load data
    dataset = ImageDataset(cfg["img_path"])

    # Set train test splits
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg['batchsize'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batchsize'], shuffle=False)

    # Model + params
    model = UNET(
        input_img_size=cfg['img_size'], # 512
        num_filters=64,                 # for a 512 image
        in_channels=1,                  # n_channels, 3d will have more...
        num_classes=1,                  # grayscale images, numclasses=1
        num_params=12
    )

    model.to(DEVICE)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable}")

    density_loss = nn.MSELoss()
    vector_loss = nn.MSELoss()
    meta_loss = nn.MSELoss()

    criterion =  density_loss + vector_loss + meta_loss
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr']) # Can change this...

    n_epochs = cfg['epochs']

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0
        batches = 0

        loss_history = {}

        for pkl_path, shg_true, meta_true, density_true, vector_true in train_loader:

            # Model
            shg, meta, density, vector, emb = model(shg_true)

            # Calculate loss
            loss = criterion("placeholder")

            # Gradient
            optimizer.zero_grad() 
            loss.backward()       # Backpropagation
            optimizer.step()      # Update parameters
            running_loss += loss.item()
            batches += 1
            loss_history[epoch] = epoch_loss

        epoch_loss = running_loss / batches
        print(f"\nEpoch loss: {epoch_loss:.10f}")

    if epoch+1 % 100 == 0:
        save_path = f'{args.save_dir}/{save_name}_{epoch+1}.pth'
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_history
        }, save_path)









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