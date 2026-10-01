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

    # Set up W and B
    wandb.init(
        name=args.run_name or f"shg_unet",
        entity=args.wandb_team    or cfg["team"],
        project=args.wandb_project or cfg["project"],
        dir=args.wandb_dir         or cfg["dir"],
        config=cfg,
    )
    
    # Load data
    img_path = cfg["img_path"]
    print(f"Loading data from: {img_path}")

    dataset = ImageDataset(img_path)
    print(f"Loaded dataset of size: {len(dataset)}")

    dataloader = DataLoader(dataset)

    # # Set train test splits
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size

    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # train_loader = DataLoader(train_dataset, batch_size=cfg['batchsize'], shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=cfg['batchsize'], shuffle=False)

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

    density_criterion = nn.MSELoss()
    vector_criterion = nn.MSELoss()
    # meta_loss = nn.MSELoss()

    # criterion =  density_loss + vector_loss + meta_loss
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr']) # Can change this...

    n_epochs = cfg['epochs']
    print(f"Training for {n_epochs} epochs")

    for epoch in range(n_epochs):
        model.train()

        running_loss = 0
        density_loss = 0
        vector_loss = 0
        
        batches = 0

        loss_history = {}

        for pkl_path, shg_true, meta_true, density_true, vector_true in dataloader:

            # Move to device
            shg_true = shg_true.to(DEVICE)
            density_true = density_true.to(DEVICE)
            vector_true = vector_true.to(DEVICE)

            optimizer.zero_grad() 

            # Model
            shg_pred, density_pred, vector_pred = model(shg_true) #meta_pred, emb

            # Calculate loss
            d_loss = density_criterion(density_pred, density_true)
            v_loss = vector_criterion(vector_pred, vector_true)
            total_loss = d_loss + v_loss

            # Gradient
            total_loss.backward()       # Backpropagation
            optimizer.step()      # Update parameters

            running_loss += total_loss.item()
            density_loss += d_loss.item()
            vector_loss += v_loss.item()

            batches += 1

        epoch_loss = running_loss / batches
        density_loss = density_loss / batches
        vector_loss = vector_loss / batches


        loss_history[epoch] = epoch_loss

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": epoch_loss, 
                "density_loss": density_loss,
                "vector_loss": vector_loss
            }
        )

        print(f"Epoch loss: {epoch_loss:.10f}")


        if (epoch+1) % 100 == 0:
            save_path = f'{args.save_dir}/{save_name}_{epoch+1}.pth'
            print(f"Saving checkpoint: {save_path}")
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history
            }, save_path)

        if (epoch-1) == n_epochs:
            final = {
                "pkl_path": pkl_path,
                "shg_true": to_cpu(shg_true),
                "meta_true": to_cpu(meta_true),
                "denstiy_true": to_cpu(denstiy_true), 
                "vector_true": to_cpu(vector_true),
                "shg_pred": to_cpu(shg_pred),
                "density_pred": to_cpu(density_pred),
                "vector_pred": to_cpu(vector_pred)
            }
            print("Save sample result")
            with open(f"{args.save_dir}/final_result_{args.run_name}.pkl", "wb") as f:
                pickle.dump(res, f)

    print(f"Saving final model")
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history
    }, f'{args.save_dir}/{save_name}.pth')

        
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
