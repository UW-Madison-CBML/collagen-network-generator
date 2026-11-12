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
    run_tag = args.run_name or "shg_unet_upsample"

    save_name = run_tag

    # Set up W and B
    wandb.init(
        name=args.run_name or f"shg_unet_upsample",
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

    # Set train test splits
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg['batchsize'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batchsize'], shuffle=False)

    # Model + params
    # model = ImageCNN(
    #     input_img_size=cfg['img_size'], # 512
    #     num_filters=64,                 # for a 512 image
    #     in_channels=1,                  # n_channels, 3d will have more...
    #     out_channels=1,
    #     num_classes=1,                  # grayscale images, numclasses=1
    #     num_params=12
    # )

    model = ImageCNN(
        input_img_size=cfg['img_size'],
        num_filters=16, # for a 512 image
        in_channels=1,  # n_channels, 3d will have more...
        out_channels=1,
        num_params=14
    )

    model.to(DEVICE)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable}")

    density_criterion = nn.MSELoss()
    vector_criterion = nn.MSELoss()
    meta_criterion = nn.MSELoss()

    # criterion =  density_loss + vector_loss + meta_loss
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr']) # Can change this...

    n_epochs = cfg['epochs']
    print(f"Training for {n_epochs} epochs")

    for epoch in range(n_epochs):
        model.train()

        running_loss = 0
        density_loss = 0
        vector_loss = 0
        meta_loss = 0
        
        batches = 0

        loss_history = {}

        for pkl_path, shg_true, meta_true, density_true, vector_true in train_loader:

            # Move to device
            shg_true = shg_true.to(DEVICE)
            density_true = density_true.to(DEVICE)
            vector_true = vector_true.to(DEVICE)
            meta_true = meta_true.to(DEVICE)

            optimizer.zero_grad() 

            # Model
            density_pred, vector_pred, meta_pred = model(shg_true) #meta_pred, emb

            # Calculate loss
            d_loss = density_criterion(density_pred, density_true)
            v_loss = vector_criterion(vector_pred, vector_true)
            m_loss = meta_criterion(meta_pred, meta_true)

            total_loss = d_loss + v_loss + meta_loss

            # Gradient
            total_loss.backward()       # Backpropagation
            optimizer.step()      # Update parameters

            running_loss += total_loss.item()
            density_loss += d_loss.item()
            vector_loss += v_loss.item()
            meta_loss += m_loss.item()

            batches += 1

        epoch_loss = running_loss / batches

        density_loss = density_loss / batches
        vector_loss = vector_loss / batches
        meta_loss = meta_loss / batches

        loss_history[epoch] = epoch_loss

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": epoch_loss, 
                "density_loss": density_loss,
                "vector_loss": vector_loss,
                "meta_loss": meta_loss
            }
        )

        print(f"Epoch loss: {epoch_loss:.10f}")


        if (epoch+1) % 100 == 0:
            save_path = f'{args.save_dir}/CNN_{save_name}_{epoch+1}.pth'
            print(f"Saving checkpoint: {save_path}")
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history
            }, save_path)

    print(f"Saving final model")
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history
    }, f'{args.save_dir}/CNN_{save_name}.pth')


    # Run validation
    file_paths = []

    all_meta_preds = []
    all_meta_targets = []

    pred_density_list = []
    target_density_list = []

    pred_vector_list = []
    target_vector_list = []

    with torch.no_grad():
        for pkl_path, shg_true, meta_true, density_true, vector_true in dataloader:
            
            file_paths.extend(pkl_path)

            shg_true = shg_true.to(DEVICE)
            density_pred, vector_pred, meta_pred = model(shg_true)

            # save results
            pred_meta_list.append(meta_pred.cpu())
            target_meta_list.append(meta_true)

            pred_density_list.append(density_pred.cpu())
            target_density_list.append(density_true.squeeze())

            pred_vector_list.append(vector_pred.cpu())
            target_vector_list.append(vector_true.squeeze())

        all_pred_meta = torch.cat(pred_meta_list, dim=0)
        all_target_meta = torch.cat(target_meta_list, dim=0)

        all_pred_density = torch.cat(pred_density_list, dim=0)
        all_target_density = torch.cat(target_density_list, dim=0)

        all_pred_vector = torch.cat(pred_vector_list, dim=0)
        all_target_vector = torch.cat(target_vector_list, dim=0)

    pkl_file = {
        "file_paths": file_paths,
        "pred_density": all_pred_density.numpy(),
        "target_density": all_target_density.numpy(),
        "pred_vector": all_pred_vector.numpy(),
        "target_vector": all_target_vector.numpy(),
        "pred_meta": all_pred_meta.numpy(),
        "target_meta": all_target_meta.numpy()
    }

    pkl_path = f"{args.save_dir}/CNN_{save_name}_test_results.pkl"
    print(f"Saving final results here: {pkl_path}")

    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_file, f)

        
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
