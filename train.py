# train.py
import os
import pickle
from glob import glob
from statistics import mean

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader
from monai.losses import SSIMLoss
from generative.losses import PerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Preparation ---
def read_paths_pair(input_dir: str, label_dir: str, pattern: str = "*.nii.gz"):
    input_files = sorted(glob(os.path.join(input_dir, pattern)))
    label_files = sorted(glob(os.path.join(label_dir, pattern)))
    return [{"input": inp, "label": lab} for inp, lab in zip(input_files, label_files)]

# MONAI Transforms
paired_transforms = Compose([
    LoadImaged(keys=["input", "label"]),
    EnsureChannelFirstd(keys=["input", "label"]),
    Orientationd(keys=["input", "label"], axcodes="RAS"),
    Spacingd(
        keys=["input", "label"],
        pixdim=(2.4, 2.4, 2.2),
        mode=("bilinear", "bilinear")
    ),
    CenterSpatialCropd(keys=["input", "label"], roi_size=(96, 96, 64)),
    ScaleIntensityRangePercentilesd(
        keys=["input", "label"],
        lower=0,
        upper=99.5,
        b_min=-1.0,
        b_max=1.0
    ),
    EnsureTyped(keys=["input", "label"]),
])

def get_dataloader(input_dir, label_dir, batch_size=3):
    data_files = read_paths_pair(input_dir, label_dir, pattern="*.nii.gz")
    paired_dataset = Dataset(data=data_files, transform=paired_transforms)
    paired_loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)
    return paired_loader

def show_sample_pair(sample):
    input_volume = sample["input"][0]
    label_volume = sample["label"][0]

    if input_volume.shape[0] > 1:
        input_volume = input_volume[0]
    if label_volume.shape[0] > 1:
        label_volume = label_volume[0]

    mid_input = input_volume[..., 30].squeeze()
    mid_label = label_volume[..., 30].squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mid_input, cmap="gray")
    axes[0].set_title("Input Mask")
    axes[0].axis("off")
    axes[1].imshow(mid_label, cmap="gray")
    axes[1].set_title("Label MRI")
    axes[1].axis("off")
    plt.show()

# --- Training Functions ---
def train_fn(train_dl, G, D, criterion_bce, criterion_mae, criterion_perceptual, 
             optimizer_g, optimizer_d, noise_std=0.05, criterion_ssim=None):
    G.train()
    D.train()

    LAMBDA_L1 = 5.0
    LAMBDA_PERCEPT = 2
    LAMBDA_SSIM = 1.5
    
    total_loss_g, total_loss_d = [], []

    for i, batch in enumerate(tqdm(train_dl)):
        input_img = batch["input"].to(device)
        real_img_clean = batch["label"].to(device)

        # Generator Forward Pass (Clean)
        fake_img_clean = G(input_img)

        # Add noise to discriminator inputs only
        real_img_noisy = real_img_clean + noise_std * torch.randn_like(real_img_clean)
        fake_img_noisy = fake_img_clean.detach() + noise_std * torch.randn_like(fake_img_clean)

        # Discriminator Forward Pass (Noisy Inputs)
        fake_pred = D(torch.cat([input_img, fake_img_noisy], dim=1))
        if isinstance(fake_pred, list):
            fake_pred = fake_pred[-1]

        real_pred = D(torch.cat([input_img, real_img_noisy], dim=1))
        if isinstance(real_pred, list):
            real_pred = real_pred[-1]

        # Dynamic Label Smoothing
        real_label = torch.rand_like(real_pred) * 0.2 + 0.8
        fake_label = torch.rand_like(fake_pred) * 0.2

        # Generator Loss
        loss_g_gan = criterion_bce(fake_pred, real_label)
        loss_g_l1 = criterion_mae(fake_img_clean, real_img_clean)
        loss_g_perceptual = criterion_perceptual(fake_img_clean, real_img_clean)
        
        loss_g_ssim = 0
        if criterion_ssim is not None:
            loss_g_ssim = criterion_ssim(fake_img_clean, real_img_clean)
            wandb.log({
                "SSIM Loss": loss_g_ssim.item(),
                "GAN Loss": loss_g_gan.item(),
                "L1 Loss": loss_g_l1.item(),
                "Perceptual Loss": loss_g_perceptual.item()
            })
        
        loss_g = loss_g_gan + LAMBDA_L1 * loss_g_l1 + LAMBDA_PERCEPT * loss_g_perceptual + LAMBDA_SSIM * loss_g_ssim

        optimizer_g.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optimizer_g.step()

        # Discriminator Update
        fake_pred = D(torch.cat([input_img, fake_img_noisy.detach()], dim=1))
        if isinstance(fake_pred, list):
            fake_pred = fake_pred[-1]
        loss_d_fake = criterion_bce(fake_pred, fake_label)
        loss_d_real = criterion_bce(real_pred, real_label)
        loss_d = (loss_d_real + loss_d_fake) * 0.75

        optimizer_d.zero_grad()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
        optimizer_d.step()

        # Track gradients
        g_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in G.parameters() if p.grad is not None]))
        d_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in D.parameters() if p.grad is not None]))

        total_loss_g.append(loss_g.item())
        total_loss_d.append(loss_d.item())

    return (mean(total_loss_g), mean(total_loss_d),
            fake_img_clean.detach().cpu(),  
            real_img_clean.detach().cpu(),
            input_img.detach().cpu(),
            g_grad_norm.item(),
            d_grad_norm.item())

def train_loop(train_dl, G, D, num_epoch, lr=0.0002, betas=(0.5, 0.999)):
    G.to(device)
    D.to(device)

    optimizer_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=betas)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=1e-5, betas=betas)

    criterion_mae = nn.L1Loss()
    criterion_bce = nn.MSELoss()
    criterion_perceptual = PerceptualLoss(
        spatial_dims=3,
        network_type="squeeze",  # alternatives: "resnet18", "resnet34", etc.
        is_fake_3d=True,
        pretrained=True,
    ).to(device)
    criterion_ssim = SSIMLoss(
        spatial_dims=3,
        data_range=2.0,
        kernel_type="gaussian",
        win_size=11,
        kernel_sigma=1.5,
        k1=0.01,
        k2=0.03,
        reduction="mean"
    ).to(device)
    
    total_loss_d, total_loss_g = [], []
    result = {"G": [], "D": []}

    for e in range(num_epoch): 
        noise_std = max(0.05 * (1 - e / num_epoch), 0.01)
        
        print(f"\nEpoch {e+1}/{num_epoch}")
        loss_g, loss_d, fake_img, real_img, input_img, g_grad_norm, d_grad_norm = train_fn(
            train_dl, G, D, criterion_bce, criterion_mae, criterion_perceptual, 
            optimizer_g, optimizer_d, noise_std, criterion_ssim
        )

        total_loss_d.append(loss_d)
        total_loss_g.append(loss_g)
        result["G"].append(loss_g)
        result["D"].append(loss_d)

        print(f"Generator Loss: {loss_g:.4f}, Discriminator Loss: {loss_d:.4f}")

        wandb.log({
            "Generator Loss": loss_g,
            "Discriminator Loss": loss_d,
            "G Grad Norm": g_grad_norm,
            "D Grad Norm": d_grad_norm,
            "Noise Std": noise_std,
            "epoch": e + 1
        })

        if (e + 1) % 3 == 0:
            saving_model(D, G, e)
            saving_logs(result)
            save_comparison(real_img, fake_img, input_img, e + 1)
            show_losses(total_loss_g, total_loss_d)

            wandb.log({
                "Comparison": wandb.Image(f'generated/comparison_epoch_{e + 1}.png'),
            }, step=e + 1)

    saving_model(D, G, num_epoch - 1)
    saving_logs(result)
    show_losses(total_loss_g, total_loss_d)
    print("Training completed successfully")
    return G, D

# --- Helper Functions ---
def save_comparison(real_img, fake_img, input_img, epoch):
    os.makedirs("generated", exist_ok=True)
    real_sample = real_img[0]
    fake_sample = fake_img[0]
    input_sample = input_img[0]

    if real_sample.shape[0] > 1:
        real_sample = real_sample[0].unsqueeze(0)
        fake_sample = fake_sample[0].unsqueeze(0)
        input_sample = input_sample[0].unsqueeze(0)

    real_slice = real_sample[..., 30]
    fake_slice = fake_sample[..., 30]
    input_slice = input_sample[..., 30]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    real_np = real_slice.cpu().detach().numpy().squeeze()
    fake_np = fake_slice.cpu().detach().numpy().squeeze()
    input_np = input_slice.cpu().detach().numpy().squeeze()

    ax1.imshow(input_np, cmap='gray')
    ax1.set_title('Input Image')
    ax1.axis('off')
    ax2.imshow(fake_np, cmap='gray')
    ax2.set_title('Generated Image')
    ax2.axis('off')
    ax3.imshow(real_np, cmap='gray')
    ax3.set_title('Real Image')
    ax3.axis('off')

    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(f'generated/comparison_epoch_{epoch}.png', bbox_inches='tight', dpi=150)
    plt.close()

def saving_logs(result):
    with open("train.pkl", "wb") as f:
        pickle.dump([result], f)

def saving_model(D, G, e):
    os.makedirs("weight", exist_ok=True)
    torch.save(G.state_dict(), f"weight/G{str(e+1)}.pth")
    torch.save(D.state_dict(), f"weight/D{str(e+1)}.pth")

def show_losses(g, d):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes.ravel()
    epochs = list(range(len(g)))
    ax[0].plot(epochs, g)
    ax[0].set_title("Generator Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[1].plot(epochs, d)
    ax[1].set_title("Discriminator Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()