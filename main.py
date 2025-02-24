# main.py
import sys
import wandb
import torch

# Ensure that the mednext repository is available (if not, clone and install it beforehand)
sys.path.append('mednext')

from models import MedNextGenerator3D, MonaiDiscriminator
from train import get_dataloader, show_sample_pair, train_loop

# Log in to wandb (replace with your own API key or set the WANDB_API_KEY env variable)
WANDB_API_KEY = "8b67af0ea5e8251ee45c6180b5132d513b68c079"  # ← Replace with your key
wandb.login(key=WANDB_API_KEY)

# Data directories (ensure these paths point to your data)
input_dir = "msk-mri-dataset-2021/BrainMSK-1C"
label_dir = "msk-mri-dataset-2021/BrainMRI"

# Create dataloader
dataloader = get_dataloader(input_dir, label_dir, batch_size=3)

# Optionally, display a sample pair from the dataset
sample_batch = next(iter(dataloader))
show_sample_pair(sample_batch)

# Instantiate the models
G = MedNextGenerator3D(input_channels=1, output_channels=1)
D = MonaiDiscriminator

# Set the number of training epochs
EPOCH = 2

# Initialize the wandb run
wandb.init(project="Pix2Pix-MRI-Style-Transfer")

# Run training
trained_G, trained_D = train_loop(dataloader, G, D, EPOCH)
