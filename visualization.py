from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import gc

gc.collect()

# ====== ADDED FOR SURFACE LOSS ======
import trimesh
from skimage.measure import marching_cubes
# ====================================

from get_data import SegDataset
from compoundlossfunction_2 import compound_loss
from model_1 import UNet, SpatialTransformer

# ----------------- Paths ----------------- #
train_txt = "/content/drive/MyDrive/train_npy.txt"
template_path = "/content/drive/MyDrive/brain_data_onehot/OASIS_OAS1_0406_MR1_seg4_onehot.npy"

# ----------------- Params ----------------- #
batch_size = 1
num_epochs = 30
learning_rate = 2e-4
weight_decay = 1e-5
target_size = (96, 96, 96)

# ====== ADDED FOR SURFACE LOSS ======
surface_weight = 0.01
surface_compute_interval = 5  # compute every N batches
# ====================================

# AMP Toggle
use_amp = True
scaler = GradScaler(enabled=use_amp)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
print("Using device:", device)

# ----------------- Dataset + Split ----------------- #
full_dataset = SegDataset(train_txt, template_path, target_size=target_size)

# Split 80% train, 20% test (414 images → 331 train, 83 test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

print(f"Dataset split: {train_size} train, {test_size} test")

# ----------------- Model + Optimizer ----------------- #
unet = UNet(in_channels=10, out_channels_flow=3, out_channels_lambda=1).to(device)
stn = SpatialTransformer(size=target_size, device=device).to(device)

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


model_path = "/content/drive/MyDrive/trained_model.pth"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)

unet.load_state_dict(checkpoint["model_state_dict"])

unet.eval()
stn.eval()

def show_alignment(moving, warped, fixed, slice_index=None):
    moving = moving[0].cpu().numpy()
    warped = warped[0].detach().cpu().numpy()
    fixed = fixed[0].cpu().numpy()
    if slice_index is None:
        slice_index = moving.shape[1] // 2
    def collapse(x): return np.argmax(x[:, slice_index], axis=0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(collapse(moving), cmap='tab10'); axs[0].set_title("Template (Moving)")
    axs[1].imshow(collapse(warped), cmap='tab10'); axs[1].set_title("Warped")
    axs[2].imshow(collapse(fixed), cmap='tab10'); axs[2].set_title("Fixed")
    for ax in axs: ax.axis('off')
    plt.show()

with torch.no_grad():
    for moving, fixed in test_loader:
        moving, fixed = moving.to(device), fixed.to(device)
        x = torch.cat([moving, fixed], dim=1)
        flow, lambda_map = unet(x)
        warped = stn(moving, flow)
        show_alignment(moving.cpu(), warped.cpu(), fixed.cpu())
        break
