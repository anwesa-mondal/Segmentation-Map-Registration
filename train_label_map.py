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
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

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


# ----------------- Dynamic Loss Weights ----------------- #
def get_loss_weights(epoch):
    if epoch <= 10:
        return {"dice": 1.0, "cross_entropy": 0.0,
                "lambda_smoothness": 0.1, "lambda_prior": 0.05, "displacement": 0.01}
    elif epoch <= 20:
        return {"dice": 1.2, "cross_entropy": 0.0,
                "lambda_smoothness": 0.1, "lambda_prior": 0.05, "displacement": 0.01}
    else:
        return {"dice": 1.5, "cross_entropy": 0.0,
                "lambda_smoothness": 0.1, "lambda_prior": 0.05, "displacement": 0.01}

# ==============================
# ====== ADDED FOR SURFACE LOSS ======
# Mesh Utilities
# ==============================

def extract_mesh_from_onehot(onehot_seg):
    seg_np = np.argmax(onehot_seg, axis=0)
    verts, faces, _, _ = marching_cubes(seg_np, level=0.5)
    return verts, faces

def warp_mesh_vertices(vertices, flow, device):
    flow = flow[0]
    D, H, W = flow.shape[1:]

    verts = torch.tensor(vertices.copy(), dtype=torch.float32, device=device)

    v_norm = verts.clone()
    v_norm[:, 0] = 2.0 * v_norm[:, 0] / (W - 1) - 1
    v_norm[:, 1] = 2.0 * v_norm[:, 1] / (H - 1) - 1
    v_norm[:, 2] = 2.0 * v_norm[:, 2] / (D - 1) - 1

    v_norm = v_norm.view(1, -1, 1, 1, 3)

    sampled_flow = F.grid_sample(
        flow.unsqueeze(0),
        v_norm,
        align_corners=False
    )

    sampled_flow = sampled_flow.view(3, -1).T
    warped_vertices = verts + sampled_flow

    return warped_vertices.detach().cpu().numpy()

def compute_surface_intersection_loss(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    try:
        intersecting = mesh.self_intersecting_faces
        num_intersections = len(intersecting)
    except:
        num_intersections = 0

    total_faces = len(faces)
    if total_faces == 0:
        return 0.0

    return num_intersections / total_faces

# ====================================

# ----------------- Helpers ----------------- #
train_losses = []
loss_log = []

def jacobian_determinant(flow):
    """
    Compute Jacobian determinant of deformation field.
    flow: (B, 3, D, H, W)
    """
    # central differences
    dz = (flow[:, :, 2:, 1:-1, 1:-1] - flow[:, :, :-2, 1:-1, 1:-1]) / 2
    dy = (flow[:, :, 1:-1, 2:, 1:-1] - flow[:, :, 1:-1, :-2, 1:-1]) / 2
    dx = (flow[:, :, 1:-1, 1:-1, 2:] - flow[:, :, 1:-1, 1:-1, :-2]) / 2

    # identity + grad
    J11 = 1 + dx[:, 0]; J12 = dy[:, 0]; J13 = dz[:, 0]
    J21 = dx[:, 1]; J22 = 1 + dy[:, 1]; J23 = dz[:, 1]
    J31 = dx[:, 2]; J32 = dy[:, 2]; J33 = 1 + dz[:, 2]

    det = (
        J11 * (J22 * J33 - J23 * J32) -
        J12 * (J21 * J33 - J23 * J31) +
        J13 * (J21 * J32 - J22 * J31)
    )
    return det

# Blue (negative) → folds / self-intersections

# Red (positive) → good orientation-preserving mapping


def compare_lambda_effect(moving, fixed, model, stn, device, slice_index=None):
    """
    Shows warped results and Jacobian with and without lambda_map regularization.
    """
    model.eval()
    moving, fixed = moving.to(device), fixed.to(device)
    x = torch.cat([moving, fixed], dim=1)

    with torch.no_grad():
        # Prediction with lambda map
        flow_with, lambda_map = model(x)
        warped_with = stn(moving, flow_with)
        jac_with = jacobian_determinant(flow_with).cpu().numpy()[0]

        # Prediction without lambda map (force uniform weights)
        # Trick: just ignore lambda_map, but still use the same flow prediction.
        # If you want a real "no-regularization" flow, you'd retrain without lambda term.
        flow_without = flow_with.clone().detach()  # same flow
        warped_without = stn(moving, flow_without)
        jac_without = jacobian_determinant(flow_without).cpu().numpy()[0]

    moving_np = moving[0].cpu().numpy()
    warped_with_np = warped_with[0].cpu().numpy()
    warped_without_np = warped_without[0].cpu().numpy()
    fixed_np = fixed[0].cpu().numpy()

    if slice_index is None:
        slice_index = moving_np.shape[1] // 2

    def collapse(x): return np.argmax(x[:, slice_index], axis=0)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: with lambda
    axs[0, 0].imshow(collapse(moving_np), cmap='tab10'); axs[0, 0].set_title("Moving")
    axs[0, 1].imshow(collapse(warped_with_np), cmap='tab10'); axs[0, 1].set_title("Warped (With λ)")
    im1 = axs[0, 2].imshow(jac_with[slice_index], cmap='bwr', vmin=-1, vmax=1)
    axs[0, 2].set_title("Jacobian (With λ)")

    # Bottom row: without lambda
    axs[1, 0].imshow(collapse(fixed_np), cmap='tab10'); axs[1, 0].set_title("Fixed")
    axs[1, 1].imshow(collapse(warped_without_np), cmap='tab10'); axs[1, 1].set_title("Warped (Without λ)")
    im2 = axs[1, 2].imshow(jac_without[slice_index], cmap='bwr', vmin=-1, vmax=1)
    axs[1, 2].set_title("Jacobian (Without λ)")

    for ax in axs.flat:
        ax.axis('off')
    plt.colorbar(im1, ax=axs[:, 2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def dice_score(pred, target, epsilon=1e-5):
    intersection = (pred * target).sum(dim=(2,3,4))
    union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean(dim=1)

best_dice = -1.0
best_model_path = "/content/drive/MyDrive/trained_model.pth"


# ----------------- Training ----------------- #
for epoch in range(1, num_epochs + 1):
    unet.train()
    total_loss = 0
    loss_weights = get_loss_weights(epoch)

    epoch_loss_dict = {k: 0.0 for k in loss_weights.keys()}
    epoch_loss_dict["surface"] = 0.0  # ====== ADDED FOR SURFACE LOSS ======

    print(f"\n--- Epoch {epoch}/{num_epochs} ---")
    for batch_idx, (moving, fixed) in enumerate(train_loader):
        moving, fixed = moving.to(device, non_blocking=True), fixed.to(device, non_blocking=True)
        x = torch.cat([moving, fixed], dim=1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            flow, lambda_map = unet(x)
            warped = stn(moving, flow)
            loss, loss_dict = compound_loss(warped, fixed, flow, lambda_map, loss_weights)
            # ====== ADDED FOR SURFACE LOSS ======
            if batch_idx % surface_compute_interval == 0:
                moving_np = moving[0].detach().cpu().numpy()
                verts, faces = extract_mesh_from_onehot(moving_np)
                warped_verts = warp_mesh_vertices(verts, flow, device)
                surface_loss_value = compute_surface_intersection_loss(warped_verts, faces)
                surface_loss = torch.tensor(surface_loss_value, device=device)
            else:
                surface_loss = torch.tensor(0.0, device=device)

            loss = loss + surface_weight * surface_loss
            loss_dict["surface"] = surface_loss
            # =====================================

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.item()

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    num_batches = len(train_loader)
    avg_loss_dict = {k: v / num_batches for k, v in epoch_loss_dict.items()}

    train_losses.append(avg_loss)
    scheduler.step(avg_loss)

    # --- Evaluation on test set --- #
    unet.eval()
    with torch.no_grad():
        dice_vals = []
        for moving, fixed in test_loader:
            moving, fixed = moving.to(device), fixed.to(device)
            x = torch.cat([moving, fixed], dim=1)
            flow, lambda_map = unet(x)
            warped = stn(moving, flow)
            dice_vals.append(dice_score(warped, fixed).mean().item())
        avg_dice = np.mean(dice_vals)

    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save({
            "epoch": epoch,
            "model_state_dict": unet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dice": best_dice,
            "loss": avg_loss,
        }, best_model_path)
        print(f"🔥 Saved new best model at epoch {epoch} with Dice {best_dice:.4f}")

    # Logging
    loss_break = {k: round(v, 5) for k, v in avg_loss_dict.items()}
    loss_break.update({"epoch": epoch, "dice": round(avg_dice, 5), "total_loss": round(avg_loss, 5)})
    loss_log.append(loss_break)

    print(f"\nEpoch {epoch}/{num_epochs} | Total Loss: {avg_loss:.4f} | Dice (Test): {avg_dice:.4f}")
    print("Loss breakdown:", loss_break)
    print("Loss Weights:", loss_weights)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ----------------- Save + Plot ----------------- #
df = pd.DataFrame(loss_log)
df.to_csv("/content/drive/MyDrive/loss_log_1.csv", index=False)
print("📈 Loss log saved.")

plt.figure(figsize=(12, 6))
for key in loss_log[0].keys():
    if key not in ["epoch", "dice", "total_loss"]:
        plt.plot([e[key] for e in loss_log], label=key)
plt.plot([e["dice"] for e in loss_log], label="dice (↑)", linestyle='--', color='purple')
plt.plot([e["total_loss"] for e in loss_log], label="total_loss", linestyle='--', color='black')
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("Loss Breakdown Over Training")
plt.legend()
plt.grid(True)
plt.show()


unet.eval()
stn.eval()

