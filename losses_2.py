import torch
import torch.nn.functional as F

# ----------------- Core segmentation & similarity losses ----------------- #

def dice_loss(y_pred, y_true, smooth=1e-5):
    """Dice loss for multi-class one-hot predictions."""
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims+2))
    intersection = 2 * (y_true * y_pred).sum(dim=vol_axes)
    union = y_true.sum(dim=vol_axes) + y_pred.sum(dim=vol_axes)
    dice = (intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def cross_entropy_loss(pred, target):
    """Cross entropy for multi-class one-hot labels."""
    return F.cross_entropy(pred, target.argmax(dim=1))

# ----------------- λ-based adaptive regularization losses ----------------- #

def lambda_weighted_smoothness_loss(flow, lambda_map):
    """
    Smoothness loss weighted per voxel by the *average* λ of adjacent voxels.
    flow:       (B, 3, D, H, W)
    lambda_map: (B, 1, D, H, W)
    """
    # x-gradient (average λ between left and right voxels)
    dx = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]) * 0.5 * (
        lambda_map[:, :, 1:, :, :] + lambda_map[:, :, :-1, :, :]
    )
    # y-gradient (average λ between top and bottom voxels)
    dy = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]) * 0.5 * (
        lambda_map[:, :, :, 1:, :] + lambda_map[:, :, :, :-1, :]
    )
    # z-gradient (average λ between front and back voxels)
    dz = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]) * 0.5 * (
        lambda_map[:, :, :, :, 1:] + lambda_map[:, :, :, :, :-1]
    )

    return torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)


def lambda_prior_loss(lambda_map, mean_val=0.5, std_val=0.1):
    """Gaussian prior to keep λ values bounded & interpretable."""
    return torch.mean((lambda_map - mean_val) ** 2 / (2 * std_val ** 2))

def displacement_magnitude_loss(flow):
    """Penalizes voxels moving too far from origin."""
    return torch.mean(flow ** 2)
