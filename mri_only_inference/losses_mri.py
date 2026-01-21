"""
Modular Loss Functions for MRI-based Registration

Combines:
1. MRI intensity losses (NCC, MSE)
2. Segmentation alignment losses (Dice, Focal)
3. Geometric regularization (Smoothness, Jacobian, Bending)
4. Lambda-based adaptive regularization
5. Boundary alignment losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MRI Intensity Losses
# =============================================================================

def ncc_loss(y_pred, y_true, win=9):
    """
    Normalized Cross Correlation loss for MRI intensity matching.
    
    Args:
        y_pred: (B, 1, D, H, W) - warped MRI
        y_true: (B, 1, D, H, W) - target MRI
        win: window size for local NCC
    """
    ndims = 3
    sum_filt = torch.ones([1, 1, win, win, win], device=y_pred.device)
    pad_size = win // 2
    
    I = y_true
    J = y_pred
    
    I2 = I * I
    J2 = J * J
    IJ = I * J
    
    I_sum = F.conv3d(I, sum_filt, padding=pad_size)
    J_sum = F.conv3d(J, sum_filt, padding=pad_size)
    I2_sum = F.conv3d(I2, sum_filt, padding=pad_size)
    J2_sum = F.conv3d(J2, sum_filt, padding=pad_size)
    IJ_sum = F.conv3d(IJ, sum_filt, padding=pad_size)
    
    win_size = win ** ndims
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    
    cc = cross * cross / (I_var * J_var + 1e-5)
    
    return 1 - torch.mean(cc)


def mse_loss(y_pred, y_true):
    """Mean Squared Error for MRI intensity matching."""
    return F.mse_loss(y_pred, y_true)


# =============================================================================
# Segmentation Alignment Losses
# =============================================================================

def dice_loss(y_pred, y_true, smooth=1e-5):
    """
    Dice loss for segmentation overlap.
    
    Args:
        y_pred: (B, C, D, H, W) - warped segmentation
        y_true: (B, C, D, H, W) - target segmentation
    """
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims + 2))
    
    intersection = (y_pred * y_true).sum(dim=vol_axes)
    union = y_pred.sum(dim=vol_axes) + y_true.sum(dim=vol_axes)
    
    dice_score = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice_score.mean()


def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    """
    y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
    ce_loss = -y_true * torch.log(y_pred)
    focal_weight = (1 - y_pred) ** gamma
    return (alpha * focal_weight * ce_loss).mean()


def boundary_loss(y_pred, y_true):
    """
    Boundary alignment loss - emphasizes alignment at anatomical boundaries.
    """
    def compute_boundary(seg):
        grad_x = seg[:, :, :, :, 1:] - seg[:, :, :, :, :-1]
        grad_y = seg[:, :, :, 1:, :] - seg[:, :, :, :-1, :]
        grad_z = seg[:, :, 1:, :, :] - seg[:, :, :-1, :, :]
        
        grad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        grad_z = F.pad(grad_z, (0, 0, 0, 0, 0, 1))
        
        return torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
    
    pred_boundary = compute_boundary(y_pred)
    true_boundary = compute_boundary(y_true)
    
    return F.mse_loss(pred_boundary, true_boundary)


# =============================================================================
# Geometric Regularization Losses
# =============================================================================

def smoothness_loss(flow):
    """First-order smoothness regularization."""
    dx = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    
    return torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)


def bending_energy_loss(flow):
    """Second-order smoothness (bending energy)."""
    d2x = flow[:, :, 2:, :, :] - 2*flow[:, :, 1:-1, :, :] + flow[:, :, :-2, :, :]
    d2y = flow[:, :, :, 2:, :] - 2*flow[:, :, :, 1:-1, :] + flow[:, :, :, :-2, :]
    d2z = flow[:, :, :, :, 2:] - 2*flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]
    
    return torch.mean(d2x**2) + torch.mean(d2y**2) + torch.mean(d2z**2)


def jacobian_det_loss(flow):
    """
    Jacobian determinant loss - penalizes folding (negative determinants).
    """
    # Compute spatial gradients
    dx_dx = flow[:, 0, 1:, :, :] - flow[:, 0, :-1, :, :]
    dx_dy = flow[:, 0, :, 1:, :] - flow[:, 0, :, :-1, :]
    dx_dz = flow[:, 0, :, :, 1:] - flow[:, 0, :, :, :-1]
    
    dy_dx = flow[:, 1, 1:, :, :] - flow[:, 1, :-1, :, :]
    dy_dy = flow[:, 1, :, 1:, :] - flow[:, 1, :, :-1, :]
    dy_dz = flow[:, 1, :, :, 1:] - flow[:, 1, :, :, :-1]
    
    dz_dx = flow[:, 2, 1:, :, :] - flow[:, 2, :-1, :, :]
    dz_dy = flow[:, 2, :, 1:, :] - flow[:, 2, :, :-1, :]
    dz_dz = flow[:, 2, :, :, 1:] - flow[:, 2, :, :, :-1]
    
    # Pad to maintain size
    dx_dx = F.pad(dx_dx, (0, 0, 0, 0, 0, 1))
    dx_dy = F.pad(dx_dy, (0, 0, 0, 1, 0, 0))
    dx_dz = F.pad(dx_dz, (0, 1, 0, 0, 0, 0))
    dy_dx = F.pad(dy_dx, (0, 0, 0, 0, 0, 1))
    dy_dy = F.pad(dy_dy, (0, 0, 0, 1, 0, 0))
    dy_dz = F.pad(dy_dz, (0, 1, 0, 0, 0, 0))
    dz_dx = F.pad(dz_dx, (0, 0, 0, 0, 0, 1))
    dz_dy = F.pad(dz_dy, (0, 0, 0, 1, 0, 0))
    dz_dz = F.pad(dz_dz, (0, 1, 0, 0, 0, 0))
    
    # Add identity
    dx_dx = dx_dx + 1.0
    dy_dy = dy_dy + 1.0
    dz_dz = dz_dz + 1.0
    
    # Compute determinant
    det = (dx_dx * (dy_dy * dz_dz - dy_dz * dz_dy) -
           dx_dy * (dy_dx * dz_dz - dy_dz * dz_dx) +
           dx_dz * (dy_dx * dz_dy - dy_dy * dz_dx))
    
    # Penalize negative determinants
    return F.relu(-det).mean()


def displacement_loss(flow):
    """Penalizes large displacements."""
    return torch.mean(flow ** 2)


# =============================================================================
# Lambda-based Adaptive Regularization
# =============================================================================

def lambda_weighted_smoothness(flow, lambda_map):
    """
    Spatially-adaptive smoothness using lambda map.
    High lambda = more smoothness, low lambda = more flexibility.
    """
    dx = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]) * 0.5 * (
        lambda_map[:, :, 1:, :, :] + lambda_map[:, :, :-1, :, :]
    )
    dy = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]) * 0.5 * (
        lambda_map[:, :, :, 1:, :] + lambda_map[:, :, :, :-1, :]
    )
    dz = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]) * 0.5 * (
        lambda_map[:, :, :, :, 1:] + lambda_map[:, :, :, :, :-1]
    )
    
    return torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)


def lambda_prior_loss(lambda_map, mean_val=0.5, std_val=0.2):
    """Gaussian prior on lambda values."""
    return torch.mean((lambda_map - mean_val) ** 2 / (2 * std_val ** 2))


# =============================================================================
# Multi-scale Consistency
# =============================================================================

def multi_scale_consistency_loss(intermediate_flows):
    """Ensures consistency between multi-scale flows."""
    if len(intermediate_flows) < 2:
        return torch.tensor(0.0, device=intermediate_flows[0].device)
    
    total_loss = 0.0
    for i in range(len(intermediate_flows) - 1):
        coarse = intermediate_flows[i]
        fine = intermediate_flows[i + 1]
        
        upsampled = F.interpolate(coarse, size=fine.shape[2:], mode='trilinear', align_corners=False)
        total_loss += F.mse_loss(upsampled, fine)
    
    return total_loss / (len(intermediate_flows) - 1)


# =============================================================================
# Comprehensive Loss Class
# =============================================================================

class MRIRegistrationLoss(nn.Module):
    """
    Comprehensive loss for MRI-based registration.
    
    Combines MRI intensity matching with segmentation evaluation.
    """
    def __init__(self, weights=None):
        super().__init__()
        
        # Default weights - tuned for MRI registration
        self.weights = weights if weights is not None else {
            # MRI intensity losses
            'ncc': 1.0,                    # Primary MRI matching
            'mse': 0.0,                    # Optional MSE (usually 0)
            
            # Segmentation evaluation losses
            'dice': 0.5,                   # Segmentation alignment
            'focal': 0.0,                  # Hard examples (optional)
            'boundary': 0.1,               # Boundary alignment
            
            # Geometric regularization
            'smoothness': 0.01,            # First-order smoothness
            'bending': 0.001,              # Second-order smoothness
            'jacobian': 0.1,               # Prevent folding
            'displacement': 0.001,         # Prevent large deformations
            
            # Lambda-based adaptive
            'lambda_smoothness': 0.05,     # Adaptive smoothness
            'lambda_prior': 0.01,          # Lambda regularization
            
            # Multi-scale
            'multi_scale': 0.01,           # Scale consistency
        }
    
    def forward(self, warped_mri, sample_mri, warped_seg, sample_seg, 
                final_flow, intermediate_flows, lambda_maps, return_components=False):
        """
        Compute comprehensive loss.
        
        Args:
            warped_mri: (B, 1, D, H, W) - warped template MRI
            sample_mri: (B, 1, D, H, W) - target MRI
            warped_seg: (B, 5, D, H, W) - warped template segmentation
            sample_seg: (B, 5, D, H, W) - target segmentation (for evaluation)
            final_flow: (B, 3, D, H, W) - final deformation field
            intermediate_flows: list of multi-scale flows
            lambda_maps: list of lambda maps
            return_components: whether to return individual losses
        """
        loss_dict = {}
        
        # MRI intensity losses
        loss_dict['ncc'] = ncc_loss(warped_mri, sample_mri)
        loss_dict['mse'] = mse_loss(warped_mri, sample_mri)
        
        # Segmentation evaluation losses
        loss_dict['dice'] = dice_loss(warped_seg, sample_seg)
        loss_dict['focal'] = focal_loss(warped_seg, sample_seg)
        loss_dict['boundary'] = boundary_loss(warped_seg, sample_seg)
        
        # Geometric regularization
        loss_dict['smoothness'] = smoothness_loss(final_flow)
        loss_dict['bending'] = bending_energy_loss(final_flow)
        loss_dict['jacobian'] = jacobian_det_loss(final_flow)
        loss_dict['displacement'] = displacement_loss(final_flow)
        
        # Lambda-based adaptive regularization
        if lambda_maps is not None and len(lambda_maps) > 0:
            final_lambda = lambda_maps[-1]
            loss_dict['lambda_smoothness'] = lambda_weighted_smoothness(final_flow, final_lambda)
            loss_dict['lambda_prior'] = lambda_prior_loss(final_lambda)
        else:
            loss_dict['lambda_smoothness'] = torch.tensor(0.0, device=final_flow.device)
            loss_dict['lambda_prior'] = torch.tensor(0.0, device=final_flow.device)
        
        # Multi-scale consistency
        loss_dict['multi_scale'] = multi_scale_consistency_loss(intermediate_flows)
        
        # Compute total weighted loss
        total_loss = sum(self.weights[k] * loss_dict[k] for k in loss_dict.keys())
        
        if return_components:
            return total_loss, loss_dict
        return total_loss


# =============================================================================
# Utility: Dice Score (for evaluation)
# =============================================================================

def compute_dice_score(y_pred, y_true, num_classes=5, epsilon=1e-5):
    """
    Compute per-class Dice scores for evaluation.
    
    Returns:
        dice_per_class: list of dice scores for each class
        mean_dice: average dice score
    """
    dice_per_class = []
    vol_axes = [2, 3, 4]
    
    for c in range(num_classes):
        pred_c = y_pred[:, c:c+1, ...]
        true_c = y_true[:, c:c+1, ...]
        
        intersection = 2 * (pred_c * true_c).sum(dim=vol_axes)
        union = pred_c.sum(dim=vol_axes) + true_c.sum(dim=vol_axes)
        dice = (intersection + epsilon) / (union + epsilon)
        dice_per_class.append(dice.mean().item())
    
    return dice_per_class, sum(dice_per_class) / len(dice_per_class)