"""
Comprehensive Visualization Script for MRI Registration Model

Analyzes model performance across multiple dimensions:
1. Registration Quality: MRI/Segmentation before vs after
2. Deformation Analysis: Flow fields, Jacobian, smoothness
3. Attention & Lambda Maps: Model interpretability
4. Per-Class Performance: Where the model excels/fails
5. Error Analysis: Spatial error distribution
6. Aggregate Statistics: Dataset-wide performance

Usage:
    python viz.py --checkpoint /path/to/checkpoint.pth --output /path/to/output_dir
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import random

# Local imports
from model_mri import MRIRegistrationNet, SpatialTransformer
from losses_mri import (
    MRIRegistrationLoss, compute_dice_score, 
    ncc_loss, jacobian_det_loss, smoothness_loss, bending_energy_loss
)


def compute_self_intersection_loss(flow):
    """
    Compute self-intersection (folding) loss from deformation field.
    
    Self-intersection occurs when the Jacobian determinant becomes negative,
    indicating that the transformation is no longer diffeomorphic (one-to-one).
    This means different parts of the image are mapped to the same location,
    causing "folding" in the deformation grid.
    
    Args:
        flow: (B, 3, D, H, W) deformation field tensor
        
    Returns:
        dict with:
            - 'loss': Mean penalty for negative Jacobian (same as jacobian_det_loss)
            - 'folding_percentage': Percentage of voxels with negative Jacobian
            - 'num_folding_voxels': Total number of voxels with self-intersection
            - 'min_jacobian': Minimum Jacobian value (most severe folding)
            - 'mean_jacobian': Mean Jacobian value
    """
    import torch
    import torch.nn.functional as F
    
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow)
    
    if flow.ndim == 4:  # (3, D, H, W) -> (1, 3, D, H, W)
        flow = flow.unsqueeze(0)
    
    # Compute spatial gradients for Jacobian matrix
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
    
    # Add identity (Jacobian of identity transform is I)
    dx_dx = dx_dx + 1.0
    dy_dy = dy_dy + 1.0
    dz_dz = dz_dz + 1.0
    
    # Compute determinant: det(J) = det([[dx_dx, dx_dy, dx_dz],
    #                                     [dy_dx, dy_dy, dy_dz],
    #                                     [dz_dx, dz_dy, dz_dz]])
    det = (dx_dx * (dy_dy * dz_dz - dy_dz * dz_dy) -
           dx_dy * (dy_dx * dz_dz - dy_dz * dz_dx) +
           dx_dz * (dy_dx * dz_dy - dy_dy * dz_dx))
    
    # Self-intersection loss: penalize negative determinants
    # Using ReLU(-det) means only negative values contribute to loss
    self_intersection_loss = F.relu(-det).mean()
    
    # Compute statistics
    det_np = det.detach().cpu().numpy()
    folding_mask = det_np < 0
    num_folding_voxels = folding_mask.sum()
    total_voxels = det_np.size
    folding_percentage = (num_folding_voxels / total_voxels) * 100
    
    return {
        'loss': self_intersection_loss.item() if isinstance(self_intersection_loss, torch.Tensor) else self_intersection_loss,
        'folding_percentage': folding_percentage,
        'num_folding_voxels': int(num_folding_voxels),
        'total_voxels': int(total_voxels),
        'min_jacobian': float(det_np.min()),
        'max_jacobian': float(det_np.max()),
        'mean_jacobian': float(det_np.mean()),
    }
from get_data_mri import MRIDataset

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# Class labels for OASIS brain segmentation
CLASS_NAMES = ['Background', 'CSF', 'Gray Matter', 'White Matter', 'Deep GM']
CLASS_COLORS = ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#533483']


# =============================================================================
# Utility Functions
# =============================================================================

def to_numpy(tensor):
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def normalize_for_display(img, percentile=99):
    """Normalize image for display with percentile clipping."""
    img = to_numpy(img)
    vmin = np.percentile(img, 100 - percentile)
    vmax = np.percentile(img, percentile)
    if vmax - vmin > 0:
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return img


def get_slice_indices(volume_shape, slices='center'):
    """Get slice indices for visualization."""
    D, H, W = volume_shape[-3:]
    if slices == 'center':
        return D // 2, H // 2, W // 2
    elif isinstance(slices, (list, tuple)):
        return slices
    return D // 2, H // 2, W // 2


def seg_to_rgb(seg_onehot, alpha=0.7):
    """Convert one-hot segmentation to RGB image."""
    seg = to_numpy(seg_onehot)
    
    # Handle different input shapes
    if seg.ndim == 4:  # (C, D, H, W) - full volume
        seg = seg.argmax(axis=0)
    elif seg.ndim == 3:  # (C, H, W) - single slice with channels
        seg = seg.argmax(axis=0)
    # If ndim == 2, it's already a label map (H, W)
    
    rgb = np.zeros((*seg.shape, 4))
    for i, color in enumerate(CLASS_COLORS):
        mask = seg == i
        rgba = mcolors.to_rgba(color)
        rgb[mask] = rgba
    rgb[..., 3] = alpha * (seg > 0).astype(float) + 0.3 * (seg == 0).astype(float)
    return rgb


def compute_jacobian_determinant(flow):
    """Compute Jacobian determinant of deformation field."""
    flow = to_numpy(flow)
    if flow.ndim == 5:  # (B, 3, D, H, W)
        flow = flow[0]
    
    # Compute gradients
    dx = np.gradient(flow[0], axis=0)
    dy = np.gradient(flow[0], axis=1)
    dz = np.gradient(flow[0], axis=2)
    
    d_dx = np.gradient(flow[1], axis=0)
    d_dy = np.gradient(flow[1], axis=1)
    d_dz = np.gradient(flow[1], axis=2)
    
    d_x = np.gradient(flow[2], axis=0)
    d_y = np.gradient(flow[2], axis=1)
    d_z = np.gradient(flow[2], axis=2)
    
    # Add identity
    dx += 1
    d_dy += 1
    d_z += 1
    
    # Compute determinant
    det = (dx * (d_dy * d_z - d_dz * d_y) -
           dy * (d_dx * d_z - d_dz * d_x) +
           dz * (d_dx * d_y - d_dy * d_x))
    
    return det


def compute_flow_magnitude(flow):
    """Compute magnitude of flow field."""
    flow = to_numpy(flow)
    if flow.ndim == 5:
        flow = flow[0]
    return np.sqrt(np.sum(flow ** 2, axis=0))


# =============================================================================
# Single Sample Visualization
# =============================================================================

class SampleVisualizer:
    """Visualize results for a single sample."""
    
    def __init__(self, output_dir, figsize_scale=1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize_scale = figsize_scale
    
    def plot_registration_overview(self, data, results, sample_idx, save=True):
        """
        Main overview showing registration quality.
        Shows: Template MRI, Sample MRI, Warped MRI, Difference maps
        """
        fig = plt.figure(figsize=(20 * self.figsize_scale, 16 * self.figsize_scale))
        gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.2)
        
        # Extract data
        template_mri = to_numpy(data['template_mri'][0, 0])
        sample_mri = to_numpy(data['sample_mri'][0, 0])
        warped_mri = to_numpy(results['warped_mri'][0, 0])
        
        template_seg = data['template_seg'][0]
        sample_seg = data['sample_seg'][0]
        warped_seg = results['warped_seg'][0]
        
        # Get center slices
        d, h, w = get_slice_indices(template_mri.shape)
        
        # Color setup
        cmap_mri = 'gray'
        cmap_diff = 'RdBu_r'
        
        # Row 1: Axial slices (MRI)
        titles_row1 = ['Template MRI', 'Sample MRI (Target)', 'Warped MRI', 
                       'Template vs Sample', 'Warped vs Sample']
        
        for i, (title, img) in enumerate(zip(titles_row1, [
            template_mri[d], sample_mri[d], warped_mri[d],
            template_mri[d] - sample_mri[d], warped_mri[d] - sample_mri[d]
        ])):
            ax = fig.add_subplot(gs[0, i])
            if i < 3:
                ax.imshow(img, cmap=cmap_mri, vmin=0, vmax=1)
            else:
                im = ax.imshow(img, cmap=cmap_diff, vmin=-0.5, vmax=0.5)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Row 2: Segmentation overlays (Axial)
        titles_row2 = ['Template Seg', 'Sample Seg (GT)', 'Warped Seg', 
                       'Overlay: Warped on Sample', 'Error Map']
        
        for i, title in enumerate(titles_row2):
            ax = fig.add_subplot(gs[1, i])
            
            if i == 0:
                ax.imshow(sample_mri[d], cmap='gray', alpha=0.3)
                ax.imshow(seg_to_rgb(template_seg[:, d]))
            elif i == 1:
                ax.imshow(sample_mri[d], cmap='gray', alpha=0.3)
                ax.imshow(seg_to_rgb(sample_seg[:, d]))
            elif i == 2:
                ax.imshow(sample_mri[d], cmap='gray', alpha=0.3)
                ax.imshow(seg_to_rgb(warped_seg[:, d]))
            elif i == 3:
                # Overlay warped on sample with transparency
                ax.imshow(sample_mri[d], cmap='gray')
                warped_labels = to_numpy(warped_seg).argmax(axis=0)[d]
                sample_labels = to_numpy(sample_seg).argmax(axis=0)[d]
                ax.contour(warped_labels, colors='cyan', linewidths=0.5, alpha=0.8)
                ax.contour(sample_labels, colors='magenta', linewidths=0.5, alpha=0.8)
            else:
                # Error map
                warped_labels = to_numpy(warped_seg).argmax(axis=0)[d]
                sample_labels = to_numpy(sample_seg).argmax(axis=0)[d]
                error = (warped_labels != sample_labels).astype(float)
                im = ax.imshow(error, cmap='hot', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Row 3: Coronal view
        for i, (title, img) in enumerate(zip(
            ['Template (Coronal)', 'Sample (Coronal)', 'Warped (Coronal)', 
             'Template Seg', 'Warped Seg'],
            [template_mri[:, h], sample_mri[:, h], warped_mri[:, h], 
             template_seg[:, :, h], warped_seg[:, :, h]]
        )):
            ax = fig.add_subplot(gs[2, i])
            if i < 3:
                ax.imshow(img, cmap=cmap_mri, vmin=0, vmax=1, aspect='auto')
            else:
                ax.imshow(sample_mri[:, h], cmap='gray', alpha=0.3, aspect='auto')
                ax.imshow(seg_to_rgb(img), aspect='auto')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Row 4: Sagittal view
        for i, (title, img) in enumerate(zip(
            ['Template (Sagittal)', 'Sample (Sagittal)', 'Warped (Sagittal)',
             'Template Seg', 'Warped Seg'],
            [template_mri[:, :, w], sample_mri[:, :, w], warped_mri[:, :, w],
             template_seg[:, :, :, w], warped_seg[:, :, :, w]]
        )):
            ax = fig.add_subplot(gs[3, i])
            if i < 3:
                ax.imshow(img, cmap=cmap_mri, vmin=0, vmax=1, aspect='auto')
            else:
                ax.imshow(sample_mri[:, :, w], cmap='gray', alpha=0.3, aspect='auto')
                ax.imshow(seg_to_rgb(img), aspect='auto')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Add metrics
        dice_per_class = results['dice_per_class']
        mean_dice = np.mean(dice_per_class)
        metrics_text = f"Mean Dice: {mean_dice:.4f} | " + \
                       " | ".join([f"{CLASS_NAMES[i]}: {dice_per_class[i]:.3f}" 
                                  for i in range(len(CLASS_NAMES))])
        fig.suptitle(f"Registration Overview - Sample {sample_idx}\n{metrics_text}", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            fig.savefig(self.output_dir / f"sample_{sample_idx:03d}_overview.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig
    
    def plot_deformation_analysis(self, data, results, sample_idx, save=True):
        """
        Analyze deformation field quality.
        Shows: Flow magnitude, Jacobian determinant, flow vectors
        """
        fig = plt.figure(figsize=(20 * self.figsize_scale, 12 * self.figsize_scale))
        gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.25)
        
        flow = to_numpy(results['flow'])
        sample_mri = to_numpy(data['sample_mri'][0, 0])
        
        # Compute derived quantities
        flow_mag = compute_flow_magnitude(flow)
        jacobian = compute_jacobian_determinant(flow)
        
        d, h, w = get_slice_indices(sample_mri.shape)
        
        # Row 1: Flow components (Axial)
        flow_single = flow[0] if flow.ndim == 5 else flow
        for i, (title, component) in enumerate(zip(
            ['Flow X (L-R)', 'Flow Y (A-P)', 'Flow Z (S-I)', 'Flow Magnitude', 'Jacobian Det'],
            [flow_single[0, d], flow_single[1, d], flow_single[2, d], 
             flow_mag[d], jacobian[d]]
        )):
            ax = fig.add_subplot(gs[0, i])
            if i < 3:
                vmax = np.abs(component).max()
                im = ax.imshow(component, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            elif i == 3:
                im = ax.imshow(component, cmap='hot')
            else:
                im = ax.imshow(component, cmap='RdYlGn', vmin=0, vmax=2)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Row 2: Flow vectors overlaid on MRI
        ax = fig.add_subplot(gs[1, 0:2])
        ax.imshow(sample_mri[d], cmap='gray', vmin=0, vmax=1)
        
        # Subsample for quiver plot
        step = 4
        y, x = np.mgrid[0:sample_mri.shape[1]:step, 0:sample_mri.shape[2]:step]
        u = flow_single[0, d, ::step, ::step]
        v = flow_single[1, d, ::step, ::step]
        
        mag = np.sqrt(u**2 + v**2)
        ax.quiver(x, y, u, v, mag, cmap='plasma', scale=3, width=0.003, alpha=0.8)
        ax.set_title('Deformation Vectors (Axial)', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Coronal view
        ax = fig.add_subplot(gs[1, 2:4])
        ax.imshow(sample_mri[:, h], cmap='gray', vmin=0, vmax=1, aspect='auto')
        y, x = np.mgrid[0:sample_mri.shape[0]:step, 0:sample_mri.shape[2]:step]
        u = flow_single[0, ::step, h, ::step]
        v = flow_single[2, ::step, h, ::step]
        mag = np.sqrt(u**2 + v**2)
        ax.quiver(x, y, u, v, mag, cmap='plasma', scale=3, width=0.003, alpha=0.8)
        ax.set_title('Deformation Vectors (Coronal)', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Jacobian statistics
        ax = fig.add_subplot(gs[1, 4])
        jacobian_flat = jacobian.flatten()
        ax.hist(jacobian_flat, bins=100, color='#0f3460', edgecolor='#e94560', alpha=0.8)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Folding threshold')
        ax.axvline(x=1, color='lime', linestyle='--', linewidth=2, label='No deformation')
        ax.set_xlabel('Jacobian Determinant')
        ax.set_ylabel('Frequency')
        ax.set_title('Jacobian Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        
        # Row 3: Multi-scale flow analysis
        if 'intermediate_flows' in results and results['intermediate_flows'] is not None:
            for i, inter_flow in enumerate(results['intermediate_flows'][:4]):
                ax = fig.add_subplot(gs[2, i])
                inter_flow_np = to_numpy(inter_flow)
                if inter_flow_np.ndim == 5:
                    inter_flow_np = inter_flow_np[0]
                inter_mag = compute_flow_magnitude(inter_flow_np)
                
                # Get center slice at this scale
                d_scale = inter_mag.shape[0] // 2
                im = ax.imshow(inter_mag[d_scale], cmap='hot')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f'Flow Scale {i+1} ({inter_mag.shape})', fontsize=10, fontweight='bold')
                ax.axis('off')
        
        # Statistics text
        ax = fig.add_subplot(gs[2, 4])
        ax.axis('off')
        
        folding_pct = (jacobian < 0).mean() * 100
        stats_text = f"""
Deformation Statistics
──────────────────────
Flow Magnitude:
  Mean: {flow_mag.mean():.4f}
  Max: {flow_mag.max():.4f}
  Std: {flow_mag.std():.4f}

Jacobian Determinant:
  Mean: {jacobian.mean():.4f}
  Min: {jacobian.min():.4f}
  Max: {jacobian.max():.4f}
  Folding: {folding_pct:.2f}%

Smoothness: {results.get('smoothness', 0):.6f}
Bending: {results.get('bending', 0):.6f}
"""
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
        
        fig.suptitle(f'Deformation Field Analysis - Sample {sample_idx}', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            fig.savefig(self.output_dir / f"sample_{sample_idx:03d}_deformation.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig
    
    def plot_grid_warp(self, data, results, sample_idx, save=True):
        """
        Grid Warp Visualization - Shows deformation field applied to synthetic grid.
        
        Visual check for folding/self-intersections:
        - Grid lines should be curved but smooth
        - If lines cross each other or look "tangled", regularization is failing
        """
        fig = plt.figure(figsize=(20 * self.figsize_scale, 8 * self.figsize_scale))
        gs = GridSpec(2, 5, figure=fig, hspace=0.25, wspace=0.2)
        
        flow = to_numpy(results['flow'])
        if flow.ndim == 5:
            flow = flow[0]
        
        sample_mri = to_numpy(data['sample_mri'][0, 0])
        jacobian = compute_jacobian_determinant(results['flow'])
        
        d, h, w = get_slice_indices(sample_mri.shape)
        
        # Create synthetic grid
        grid_spacing = 8
        H, W = sample_mri.shape[1], sample_mri.shape[2]
        
        def create_grid_image(height, width, spacing):
            """Create a checkerboard/grid pattern."""
            grid = np.zeros((height, width))
            # Horizontal lines
            for i in range(0, height, spacing):
                grid[i:i+1, :] = 1
            # Vertical lines
            for j in range(0, width, spacing):
                grid[:, j:j+1] = 1
            return grid
        
        def warp_grid_2d(grid, flow_x, flow_y):
            """Warp a 2D grid using flow field."""
            H, W = grid.shape
            y, x = np.mgrid[0:H, 0:W].astype(np.float32)
            
            # Scale flow to pixel coordinates (flow is in normalized coords [-1, 1])
            scale_x = W / 2
            scale_y = H / 2
            
            new_x = x + flow_x * scale_x
            new_y = y + flow_y * scale_y
            
            # Clip to valid range
            new_x = np.clip(new_x, 0, W - 1)
            new_y = np.clip(new_y, 0, H - 1)
            
            from scipy.ndimage import map_coordinates
            warped = map_coordinates(grid, [new_y, new_x], order=1, mode='constant')
            return warped
        
        # Row 1: Axial view
        grid_ax = create_grid_image(H, W, grid_spacing)
        flow_x_ax = flow[0, d]  # X component at axial slice
        flow_y_ax = flow[1, d]  # Y component at axial slice
        warped_grid_ax = warp_grid_2d(grid_ax, flow_x_ax, flow_y_ax)
        
        # Original grid
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(grid_ax, cmap='gray')
        ax.set_title('Original Grid', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Warped grid
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(warped_grid_ax, cmap='gray')
        ax.set_title('Warped Grid (Axial)', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Warped grid overlaid on MRI
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(sample_mri[d], cmap='gray', vmin=0, vmax=1)
        ax.imshow(warped_grid_ax, cmap='Reds', alpha=0.5)
        ax.set_title('Grid on MRI', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Jacobian determinant
        ax = fig.add_subplot(gs[0, 3])
        im = ax.imshow(jacobian[d], cmap='RdYlGn', vmin=0, vmax=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Jacobian Det (Axial)\n<0 = Folding', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Folding regions highlighted
        ax = fig.add_subplot(gs[0, 4])
        ax.imshow(sample_mri[d], cmap='gray', vmin=0, vmax=1)
        folding_mask = jacobian[d] < 0
        if folding_mask.any():
            ax.imshow(np.ma.masked_where(~folding_mask, folding_mask), 
                     cmap='Reds', alpha=0.8, vmin=0, vmax=1)
        ax.set_title(f'Folding Regions\n({folding_mask.sum()} voxels)', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Row 2: Coronal view
        D_vol = sample_mri.shape[0]
        grid_cor = create_grid_image(D_vol, W, grid_spacing)
        flow_x_cor = flow[0, :, h, :]
        flow_z_cor = flow[2, :, h, :]
        warped_grid_cor = warp_grid_2d(grid_cor, flow_x_cor, flow_z_cor)
        
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(grid_cor, cmap='gray', aspect='auto')
        ax.set_title('Original Grid', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(warped_grid_cor, cmap='gray', aspect='auto')
        ax.set_title('Warped Grid (Coronal)', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 2])
        ax.imshow(sample_mri[:, h], cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax.imshow(warped_grid_cor, cmap='Reds', alpha=0.5, aspect='auto')
        ax.set_title('Grid on MRI', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 3])
        im = ax.imshow(jacobian[:, h], cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Jacobian Det (Coronal)', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Statistics
        ax = fig.add_subplot(gs[1, 4])
        ax.axis('off')
        
        folding_pct = (jacobian < 0).mean() * 100
        jac_stats = f"""
Deformation Quality
───────────────────
Jacobian Statistics:
  Mean: {jacobian.mean():.4f}
  Min:  {jacobian.min():.4f}
  Max:  {jacobian.max():.4f}

Folding Analysis:
  Folding %: {folding_pct:.3f}%
  Folding voxels: {(jacobian < 0).sum()}

Interpretation:
  ✓ Jac > 0: Valid deformation
  ✗ Jac < 0: Self-intersection
  Jac ≈ 1: No volume change
"""
        color = '#2ecc71' if folding_pct < 0.1 else ('#f39c12' if folding_pct < 1 else '#e74c3c')
        ax.text(0.05, 0.95, jac_stats, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        fig.suptitle(f'Grid Warp & Folding Analysis - Sample {sample_idx}',
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            fig.savefig(self.output_dir / f"sample_{sample_idx:03d}_grid_warp.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig
    
    def plot_template_vs_sample(self, data, results, sample_idx, save=True):
        """
        Simplified Template vs Sample comparison.
        Shows the registration problem clearly: how different is template from sample,
        and how well does warping bridge the gap.
        """
        fig = plt.figure(figsize=(18 * self.figsize_scale, 10 * self.figsize_scale))
        gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.2)
        
        template_mri = to_numpy(data['template_mri'][0, 0])
        sample_mri = to_numpy(data['sample_mri'][0, 0])
        warped_mri = to_numpy(results['warped_mri'][0, 0])
        
        template_seg = to_numpy(data['template_seg'][0])
        sample_seg = to_numpy(data['sample_seg'][0])
        warped_seg = to_numpy(results['warped_seg'][0])
        
        d, h, w = get_slice_indices(sample_mri.shape)
        
        # Compute initial dice (template vs sample, no registration)
        initial_dice_per_class = []
        for c in range(template_seg.shape[0]):
            intersection = 2 * (template_seg[c] * sample_seg[c]).sum()
            union = template_seg[c].sum() + sample_seg[c].sum()
            initial_dice_per_class.append((intersection + 1e-5) / (union + 1e-5))
        initial_dice = np.mean(initial_dice_per_class)
        
        final_dice = np.mean(results['dice_per_class'])
        improvement = final_dice - initial_dice
        
        # Row 1: MRI comparison
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(template_mri[d], cmap='gray', vmin=0, vmax=1)
        ax.set_title('Template MRI', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(sample_mri[d], cmap='gray', vmin=0, vmax=1)
        ax.set_title('Sample MRI (Target)', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(warped_mri[d], cmap='gray', vmin=0, vmax=1)
        ax.set_title('Warped Template', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Difference: before vs after
        ax = fig.add_subplot(gs[0, 3])
        diff_before = np.abs(template_mri[d] - sample_mri[d])
        diff_after = np.abs(warped_mri[d] - sample_mri[d])
        # Show improvement: green where it got better, red where worse
        improvement_map = diff_before - diff_after
        im = ax.imshow(improvement_map, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Improvement Map\n(Green=Better)', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Row 2: Segmentation comparison
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(sample_mri[d], cmap='gray', alpha=0.3)
        ax.imshow(seg_to_rgb(template_seg[:, d]))
        ax.set_title(f'Template Seg\n(Init Dice: {initial_dice:.3f})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(sample_mri[d], cmap='gray', alpha=0.3)
        ax.imshow(seg_to_rgb(sample_seg[:, d]))
        ax.set_title('Sample Seg (GT)', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 2])
        ax.imshow(sample_mri[d], cmap='gray', alpha=0.3)
        ax.imshow(seg_to_rgb(warped_seg[:, d]))
        ax.set_title(f'Warped Seg\n(Final Dice: {final_dice:.3f})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Summary statistics
        ax = fig.add_subplot(gs[1, 3])
        ax.axis('off')
        
        # Per-class comparison
        class_text = "Per-Class Dice Scores:\n" + "─" * 25 + "\n"
        class_text += f"{'Class':<12} {'Initial':>8} {'Final':>8} {'Δ':>8}\n"
        class_text += "─" * 25 + "\n"
        for c, name in enumerate(CLASS_NAMES):
            delta = results['dice_per_class'][c] - initial_dice_per_class[c]
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            class_text += f"{name:<12} {initial_dice_per_class[c]:>8.3f} {results['dice_per_class'][c]:>8.3f} {delta_str:>8}\n"
        class_text += "─" * 25 + "\n"
        class_text += f"{'MEAN':<12} {initial_dice:>8.3f} {final_dice:>8.3f} {'+' if improvement >= 0 else ''}{improvement:.3f}\n"
        
        ax.text(0.05, 0.95, class_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
        
        fig.suptitle(f'Template vs Sample Analysis - Sample {sample_idx}',
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            fig.savefig(self.output_dir / f"sample_{sample_idx:03d}_template_vs_sample.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig, initial_dice, final_dice


# =============================================================================
# Aggregate Analysis
# =============================================================================

class AggregateAnalyzer:
    """Analyze performance across entire dataset."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.all_dice_scores = []
        self.all_dice_per_class = []
        self.all_ncc_scores = []
        self.all_jacobian_stats = []
        self.all_smoothness = []
        self.all_flow_stats = []
        self.sample_indices = []
        
        # New metrics for clinical validity
        self.all_initial_dice = []  # Pre-registration dice
        self.all_initial_dice_per_class = []
        self.all_volumes_gt = []  # Ground truth volumes per class
        self.all_volumes_pred = []  # Predicted (warped) volumes per class
    
    def add_sample(self, sample_idx, results, data, initial_dice=None, initial_dice_per_class=None):
        """Add metrics from a single sample."""
        self.sample_indices.append(sample_idx)
        
        # Dice scores
        dice_per_class = results['dice_per_class']
        self.all_dice_per_class.append(dice_per_class)
        self.all_dice_scores.append(np.mean(dice_per_class))
        
        # Initial dice (pre-registration)
        if initial_dice is not None:
            self.all_initial_dice.append(initial_dice)
        if initial_dice_per_class is not None:
            self.all_initial_dice_per_class.append(initial_dice_per_class)
        
        # Compute volumes for volumetric correlation
        sample_seg = to_numpy(data['sample_seg'][0])  # (C, D, H, W)
        warped_seg = to_numpy(results['warped_seg'][0])
        
        gt_volumes = []
        pred_volumes = []
        for c in range(sample_seg.shape[0]):
            gt_vol = (sample_seg[c] > 0.5).sum()
            pred_vol = (warped_seg[c] > 0.5).sum()
            gt_volumes.append(gt_vol)
            pred_volumes.append(pred_vol)
        
        self.all_volumes_gt.append(gt_volumes)
        self.all_volumes_pred.append(pred_volumes)
        
        # NCC
        if 'ncc' in results:
            self.all_ncc_scores.append(results['ncc'])
        
        # Flow statistics
        flow = to_numpy(results['flow'])
        flow_mag = compute_flow_magnitude(flow)
        self.all_flow_stats.append({
            'mean': flow_mag.mean(),
            'max': flow_mag.max(),
            'std': flow_mag.std()
        })
        
        # Jacobian statistics
        jacobian = compute_jacobian_determinant(flow)
        self.all_jacobian_stats.append({
            'mean': jacobian.mean(),
            'min': jacobian.min(),
            'max': jacobian.max(),
            'folding_pct': (jacobian < 0).mean() * 100
        })
        
        # Smoothness
        if 'smoothness' in results:
            self.all_smoothness.append(results['smoothness'])
    
    def plot_dice_distribution(self, save=True):
        """Plot Dice score distribution across dataset."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        dice_array = np.array(self.all_dice_per_class)
        
        # Overall Dice distribution
        ax = axes[0, 0]
        ax.hist(self.all_dice_scores, bins=30, color='#0f3460', edgecolor='#e94560', alpha=0.8)
        ax.axvline(x=np.mean(self.all_dice_scores), color='lime', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(self.all_dice_scores):.4f}')
        ax.set_xlabel('Dice Score')
        ax.set_ylabel('Count')
        ax.set_title('Overall Dice Distribution', fontweight='bold')
        ax.legend()
        
        # Per-class box plot
        ax = axes[0, 1]
        bp = ax.boxplot(dice_array, labels=CLASS_NAMES, patch_artist=True)
        for patch, color in zip(bp['boxes'], CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Dice Score')
        ax.set_title('Per-Class Dice Distribution', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Per-class violin plot
        ax = axes[0, 2]
        parts = ax.violinplot(dice_array, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(CLASS_COLORS[i])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(CLASS_NAMES) + 1))
        ax.set_xticklabels(CLASS_NAMES, rotation=45)
        ax.set_ylabel('Dice Score')
        ax.set_title('Per-Class Dice Violin Plot', fontweight='bold')
        
        # Per-class mean comparison
        ax = axes[1, 0]
        means = dice_array.mean(axis=0)
        stds = dice_array.std(axis=0)
        x = range(len(CLASS_NAMES))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=CLASS_COLORS, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=45)
        ax.set_ylabel('Dice Score')
        ax.set_title('Per-Class Mean Dice (±std)', fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Dice correlation between classes
        ax = axes[1, 1]
        corr_matrix = np.corrcoef(dice_array.T)
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(CLASS_NAMES)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Class Dice Correlation', fontweight='bold')
        
        # Dice over samples
        ax = axes[1, 2]
        for c, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            ax.plot(self.sample_indices, dice_array[:, c], 'o-', label=name, 
                   color=color, alpha=0.7, markersize=4)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Dice Score')
        ax.set_title('Dice Score per Sample', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        
        fig.suptitle('Dice Score Analysis Across Dataset', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "aggregate_dice_analysis.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig
    
    def plot_deformation_statistics(self, save=True):
        """Plot deformation field statistics across dataset."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Flow magnitude statistics
        flow_means = [s['mean'] for s in self.all_flow_stats]
        flow_maxs = [s['max'] for s in self.all_flow_stats]
        
        ax = axes[0, 0]
        ax.plot(self.sample_indices, flow_means, 'o-', color='#e94560', label='Mean', alpha=0.8)
        ax.plot(self.sample_indices, flow_maxs, 's-', color='#533483', label='Max', alpha=0.8)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Flow Magnitude')
        ax.set_title('Flow Magnitude per Sample', fontweight='bold')
        ax.legend()
        
        # Flow magnitude distribution
        ax = axes[0, 1]
        ax.hist(flow_means, bins=30, color='#e94560', alpha=0.8, label='Mean')
        ax.hist(flow_maxs, bins=30, color='#533483', alpha=0.5, label='Max')
        ax.set_xlabel('Flow Magnitude')
        ax.set_ylabel('Count')
        ax.set_title('Flow Magnitude Distribution', fontweight='bold')
        ax.legend()
        
        # Jacobian statistics
        jac_means = [s['mean'] for s in self.all_jacobian_stats]
        jac_mins = [s['min'] for s in self.all_jacobian_stats]
        folding_pcts = [s['folding_pct'] for s in self.all_jacobian_stats]
        
        ax = axes[0, 2]
        ax.plot(self.sample_indices, jac_means, 'o-', color='#16213e', label='Mean', alpha=0.8)
        ax.plot(self.sample_indices, jac_mins, 's-', color='#e94560', label='Min', alpha=0.8)
        ax.axhline(y=0, color='red', linestyle='--', label='Folding threshold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Jacobian Determinant')
        ax.set_title('Jacobian Statistics per Sample', fontweight='bold')
        ax.legend()
        
        # Folding percentage
        ax = axes[1, 0]
        ax.bar(self.sample_indices, folding_pcts, color='#e94560', alpha=0.8)
        ax.axhline(y=np.mean(folding_pcts), color='lime', linestyle='--', 
                  label=f'Mean: {np.mean(folding_pcts):.2f}%')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Folding %')
        ax.set_title('Deformation Folding per Sample', fontweight='bold')
        ax.legend()
        
        # Dice vs Flow magnitude correlation
        ax = axes[1, 1]
        sc = ax.scatter(flow_means, self.all_dice_scores, c=folding_pcts, 
                       cmap='RdYlGn_r', s=50, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='Folding %')
        ax.set_xlabel('Mean Flow Magnitude')
        ax.set_ylabel('Mean Dice Score')
        ax.set_title('Dice vs Flow Magnitude', fontweight='bold')
        
        # Dice vs Jacobian min correlation
        ax = axes[1, 2]
        sc = ax.scatter(jac_mins, self.all_dice_scores, c=flow_means, 
                       cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='Flow Mean')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Min Jacobian Determinant')
        ax.set_ylabel('Mean Dice Score')
        ax.set_title('Dice vs Min Jacobian', fontweight='bold')
        
        fig.suptitle('Deformation Field Analysis Across Dataset', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "aggregate_deformation_analysis.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig
    
    def plot_best_worst_samples(self, all_data, all_results, n=3, save=True):
        """Show best and worst performing samples."""
        dice_array = np.array(self.all_dice_per_class)
        mean_dice = dice_array.mean(axis=1)
        
        best_indices = np.argsort(mean_dice)[-n:][::-1]
        worst_indices = np.argsort(mean_dice)[:n]
        
        fig = plt.figure(figsize=(24, 12))
        gs = GridSpec(2, n * 2, figure=fig, hspace=0.3, wspace=0.2)
        
        def plot_sample(ax_mri, ax_seg, data, results, title, dice):
            sample_mri = to_numpy(data['sample_mri'][0, 0])
            warped_mri = to_numpy(results['warped_mri'][0, 0])
            sample_seg = data['sample_seg'][0]
            warped_seg = results['warped_seg'][0]
            
            d = sample_mri.shape[0] // 2
            
            # MRI comparison
            diff = np.abs(warped_mri[d] - sample_mri[d])
            ax_mri.imshow(sample_mri[d], cmap='gray', alpha=0.5)
            ax_mri.imshow(diff, cmap='hot', alpha=0.5, vmax=0.3)
            ax_mri.set_title(f'{title}\nDice: {dice:.4f}', fontsize=10, fontweight='bold')
            ax_mri.axis('off')
            
            # Segmentation comparison
            ax_seg.imshow(sample_mri[d], cmap='gray', alpha=0.3)
            warped_labels = to_numpy(warped_seg).argmax(axis=0)[d]
            sample_labels = to_numpy(sample_seg).argmax(axis=0)[d]
            error = (warped_labels != sample_labels).astype(float)
            ax_seg.imshow(error, cmap='Reds', alpha=0.7)
            ax_seg.axis('off')
        
        # Best samples
        for i, idx in enumerate(best_indices):
            ax_mri = fig.add_subplot(gs[0, i * 2])
            ax_seg = fig.add_subplot(gs[0, i * 2 + 1])
            plot_sample(ax_mri, ax_seg, all_data[idx], all_results[idx], 
                       f'Best #{i+1} (Sample {self.sample_indices[idx]})', mean_dice[idx])
        
        # Worst samples
        for i, idx in enumerate(worst_indices):
            ax_mri = fig.add_subplot(gs[1, i * 2])
            ax_seg = fig.add_subplot(gs[1, i * 2 + 1])
            plot_sample(ax_mri, ax_seg, all_data[idx], all_results[idx],
                       f'Worst #{i+1} (Sample {self.sample_indices[idx]})', mean_dice[idx])
        
        fig.suptitle('Best vs Worst Performing Samples\n(Left: MRI Error, Right: Seg Error)',
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            fig.savefig(self.output_dir / "aggregate_best_worst_samples.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig
    
    def plot_difficulty_vs_performance(self, save=True):
        """
        Difficulty vs Performance Scatter Plot.
        
        X-Axis: Initial Dice Score (Pre-registration overlap) - represents "difficulty"
        Y-Axis: Final Dice Score (Post-registration)
        
        Interpretation:
        - Flat line at top: Model is robust to large anatomical variations
        - Steep positive correlation: Model only works on easy cases
        """
        if len(self.all_initial_dice) == 0:
            print("Warning: No initial dice scores available for difficulty analysis")
            return None
        
        from scipy import stats
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        initial_dice = np.array(self.all_initial_dice)
        final_dice = np.array(self.all_dice_scores)
        improvement = final_dice - initial_dice
        
        # Main scatter plot
        ax = axes[0]
        sc = ax.scatter(initial_dice, final_dice, c=improvement, cmap='RdYlGn', 
                       s=80, alpha=0.8, edgecolors='white', linewidths=0.5)
        
        # Add identity line (no improvement)
        lims = [min(initial_dice.min(), final_dice.min()) - 0.05, 
                max(initial_dice.max(), final_dice.max()) + 0.05]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='No improvement (y=x)')
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(initial_dice, final_dice)
        x_line = np.linspace(initial_dice.min(), initial_dice.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
               label=f'Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r_value**2:.3f}')
        
        plt.colorbar(sc, ax=ax, label='Improvement (Final - Initial)')
        ax.set_xlabel('Initial Dice (Pre-Registration)', fontsize=11)
        ax.set_ylabel('Final Dice (Post-Registration)', fontsize=11)
        ax.set_title('Difficulty vs Performance\n(Low initial = Hard case)', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.3)
        
        # Improvement histogram
        ax = axes[1]
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in improvement]
        ax.bar(range(len(improvement)), improvement, color=colors, alpha=0.8)
        ax.axhline(y=0, color='white', linestyle='-', linewidth=1)
        ax.axhline(y=np.mean(improvement), color='cyan', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(improvement):.4f}')
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Dice Improvement', fontsize=11)
        ax.set_title('Per-Sample Improvement\n(Green=Improved, Red=Degraded)', fontsize=12, fontweight='bold')
        ax.legend()
        
        # Summary statistics
        ax = axes[2]
        ax.axis('off')
        
        # Compute robustness metrics
        hard_cases = initial_dice < np.percentile(initial_dice, 25)
        easy_cases = initial_dice > np.percentile(initial_dice, 75)
        
        hard_final = final_dice[hard_cases].mean() if hard_cases.any() else 0
        easy_final = final_dice[easy_cases].mean() if easy_cases.any() else 0
        
        stats_text = f"""
Difficulty vs Performance Analysis
──────────────────────────────────

Overall Statistics:
  Initial Dice (mean): {initial_dice.mean():.4f} ± {initial_dice.std():.4f}
  Final Dice (mean):   {final_dice.mean():.4f} ± {final_dice.std():.4f}
  Improvement (mean):  {improvement.mean():.4f} ± {improvement.std():.4f}

Regression Analysis:
  Slope: {slope:.4f}
  R²: {r_value**2:.4f}
  p-value: {p_value:.2e}

Robustness Check:
  Hard cases (bottom 25%):
    Initial: {initial_dice[hard_cases].mean():.4f}
    Final:   {hard_final:.4f}
  
  Easy cases (top 25%):
    Initial: {initial_dice[easy_cases].mean():.4f}
    Final:   {easy_final:.4f}

Interpretation:
  {'✓ ROBUST: Model works well on hard cases' if slope < 0.5 else '⚠ CAUTION: Performance depends on difficulty'}
  {'✓ All samples improved' if (improvement > 0).all() else f'⚠ {(improvement < 0).sum()} samples degraded'}
"""
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
        
        fig.suptitle('Difficulty vs Performance Analysis', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / "aggregate_difficulty_vs_performance.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig
    
    def plot_volumetric_correlation(self, save=True):
        """
        Volumetric Correlation Plot - Clinical Validity Check.
        
        Shows whether registration preserves structure volumes.
        X-Axis: Ground Truth Volume (from Sample Seg)
        Y-Axis: Predicted Volume (from Warped Seg)
        
        High R² indicates the model doesn't systematically shrink/expand structures.
        """
        if len(self.all_volumes_gt) == 0:
            print("Warning: No volume data available")
            return None
        
        from scipy import stats
        
        gt_volumes = np.array(self.all_volumes_gt)  # (N_samples, N_classes)
        pred_volumes = np.array(self.all_volumes_pred)
        
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # Per-class scatter plots (skip background)
        for c in range(1, min(5, gt_volumes.shape[1])):  # Skip background (c=0)
            row = (c - 1) // 3
            col = (c - 1) % 3
            ax = fig.add_subplot(gs[row, col])
            
            gt_c = gt_volumes[:, c]
            pred_c = pred_volumes[:, c]
            
            # Scatter plot
            ax.scatter(gt_c, pred_c, c=CLASS_COLORS[c], s=60, alpha=0.7, 
                      edgecolors='white', linewidths=0.5)
            
            # Identity line
            max_vol = max(gt_c.max(), pred_c.max()) * 1.1
            ax.plot([0, max_vol], [0, max_vol], 'w--', alpha=0.5, label='y=x (perfect)')
            
            # Regression
            if gt_c.std() > 0:
                slope, intercept, r_value, p_value, _ = stats.linregress(gt_c, pred_c)
                x_line = np.linspace(gt_c.min(), gt_c.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color='#e94560', linewidth=2,
                       label=f'Fit: R²={r_value**2:.3f}')
                
                # Compute bias
                bias = (pred_c - gt_c).mean()
                bias_pct = (bias / gt_c.mean()) * 100 if gt_c.mean() > 0 else 0
                r2_val = r_value**2
            else:
                bias_pct = 0
                r2_val = 0
            
            ax.set_xlabel('GT Volume (voxels)', fontsize=10)
            ax.set_ylabel('Predicted Volume (voxels)', fontsize=10)
            ax.set_title(f'{CLASS_NAMES[c]}\nR²={r2_val:.3f}, Bias={bias_pct:+.1f}%', 
                        fontsize=11, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Summary panel
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')
        
        summary_text = "Volumetric Correlation Summary\n" + "─" * 30 + "\n\n"
        summary_text += f"{'Structure':<15} {'R²':>8} {'Bias %':>10}\n"
        summary_text += "─" * 30 + "\n"
        
        all_r2 = []
        for c in range(1, gt_volumes.shape[1]):
            gt_c = gt_volumes[:, c]
            pred_c = pred_volumes[:, c]
            if gt_c.std() > 0:
                _, _, r_value, _, _ = stats.linregress(gt_c, pred_c)
                r2 = r_value ** 2
                bias_pct = ((pred_c - gt_c).mean() / gt_c.mean()) * 100 if gt_c.mean() > 0 else 0
                all_r2.append(r2)
                summary_text += f"{CLASS_NAMES[c]:<15} {r2:>8.3f} {bias_pct:>+10.1f}%\n"
        
        summary_text += "─" * 30 + "\n"
        summary_text += f"{'Mean R²':<15} {np.mean(all_r2):>8.3f}\n\n"
        
        summary_text += "Interpretation:\n"
        if np.mean(all_r2) > 0.9:
            summary_text += "✓ EXCELLENT: Volumes well preserved\n"
        elif np.mean(all_r2) > 0.7:
            summary_text += "✓ GOOD: Volumes reasonably preserved\n"
        else:
            summary_text += "⚠ WARNING: Volume preservation issues\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
        
        fig.suptitle('Volumetric Correlation - Clinical Validity Check\n(Does registration preserve structure volumes?)',
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            fig.savefig(self.output_dir / "aggregate_volumetric_correlation.png",
                       dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
        return fig
    
    def generate_report(self, save=True):
        """Generate text summary report."""
        dice_array = np.array(self.all_dice_per_class)
        
        report = f"""
{'='*70}
MRI REGISTRATION MODEL PERFORMANCE REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of samples: {len(self.sample_indices)}

{'='*70}
DICE SCORE SUMMARY
{'='*70}
Overall Mean Dice: {np.mean(self.all_dice_scores):.4f} ± {np.std(self.all_dice_scores):.4f}
Best Sample Dice: {np.max(self.all_dice_scores):.4f} (Sample {self.sample_indices[np.argmax(self.all_dice_scores)]})
Worst Sample Dice: {np.min(self.all_dice_scores):.4f} (Sample {self.sample_indices[np.argmin(self.all_dice_scores)]})

Per-Class Performance:
"""
        for c, name in enumerate(CLASS_NAMES):
            class_dice = dice_array[:, c]
            report += f"  {name:15s}: {class_dice.mean():.4f} ± {class_dice.std():.4f} "
            report += f"(range: {class_dice.min():.4f} - {class_dice.max():.4f})\n"
        
        report += f"""
{'='*70}
DEFORMATION FIELD STATISTICS
{'='*70}
Flow Magnitude:
  Mean: {np.mean([s['mean'] for s in self.all_flow_stats]):.4f}
  Max (avg): {np.mean([s['max'] for s in self.all_flow_stats]):.4f}

Jacobian Determinant:
  Mean: {np.mean([s['mean'] for s in self.all_jacobian_stats]):.4f}
  Min (avg): {np.mean([s['min'] for s in self.all_jacobian_stats]):.4f}
  Folding % (avg): {np.mean([s['folding_pct'] for s in self.all_jacobian_stats]):.2f}%

{'='*70}
FAILURE ANALYSIS
{'='*70}
"""
        # Identify failure patterns
        worst_class = np.argmin(dice_array.mean(axis=0))
        best_class = np.argmax(dice_array.mean(axis=0))
        high_folding = sum(1 for s in self.all_jacobian_stats if s['folding_pct'] > 1)
        
        report += f"""
Weakest Class: {CLASS_NAMES[worst_class]} (Dice: {dice_array[:, worst_class].mean():.4f})
Strongest Class: {CLASS_NAMES[best_class]} (Dice: {dice_array[:, best_class].mean():.4f})
Samples with >1% folding: {high_folding}/{len(self.sample_indices)}

{'='*70}
"""
        
        if save:
            with open(self.output_dir / "performance_report.txt", 'w') as f:
                f.write(report)
        
        return report


# =============================================================================
# Main Visualization Pipeline
# =============================================================================

def run_inference(model, stn, data, device):
    """Run model inference on a single sample."""
    model.eval()
    
    with torch.no_grad():
        template_mri = data['template_mri'].unsqueeze(0).to(device)
        template_seg = data['template_seg'].unsqueeze(0).to(device)
        sample_mri = data['sample_mri'].unsqueeze(0).to(device)
        sample_seg = data['sample_seg'].unsqueeze(0).to(device)
        
        # Forward pass
        final_flow, intermediate_flows, lambda_maps, attention_maps = model(
            template_mri, template_seg, sample_mri
        )
        
        # Warp template
        warped_mri = stn(template_mri, final_flow)
        warped_seg = stn(template_seg, final_flow)
        
        # Compute metrics
        dice_per_class, mean_dice = compute_dice_score(warped_seg, sample_seg)
        
        # Compute additional metrics
        flow_np = to_numpy(final_flow)
        flow_tensor = final_flow.cpu()
        
        # Compute self-intersection (folding) loss
        self_intersection_stats = compute_self_intersection_loss(flow_tensor)
        
        results = {
            'flow': final_flow.cpu(),
            'warped_mri': warped_mri.cpu(),
            'warped_seg': warped_seg.cpu(),
            'intermediate_flows': [f.cpu() for f in intermediate_flows],
            'lambda_maps': [l.cpu() for l in lambda_maps],
            'attention_maps': [a.cpu() for a in attention_maps],
            'dice_per_class': dice_per_class,
            'mean_dice': mean_dice,
            'ncc': ncc_loss(warped_mri, sample_mri).item(),
            'smoothness': smoothness_loss(flow_tensor).item(),
            'bending': bending_energy_loss(flow_tensor).item(),
            # Self-intersection (folding) metrics
            'self_intersection_loss': self_intersection_stats['loss'],
            'folding_percentage': self_intersection_stats['folding_percentage'],
            'num_folding_voxels': self_intersection_stats['num_folding_voxels'],
            'min_jacobian': self_intersection_stats['min_jacobian'],
            'max_jacobian': self_intersection_stats['max_jacobian'],
            'mean_jacobian': self_intersection_stats['mean_jacobian'],
        }
        
    return results


def main():
    parser = argparse.ArgumentParser(description='MRI Registration Visualization')
    parser.add_argument('--which_timestamp', type=str, required=True,
                       help='Timestamp of the training run to visualize')
    parser.add_argument('--data_split', type=str, default='val',
                       choices=['train', 'val'], help='Data split to visualize')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of samples to visualize (-1 for all)')
    parser.add_argument('--detailed_samples', type=int, default=10,
                       help='Number of samples for detailed visualization')
    parser.add_argument('--device', type=str, default='cuda:4',
                       help='Device to run inference on')
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128],
                       help='Target volume size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    # Build paths from timestamp
    base_dir = f"/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_mri_acm/{args.which_timestamp}"
    checkpoint_path = f"{base_dir}/checkpoints/best_model.pth"
    output_base = f"{base_dir}/viz_output"
    
    # Configuration
    target_size = tuple(args.target_size)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Paths (adjust as needed)
    if args.data_split == 'val':
        data_list = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/val.txt"
    else:
        data_list = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/train.txt"
    
    template_mri_path = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/brain.npy"
    template_seg_path = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy"
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_base) / f"viz_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print("MRI Registration Visualization")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Data split: {args.data_split}")
    
    # Load model
    print("\nLoading model...")
    model = MRIRegistrationNet(seg_channels=5).to(device)
    stn = SpatialTransformer(size=target_size, device=device).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best Dice: {checkpoint.get('best_dice', 'unknown')}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MRIDataset(
        data_list, template_mri_path, template_seg_path,
        target_size=target_size
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine number of samples and randomly select indices
    total_samples = len(dataset)
    num_samples = total_samples if args.num_samples == -1 else min(args.num_samples, total_samples)
    detailed_samples = min(args.detailed_samples, num_samples)
    
    # Randomly sample indices
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    sample_indices = sorted(all_indices[:num_samples])  # Sort for consistent ordering in plots
    
    # Select detailed sample indices from the sampled indices
    detailed_indices = set(random.sample(sample_indices, detailed_samples))
    
    print(f"\nRandomly selected {num_samples} samples from {total_samples} total (seed={args.seed})")
    print(f"Detailed visualization for {detailed_samples} randomly selected samples")
    print(f"Sample indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
    print(f"Detailed indices: {sorted(detailed_indices)}")
    
    # Initialize visualizers
    sample_viz = SampleVisualizer(output_dir / "samples")
    aggregate_viz = AggregateAnalyzer(output_dir / "aggregate")
    
    # Storage for best/worst analysis
    all_data = []
    all_results = []
    
    # Run visualization
    for idx in tqdm(sample_indices, desc="Processing samples"):
        data = dataset[idx]
        
        # Convert to batch format for model (keeping original for viz)
        data_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                     for k, v in data.items()}
        
        # Run inference
        results = run_inference(model, stn, data, device)
        
        # Compute initial dice (pre-registration) for difficulty analysis
        template_seg = to_numpy(data_batch['template_seg'][0])
        sample_seg = to_numpy(data_batch['sample_seg'][0])
        initial_dice_per_class = []
        for c in range(template_seg.shape[0]):
            intersection = 2 * (template_seg[c] * sample_seg[c]).sum()
            union = template_seg[c].sum() + sample_seg[c].sum()
            initial_dice_per_class.append((intersection + 1e-5) / (union + 1e-5))
        initial_dice = np.mean(initial_dice_per_class)
        
        # Store for aggregate analysis
        all_data.append(data_batch)
        all_results.append(results)
        
        # Add to aggregate analyzer with initial dice
        aggregate_viz.add_sample(idx, results, data_batch, 
                                initial_dice=initial_dice, 
                                initial_dice_per_class=initial_dice_per_class)
        
        # Detailed visualization for randomly selected samples
        if idx in detailed_indices:
            sample_viz.plot_registration_overview(data_batch, results, idx)
            sample_viz.plot_deformation_analysis(data_batch, results, idx)
            sample_viz.plot_grid_warp(data_batch, results, idx)
            _, _, _ = sample_viz.plot_template_vs_sample(data_batch, results, idx)
    
    # Generate aggregate visualizations
    print("\nGenerating aggregate analysis...")
    aggregate_viz.plot_dice_distribution()
    aggregate_viz.plot_deformation_statistics()
    aggregate_viz.plot_best_worst_samples(all_data, all_results, n=3)
    aggregate_viz.plot_difficulty_vs_performance()
    aggregate_viz.plot_volumetric_correlation()
    
    # Generate report
    report = aggregate_viz.generate_report()
    print(report)
    
    # Save configuration
    config = {
        'which_timestamp': args.which_timestamp,
        'checkpoint': checkpoint_path,
        'data_split': args.data_split,
        'num_samples': num_samples,
        'detailed_samples': detailed_samples,
        'sample_indices': sample_indices,
        'detailed_indices': sorted(list(detailed_indices)),
        'target_size': list(target_size),
        'device': str(device),
        'timestamp': timestamp,
        'seed': args.seed
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Visualization complete!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

