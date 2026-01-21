"""
GradCAM-style Analysis for MRI Registration Network

Provides visualization of:
1. Attention maps from Segmentation Attention Modules (SAM)
2. GradCAM activations for encoder/decoder layers
3. Lambda maps (adaptive regularization)
4. Flow field visualizations
5. Feature importance analysis

Usage:
    python gradcam_analysis.py --checkpoint <path_to_checkpoint>
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Local imports
from model_mri import MRIRegistrationNet, SpatialTransformer
from get_data_mri import MRIDataset
from losses_mri import compute_dice_score


# =============================================================================
# GradCAM Hook Manager
# =============================================================================

class GradCAMHook:
    """Hook to capture activations and gradients for GradCAM."""
    
    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self.handles = []
    
    def register_forward_hook(self, module, name):
        """Register forward hook to capture activations."""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        handle = module.register_forward_hook(hook)
        self.handles.append(handle)
    
    def register_backward_hook(self, module, name):
        """Register backward hook to capture gradients."""
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        handle = module.register_full_backward_hook(hook)
        self.handles.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def compute_gradcam(self, activation_name, gradient_name):
        """
        Compute GradCAM from activations and gradients.
        
        Returns:
            cam: (B, D, H, W) - GradCAM heatmap
        """
        if activation_name not in self.activations or gradient_name not in self.gradients:
            return None
        
        activations = self.activations[activation_name]  # (B, C, D, H, W)
        gradients = self.gradients[gradient_name]        # (B, C, D, H, W)
        
        # Global average pooling of gradients (channel importance weights)
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        # ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze(1)  # (B, D, H, W)
        for b in range(cam.shape[0]):
            cam_min = cam[b].min()
            cam_max = cam[b].max()
            if cam_max - cam_min > 1e-8:
                cam[b] = (cam[b] - cam_min) / (cam_max - cam_min)
        
        return cam


# =============================================================================
# Model Analyzer
# =============================================================================

class MRIRegistrationAnalyzer:
    """Comprehensive analyzer for MRI registration network."""
    
    def __init__(self, model, stn, device='cuda:6'):
        self.model = model
        self.stn = stn
        self.device = device
        self.hook_manager = GradCAMHook()
        
        # Register hooks on key layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on important layers."""
        # Encoder layers
        for i, enc in enumerate([self.model.encoder.enc1, 
                                  self.model.encoder.enc2,
                                  self.model.encoder.enc3,
                                  self.model.encoder.enc4]):
            self.hook_manager.register_forward_hook(enc, f'enc{i+1}')
            self.hook_manager.register_backward_hook(enc, f'enc{i+1}_grad')
        
        # Decoder layers
        for i, dec in enumerate([self.model.decoder.dec1,
                                  self.model.decoder.dec2,
                                  self.model.decoder.dec3,
                                  self.model.decoder.dec4]):
            self.hook_manager.register_forward_hook(dec, f'dec{i+1}')
            self.hook_manager.register_backward_hook(dec, f'dec{i+1}_grad')
        
        # Bottleneck
        self.hook_manager.register_forward_hook(self.model.encoder.bottleneck, 'bottleneck')
        self.hook_manager.register_backward_hook(self.model.encoder.bottleneck, 'bottleneck_grad')
    
    def analyze_sample(self, template_mri, template_seg, sample_mri, sample_seg):
        """
        Perform comprehensive analysis on a single sample.
        
        Returns:
            results: dict containing all analysis results
        """
        self.model.eval()
        
        # Forward pass
        with torch.set_grad_enabled(True):
            template_mri.requires_grad_(True)
            
            final_flow, intermediate_flows, lambda_maps, attention_maps = self.model(
                template_mri, template_seg, sample_mri
            )
            
            # Warp template
            warped_mri = self.stn(template_mri, final_flow)
            warped_seg = self.stn(template_seg, final_flow)
            
            # Compute loss for gradient computation
            loss = F.mse_loss(warped_mri, sample_mri)
            
            # Backward pass to get gradients
            self.model.zero_grad()
            loss.backward()
        
        # Compute GradCAM for each layer
        gradcams = {}
        for i in range(1, 5):
            cam = self.hook_manager.compute_gradcam(f'enc{i}', f'enc{i}_grad')
            if cam is not None:
                gradcams[f'encoder_scale{i}'] = cam
            
            cam = self.hook_manager.compute_gradcam(f'dec{i}', f'dec{i}_grad')
            if cam is not None:
                gradcams[f'decoder_scale{i}'] = cam
        
        # Bottleneck GradCAM
        cam = self.hook_manager.compute_gradcam('bottleneck', 'bottleneck_grad')
        if cam is not None:
            gradcams['bottleneck'] = cam
        
        # Compute Dice scores
        dice_per_class, mean_dice = compute_dice_score(warped_seg, sample_seg, num_classes=5)
        
        # Collect results
        results = {
            'template_mri': template_mri.detach().cpu(),
            'template_seg': template_seg.detach().cpu(),
            'sample_mri': sample_mri.detach().cpu(),
            'sample_seg': sample_seg.detach().cpu(),
            'warped_mri': warped_mri.detach().cpu(),
            'warped_seg': warped_seg.detach().cpu(),
            'final_flow': final_flow.detach().cpu(),
            'intermediate_flows': [f.detach().cpu() for f in intermediate_flows],
            'lambda_maps': [l.detach().cpu() for l in lambda_maps],
            'attention_maps': [a.detach().cpu() for a in attention_maps],
            'gradcams': {k: v.detach().cpu() for k, v in gradcams.items()},
            'dice_per_class': dice_per_class,
            'mean_dice': mean_dice,
            'loss': loss.item()
        }
        
        return results
    
    def cleanup(self):
        """Remove all hooks."""
        self.hook_manager.remove_hooks()


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_slice(volume, slice_idx=None, axis=2, cmap='gray', vmin=None, vmax=None, ax=None):
    """Plot a 2D slice from a 3D volume."""
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2
    
    if axis == 0:
        slice_2d = volume[slice_idx, :, :]
    elif axis == 1:
        slice_2d = volume[:, slice_idx, :]
    else:  # axis == 2
        slice_2d = volume[:, :, slice_idx]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    im = ax.imshow(slice_2d.T, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    return im


def plot_overlay(background, overlay, slice_idx=None, axis=2, alpha=0.5, ax=None):
    """Plot overlay heatmap on background image."""
    if slice_idx is None:
        slice_idx = background.shape[axis] // 2
    
    if axis == 0:
        bg_slice = background[slice_idx, :, :]
        ov_slice = overlay[slice_idx, :, :]
    elif axis == 1:
        bg_slice = background[:, slice_idx, :]
        ov_slice = overlay[:, slice_idx, :]
    else:
        bg_slice = background[:, :, slice_idx]
        ov_slice = overlay[:, :, slice_idx]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Plot background
    ax.imshow(bg_slice.T, cmap='gray', origin='lower')
    
    # Plot overlay with transparency
    ax.imshow(ov_slice.T, cmap='jet', origin='lower', alpha=alpha, vmin=0, vmax=1)
    ax.axis('off')


def visualize_attention_maps(results, output_dir, slice_idx=None):
    """Visualize attention maps from SAM modules."""
    attention_maps = results['attention_maps']
    template_mri = results['template_mri'][0, 0].numpy()
    
    if slice_idx is None:
        slice_idx = template_mri.shape[2] // 2
    
    num_scales = len(attention_maps)
    fig, axes = plt.subplots(2, num_scales, figsize=(4*num_scales, 8))
    
    if num_scales == 1:
        axes = axes.reshape(-1, 1)
    
    for i, attn_map in enumerate(attention_maps):
        attn = attn_map[0].mean(dim=0).numpy()  # Average over channels
        
        # Resize to match template size
        attn_resized = F.interpolate(
            torch.tensor(attn).unsqueeze(0).unsqueeze(0),
            size=template_mri.shape,
            mode='trilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # Plot attention map alone
        plot_slice(attn_resized, slice_idx=slice_idx, axis=2, cmap='jet', ax=axes[0, i])
        axes[0, i].set_title(f'SAM Scale {i+1}')
        
        # Plot overlay on MRI
        plot_overlay(template_mri, attn_resized, slice_idx=slice_idx, axis=2, 
                    alpha=0.5, ax=axes[1, i])
        axes[1, i].set_title(f'SAM Scale {i+1} Overlay')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_gradcams(results, output_dir, slice_idx=None):
    """Visualize GradCAM heatmaps."""
    gradcams = results['gradcams']
    template_mri = results['template_mri'][0, 0].numpy()
    
    if slice_idx is None:
        slice_idx = template_mri.shape[2] // 2
    
    # Encoder GradCAMs
    encoder_cams = {k: v for k, v in gradcams.items() if 'encoder' in k}
    if encoder_cams:
        fig, axes = plt.subplots(2, len(encoder_cams), figsize=(4*len(encoder_cams), 8))
        if len(encoder_cams) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (name, cam) in enumerate(sorted(encoder_cams.items())):
            cam_np = cam[0].numpy()
            
            # Resize to match template
            cam_resized = F.interpolate(
                torch.tensor(cam_np).unsqueeze(0).unsqueeze(0),
                size=template_mri.shape,
                mode='trilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Plot CAM alone
            plot_slice(cam_resized, slice_idx=slice_idx, axis=2, cmap='jet', ax=axes[0, i])
            axes[0, i].set_title(name.replace('_', ' ').title())
            
            # Plot overlay
            plot_overlay(template_mri, cam_resized, slice_idx=slice_idx, axis=2,
                        alpha=0.5, ax=axes[1, i])
            axes[1, i].set_title(f'{name.replace("_", " ").title()} Overlay')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradcam_encoder.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Decoder GradCAMs
    decoder_cams = {k: v for k, v in gradcams.items() if 'decoder' in k}
    if decoder_cams:
        fig, axes = plt.subplots(2, len(decoder_cams), figsize=(4*len(decoder_cams), 8))
        if len(decoder_cams) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (name, cam) in enumerate(sorted(decoder_cams.items())):
            cam_np = cam[0].numpy()
            
            # Resize to match template
            cam_resized = F.interpolate(
                torch.tensor(cam_np).unsqueeze(0).unsqueeze(0),
                size=template_mri.shape,
                mode='trilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # Plot CAM alone
            plot_slice(cam_resized, slice_idx=slice_idx, axis=2, cmap='jet', ax=axes[0, i])
            axes[0, i].set_title(name.replace('_', ' ').title())
            
            # Plot overlay
            plot_overlay(template_mri, cam_resized, slice_idx=slice_idx, axis=2,
                        alpha=0.5, ax=axes[1, i])
            axes[1, i].set_title(f'{name.replace("_", " ").title()} Overlay')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradcam_decoder.png'), dpi=150, bbox_inches='tight')
        plt.close()


def visualize_lambda_maps(results, output_dir, slice_idx=None):
    """Visualize lambda maps (adaptive regularization)."""
    lambda_maps = results['lambda_maps']
    template_mri = results['template_mri'][0, 0].numpy()
    
    if slice_idx is None:
        slice_idx = template_mri.shape[2] // 2
    
    num_scales = len(lambda_maps)
    fig, axes = plt.subplots(2, num_scales, figsize=(4*num_scales, 8))
    
    if num_scales == 1:
        axes = axes.reshape(-1, 1)
    
    for i, lambda_map in enumerate(lambda_maps):
        lmap = lambda_map[0, 0].numpy()
        
        # Resize to match template
        lmap_resized = F.interpolate(
            torch.tensor(lmap).unsqueeze(0).unsqueeze(0),
            size=template_mri.shape,
            mode='trilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # Plot lambda map
        plot_slice(lmap_resized, slice_idx=slice_idx, axis=2, cmap='viridis', 
                  vmin=0, vmax=1, ax=axes[0, i])
        axes[0, i].set_title(f'Lambda Scale {i+1}')
        
        # Plot overlay
        plot_overlay(template_mri, lmap_resized, slice_idx=slice_idx, axis=2,
                    alpha=0.5, ax=axes[1, i])
        axes[1, i].set_title(f'Lambda Scale {i+1} Overlay')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lambda_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_flow_field(results, output_dir, slice_idx=None):
    """Visualize deformation flow field."""
    final_flow = results['final_flow'][0].numpy()  # (3, D, H, W)
    template_mri = results['template_mri'][0, 0].numpy()
    
    if slice_idx is None:
        slice_idx = template_mri.shape[2] // 2
    
    # Compute flow magnitude for full volume
    flow_mag_3d = np.sqrt(np.sum(final_flow**2, axis=0))
    
    # Extract slice
    flow_slice = final_flow[:, :, :, slice_idx]  # (3, D, H)
    mri_slice = template_mri[:, :, slice_idx]
    
    # Flow magnitude for the slice
    flow_mag = np.sqrt(np.sum(flow_slice**2, axis=0))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Flow components
    plot_slice(template_mri, slice_idx=slice_idx, axis=2, cmap='gray', ax=axes[0, 0])
    axes[0, 0].set_title('Template MRI')
    
    im1 = axes[0, 1].imshow(flow_slice[0].T, cmap='RdBu_r', origin='lower')
    axes[0, 1].set_title('Flow X')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(flow_slice[1].T, cmap='RdBu_r', origin='lower')
    axes[0, 2].set_title('Flow Y')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[1, 0].imshow(flow_slice[2].T, cmap='RdBu_r', origin='lower')
    axes[1, 0].set_title('Flow Z')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(flow_mag.T, cmap='hot', origin='lower')
    axes[1, 1].set_title('Flow Magnitude')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Flow overlay (using 3D volume)
    plot_overlay(template_mri, flow_mag_3d, slice_idx=slice_idx, axis=2, alpha=0.5, ax=axes[1, 2])
    axes[1, 2].set_title('Flow Magnitude Overlay')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flow_field.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_registration_results(results, output_dir, slice_idx=None):
    """Visualize registration input/output."""
    template_mri = results['template_mri'][0, 0].numpy()
    sample_mri = results['sample_mri'][0, 0].numpy()
    warped_mri = results['warped_mri'][0, 0].numpy()
    
    template_seg = results['template_seg'][0].argmax(dim=0).numpy()
    sample_seg = results['sample_seg'][0].argmax(dim=0).numpy()
    warped_seg = results['warped_seg'][0].argmax(dim=0).numpy()
    
    if slice_idx is None:
        slice_idx = template_mri.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # MRI
    plot_slice(template_mri, slice_idx=slice_idx, axis=2, cmap='gray', ax=axes[0, 0])
    axes[0, 0].set_title('Template MRI')
    
    plot_slice(sample_mri, slice_idx=slice_idx, axis=2, cmap='gray', ax=axes[0, 1])
    axes[0, 1].set_title('Sample MRI')
    
    plot_slice(warped_mri, slice_idx=slice_idx, axis=2, cmap='gray', ax=axes[0, 2])
    axes[0, 2].set_title('Warped MRI')
    
    # Segmentation
    plot_slice(template_seg, slice_idx=slice_idx, axis=2, cmap='tab10', 
              vmin=0, vmax=4, ax=axes[1, 0])
    axes[1, 0].set_title('Template Seg')
    
    plot_slice(sample_seg, slice_idx=slice_idx, axis=2, cmap='tab10',
              vmin=0, vmax=4, ax=axes[1, 1])
    axes[1, 1].set_title('Sample Seg')
    
    plot_slice(warped_seg, slice_idx=slice_idx, axis=2, cmap='tab10',
              vmin=0, vmax=4, ax=axes[1, 2])
    axes[1, 2].set_title(f'Warped Seg (Dice: {results["mean_dice"]:.4f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'registration_results.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_report(results, output_dir):
    """Create a summary report with key metrics."""
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MRI Registration Network - GradCAM Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Registration Quality:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Dice Score: {results['mean_dice']:.4f}\n")
        f.write(f"Registration Loss: {results['loss']:.6f}\n\n")
        
        f.write("Per-Class Dice Scores:\n")
        f.write("-" * 70 + "\n")
        for i, dice in enumerate(results['dice_per_class']):
            f.write(f"Class {i}: {dice:.4f}\n")
        f.write("\n")
        
        f.write("Flow Statistics:\n")
        f.write("-" * 70 + "\n")
        flow = results['final_flow'][0].numpy()
        flow_mag = np.sqrt(np.sum(flow**2, axis=0))
        f.write(f"Flow magnitude - Mean: {flow_mag.mean():.4f}, "
                f"Max: {flow_mag.max():.4f}, Std: {flow_mag.std():.4f}\n")
        f.write(f"Flow X - Mean: {flow[0].mean():.4f}, Range: [{flow[0].min():.4f}, {flow[0].max():.4f}]\n")
        f.write(f"Flow Y - Mean: {flow[1].mean():.4f}, Range: [{flow[1].min():.4f}, {flow[1].max():.4f}]\n")
        f.write(f"Flow Z - Mean: {flow[2].mean():.4f}, Range: [{flow[2].min():.4f}, {flow[2].max():.4f}]\n\n")
        
        f.write("Attention Map Statistics:\n")
        f.write("-" * 70 + "\n")
        for i, attn_map in enumerate(results['attention_maps']):
            attn = attn_map[0].numpy()
            f.write(f"Scale {i+1} - Mean: {attn.mean():.4f}, "
                   f"Range: [{attn.min():.4f}, {attn.max():.4f}]\n")
        f.write("\n")
        
        f.write("Lambda Map Statistics:\n")
        f.write("-" * 70 + "\n")
        for i, lambda_map in enumerate(results['lambda_maps']):
            lmap = lambda_map[0].numpy()
            f.write(f"Scale {i+1} - Mean: {lmap.mean():.4f}, "
                   f"Range: [{lmap.min():.4f}, {lmap.max():.4f}]\n")
        f.write("\n")
        
        f.write("GradCAM Layers Analyzed:\n")
        f.write("-" * 70 + "\n")
        for name in sorted(results['gradcams'].keys()):
            f.write(f"- {name}\n")
        f.write("\n")
    
    print(f"Summary report saved to: {report_path}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GradCAM Analysis for MRI Registration')
    parser.add_argument('--checkpoint', type=str, 
                       default='/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/training_mri_acm/20251129_053413/checkpoints/checkpoint_epoch_060.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--val_txt', type=str,
                       default='/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/val.txt',
                       help='Path to validation data list')
    parser.add_argument('--template_mri', type=str,
                       default='/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/brain.npy',
                       help='Path to template MRI')
    parser.add_argument('--template_seg', type=str,
                       default='/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy',
                       help='Path to template segmentation')
    parser.add_argument('--output_dir', type=str,
                       default='/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/training_mri_acm/20251129_053413/gradcam_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda:6',
                       help='Device to use')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to analyze')
    parser.add_argument('--slice_idx', type=int, default=None,
                       help='Slice index to visualize (default: middle slice)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MRI Registration Network - GradCAM Analysis")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = MRIRegistrationNet(seg_channels=5).to(device)
    stn = SpatialTransformer(size=(128, 128, 128), device=device).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation Dice: {checkpoint.get('val_dice', 'N/A')}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = MRIDataset(
        args.val_txt, args.template_mri, args.template_seg,
        target_size=(128, 128, 128)
    )
    print(f"Dataset size: {len(dataset)}")
    print()
    
    # Create analyzer
    analyzer = MRIRegistrationAnalyzer(model, stn, device)
    
    # Analyze samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"Analyzing {num_samples} samples...")
    print()
    
    for idx in range(num_samples):
        print(f"Processing sample {idx+1}/{num_samples}...")
        
        # Load data
        batch = dataset[idx]
        template_mri = batch['template_mri'].unsqueeze(0).to(device)
        template_seg = batch['template_seg'].unsqueeze(0).to(device)
        sample_mri = batch['sample_mri'].unsqueeze(0).to(device)
        sample_seg = batch['sample_seg'].unsqueeze(0).to(device)
        
        # Analyze
        results = analyzer.analyze_sample(template_mri, template_seg, sample_mri, sample_seg)
        
        # Create sample-specific output directory
        sample_dir = os.path.join(args.output_dir, f'sample_{idx:03d}')
        Path(sample_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        print(f"  - Generating visualizations...")
        visualize_registration_results(results, sample_dir, slice_idx=args.slice_idx)
        visualize_attention_maps(results, sample_dir, slice_idx=args.slice_idx)
        visualize_gradcams(results, sample_dir, slice_idx=args.slice_idx)
        visualize_lambda_maps(results, sample_dir, slice_idx=args.slice_idx)
        visualize_flow_field(results, sample_dir, slice_idx=args.slice_idx)
        create_summary_report(results, sample_dir)
        
        print(f"  - Mean Dice: {results['mean_dice']:.4f}")
        print(f"  - Saved to: {sample_dir}")
        print()
    
    # Cleanup
    analyzer.cleanup()
    
    print("=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

