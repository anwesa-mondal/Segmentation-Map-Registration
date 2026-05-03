"""
Compute Self-Intersection Loss for MRI Registration Model

Self-intersection (folding) occurs when the Jacobian determinant of the 
deformation field becomes negative. This indicates that the transformation
is no longer diffeomorphic (one-to-one and invertible), meaning different 
parts of the source image are mapped to the same location in the target space.

The self-intersection loss is computed as:
    L_self_intersection = mean(ReLU(-det(J)))

Where:
- J is the Jacobian matrix of the deformation field
- det(J) is the determinant at each voxel
- ReLU(-det(J)) penalizes only negative determinants (folding regions)

Usage:
    python compute_self_intersection.py --which_timestamp 20251129_053413 --num_samples 10
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import yaml

# Local imports
from model_mri import MRIRegistrationNet, SpatialTransformer
from losses_mri import compute_dice_score, jacobian_det_loss
from get_data_mri import MRIDataset


def _load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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
        dict with detailed self-intersection statistics
    """
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
    
    # Compute determinant: det(J)
    det = (dx_dx * (dy_dy * dz_dz - dy_dz * dz_dy) -
           dx_dy * (dy_dx * dz_dz - dy_dz * dz_dx) +
           dx_dz * (dy_dx * dz_dy - dy_dy * dz_dx))
    
    # Self-intersection loss: penalize negative determinants
    self_intersection_loss = F.relu(-det).mean()
    
    # Compute statistics
    det_np = det.detach().cpu().numpy()
    folding_mask = det_np < 0
    num_folding_voxels = folding_mask.sum()
    total_voxels = det_np.size
    folding_percentage = (num_folding_voxels / total_voxels) * 100
    
    return {
        'loss': self_intersection_loss.item(),
        'folding_percentage': folding_percentage,
        'num_folding_voxels': int(num_folding_voxels),
        'total_voxels': int(total_voxels),
        'min_jacobian': float(det_np.min()),
        'max_jacobian': float(det_np.max()),
        'mean_jacobian': float(det_np.mean()),
        'std_jacobian': float(det_np.std()),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute Self-Intersection Loss')
    parser.add_argument('--which_timestamp', type=str, default='20251129_053413',
                       help='Timestamp of the training run')
    parser.add_argument('--data_split', type=str, default='val',
                       choices=['train', 'val'], help='Data split to use')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to analyze (-1 for all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run inference on')
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128],
                       help='Target volume size')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml (reads use_affine and data paths)')

    args = parser.parse_args()

    # Load config (falls back to defaults if not provided)
    cfg = None
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    if Path(config_path).exists():
        cfg = _load_config(config_path)

    # Build paths
    base_dir = f"/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_mri_acm/{args.which_timestamp}"
    checkpoint_path = f"{base_dir}/checkpoints/best_model.pth"

    target_size = tuple(args.target_size)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Data paths — prefer config, fall back to hardcoded defaults
    if cfg is not None:
        template_mri_path = cfg['data']['template_mri_path']
        template_seg_path = cfg['data']['template_seg_path']
        if args.data_split == 'val':
            data_list = cfg['data']['val_txt']
        else:
            data_list = cfg['data']['train_txt']
    else:
        if args.data_split == 'val':
            data_list = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/val.txt"
        else:
            data_list = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/train.txt"
        template_mri_path = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/brain.npy"
        template_seg_path = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy"

    # Read use_affine from config so checkpoint and model always match
    use_affine = cfg.get('affine', {}).get('enabled', False) if cfg else False

    print("=" * 70)
    print("SELF-INTERSECTION LOSS ANALYSIS")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Data split: {args.data_split}")
    print(f"use_affine: {use_affine}")

    # Load model
    print("\nLoading model...")
    model = MRIRegistrationNet(seg_channels=5, use_affine=use_affine).to(device)
    stn = SpatialTransformer(size=target_size, device=device).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best Dice from training: {checkpoint.get('best_dice', 'unknown'):.4f}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MRIDataset(
        data_list, template_mri_path, template_seg_path,
        target_size=target_size
    )
    
    num_samples = len(dataset) if args.num_samples == -1 else min(args.num_samples, len(dataset))
    print(f"Analyzing {num_samples} samples from {len(dataset)} total")
    
    # Storage for results
    all_results = []
    
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE...")
    print("=" * 70)
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Processing samples"):
            data = dataset[idx]
            
            # Move to device
            template_mri = data['template_mri'].unsqueeze(0).to(device)
            template_seg = data['template_seg'].unsqueeze(0).to(device)
            sample_mri = data['sample_mri'].unsqueeze(0).to(device)
            sample_seg = data['sample_seg'].unsqueeze(0).to(device)
            
            # Forward pass
            final_flow, intermediate_flows, lambda_maps, attention_maps, affine_matrix = model(
                template_mri, template_seg, sample_mri
            )

            # Warp template: compose affine + deformation when affine is enabled
            if affine_matrix is not None:
                affine_grid = F.affine_grid(affine_matrix, template_mri.size(), align_corners=False)
                aligned_seg = F.grid_sample(template_seg, affine_grid, mode='nearest',
                                            padding_mode='border', align_corners=False)
                warped_seg = stn(aligned_seg, final_flow)
            else:
                warped_seg = stn(template_seg, final_flow)
            
            # Compute metrics
            dice_per_class, mean_dice = compute_dice_score(warped_seg, sample_seg)
            
            # Compute self-intersection loss
            flow_cpu = final_flow.cpu()
            self_intersection_stats = compute_self_intersection_loss(flow_cpu)
            
            # Also compute using the original jacobian_det_loss for comparison
            jacobian_loss_original = jacobian_det_loss(flow_cpu).item()
            
            result = {
                'sample_idx': idx,
                'mean_dice': mean_dice,
                'dice_per_class': dice_per_class,
                **self_intersection_stats,
                'jacobian_det_loss_original': jacobian_loss_original,
            }
            all_results.append(result)
    
    # Aggregate statistics
    print("\n" + "=" * 70)
    print("SELF-INTERSECTION LOSS RESULTS")
    print("=" * 70)
    
    losses = [r['loss'] for r in all_results]
    folding_pcts = [r['folding_percentage'] for r in all_results]
    min_jacs = [r['min_jacobian'] for r in all_results]
    mean_jacs = [r['mean_jacobian'] for r in all_results]
    dice_scores = [r['mean_dice'] for r in all_results]
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    SELF-INTERSECTION LOSS SUMMARY                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  What is Self-Intersection Loss?                                     ║
║  ─────────────────────────────────────────────────────────────────   ║
║  Self-intersection (folding) occurs when the Jacobian determinant    ║
║  of the deformation field becomes negative. This means:              ║
║                                                                      ║
║    • det(J) > 0: Valid transformation (no folding)                   ║
║    • det(J) ≈ 1: No volume change                                    ║
║    • det(J) < 0: FOLDING - different source points map to same       ║
║                  target location (transformation is not invertible)  ║
║                                                                      ║
║  Loss = mean(ReLU(-det(J))) = penalizes only negative determinants   ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                         AGGREGATE STATISTICS                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Self-Intersection Loss:                                             ║
║    Mean:  {np.mean(losses):>10.6f}                                         ║
║    Std:   {np.std(losses):>10.6f}                                          ║
║    Min:   {np.min(losses):>10.6f}                                          ║
║    Max:   {np.max(losses):>10.6f}                                          ║
║                                                                      ║
║  Folding Percentage (% voxels with det(J) < 0):                      ║
║    Mean:  {np.mean(folding_pcts):>10.4f}%                                ║
║    Std:   {np.std(folding_pcts):>10.4f}%                                 ║
║    Min:   {np.min(folding_pcts):>10.4f}%                                 ║
║    Max:   {np.max(folding_pcts):>10.4f}%                                 ║
║                                                                      ║
║  Jacobian Determinant:                                               ║
║    Mean (avg):     {np.mean(mean_jacs):>10.4f}                           ║
║    Min (worst):    {np.min(min_jacs):>10.4f}                             ║
║                                                                      ║
║  Registration Quality (Dice Score):                                  ║
║    Mean:  {np.mean(dice_scores):>10.4f}                                  ║
║    Std:   {np.std(dice_scores):>10.4f}                                   ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                          INTERPRETATION                              ║
╠══════════════════════════════════════════════════════════════════════╣""")
    
    # Interpretation
    mean_folding = np.mean(folding_pcts)
    if mean_folding < 0.1:
        quality = "EXCELLENT"
        interpretation = "Almost no folding - deformations are diffeomorphic"
    elif mean_folding < 1.0:
        quality = "GOOD"
        interpretation = "Minimal folding - acceptable for most applications"
    elif mean_folding < 5.0:
        quality = "MODERATE"
        interpretation = "Some folding present - may affect registration quality"
    else:
        quality = "POOR"
        interpretation = "Significant folding - consider increasing regularization"
    
    print(f"""║                                                                      ║
║  Quality Assessment: {quality:<48} ║
║  {interpretation:<67} ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Per-sample details
    print("\nPER-SAMPLE RESULTS:")
    print("-" * 90)
    print(f"{'Sample':<8} {'Dice':>8} {'Self-Int Loss':>14} {'Folding %':>12} {'Min Jac':>10} {'Mean Jac':>10}")
    print("-" * 90)
    
    for r in all_results:
        print(f"{r['sample_idx']:<8} {r['mean_dice']:>8.4f} {r['loss']:>14.6f} {r['folding_percentage']:>11.4f}% {r['min_jacobian']:>10.4f} {r['mean_jacobian']:>10.4f}")
    
    print("-" * 90)
    print(f"{'MEAN':<8} {np.mean(dice_scores):>8.4f} {np.mean(losses):>14.6f} {np.mean(folding_pcts):>11.4f}% {np.mean(min_jacs):>10.4f} {np.mean(mean_jacs):>10.4f}")
    
    # Save results to JSON
    output_file = Path(base_dir) / "self_intersection_analysis.json"
    summary = {
        'config': {
            'timestamp': args.which_timestamp,
            'data_split': args.data_split,
            'num_samples': num_samples,
            'device': str(device),
        },
        'aggregate': {
            'self_intersection_loss': {
                'mean': float(np.mean(losses)),
                'std': float(np.std(losses)),
                'min': float(np.min(losses)),
                'max': float(np.max(losses)),
            },
            'folding_percentage': {
                'mean': float(np.mean(folding_pcts)),
                'std': float(np.std(folding_pcts)),
                'min': float(np.min(folding_pcts)),
                'max': float(np.max(folding_pcts)),
            },
            'jacobian': {
                'mean_of_means': float(np.mean(mean_jacs)),
                'worst_min': float(np.min(min_jacs)),
            },
            'dice': {
                'mean': float(np.mean(dice_scores)),
                'std': float(np.std(dice_scores)),
            },
            'quality_assessment': quality,
        },
        'per_sample': all_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

