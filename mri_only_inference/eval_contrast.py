"""
Evaluate a saved MRI registration model on the validation set
under consistent contrast conditions (simulating real scanner protocols).

Each evaluation run samples ONE fixed contrast transform and applies it
identically to every image in the validation set -- mimicking a real
scenario where all scans come from the same protocol (e.g. all T1, all
FLAIR, etc.).

Usage:
    python eval_contrast.py \
        --checkpoint /path/to/best_model.pth \
        [--config config.yaml] \
        [--device cuda:6] \
        [--num_runs 10]
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

from model_mri import MRIRegistrationNet, SpatialTransformer
from losses_mri import compute_dice_score
from get_data_mri import MRIDataset, detect_and_correct_inversion


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def sample_contrast_params(aug_config, rng=None):
    """
    Sample a single fixed set of contrast-transform parameters,
    representing one "scanner protocol".

    Returns a dict that fully determines the transform.
    """
    if rng is None:
        rng = np.random.default_rng()

    gamma_lo, gamma_hi = aug_config.get('gamma_range', [0.7, 1.5])
    scale_lo, scale_hi = aug_config.get('scale_range', [0.85, 1.15])
    offset_lo, offset_hi = aug_config.get('offset_range', [-0.1, 0.1])
    hist_lo, hist_hi = aug_config.get('histogram_alpha_range', [0.85, 1.15])
    hist_shift_prob = aug_config.get('histogram_shift_prob', 0.3)
    inversion_prob = aug_config.get('inversion_prob', 0.1)

    params = {
        'gamma': rng.uniform(gamma_lo, gamma_hi),
        'scale': rng.uniform(scale_lo, scale_hi),
        'offset': rng.uniform(offset_lo, offset_hi),
        'do_hist_shift': rng.random() < hist_shift_prob,
        'hist_alpha': rng.uniform(hist_lo, hist_hi),
        'do_invert': rng.random() < inversion_prob,
    }
    return params


def apply_fixed_contrast(mri, params):
    """Apply a deterministic contrast transform defined by `params`."""
    out = mri.clone()

    out.pow_(params['gamma'])
    out.mul_(params['scale']).add_(params['offset'])

    if params['do_hist_shift']:
        mean_val = out.mean()
        out.sub_(mean_val).mul_(params['hist_alpha']).add_(mean_val)

    if params['do_invert']:
        out.neg_().add_(1.0)

    v_min, v_max = out.min(), out.max()
    rng = v_max - v_min
    if rng > 1e-6:
        out.sub_(v_min).div_(rng)
    else:
        out.clamp_(0, 1)

    return out


def describe_params(params):
    """Human-readable one-liner for a contrast parameter set."""
    parts = [
        f"gamma={params['gamma']:.3f}",
        f"scale={params['scale']:.3f}",
        f"offset={params['offset']:.4f}",
    ]
    if params['do_hist_shift']:
        parts.append(f"hist_alpha={params['hist_alpha']:.3f}")
    if params['do_invert']:
        parts.append("INVERTED")
    return ", ".join(parts)


class FixedContrastValDataset(Dataset):
    """
    Wraps an MRIDataset (with contrast_augmentation=False) and applies
    a single fixed contrast transform to every sample.
    """

    def __init__(self, base_dataset, contrast_params):
        self.base = base_dataset
        self.params = contrast_params

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        sample['template_mri'] = apply_fixed_contrast(sample['template_mri'], self.params)
        sample['sample_mri'] = apply_fixed_contrast(sample['sample_mri'], self.params)
        sample['template_mri'], _ = detect_and_correct_inversion(sample['template_mri'])
        sample['sample_mri'], _ = detect_and_correct_inversion(sample['sample_mri'])
        return sample


@torch.no_grad()
def evaluate(model, stn, dataloader, device, num_classes=5):
    model.eval()
    all_dice = []
    all_per_class = [[] for _ in range(num_classes)]

    for batch in tqdm(dataloader, desc="Evaluating", ncols=90):
        template_mri = batch['template_mri'].to(device)
        template_seg = batch['template_seg'].to(device)
        sample_mri = batch['sample_mri'].to(device)
        sample_seg = batch['sample_seg'].to(device)

        final_flow, _, _, _, affine_matrix = model(template_mri, template_seg, sample_mri)

        # Compose affine + deformation when affine stage is enabled
        if affine_matrix is not None:
            affine_grid = F.affine_grid(affine_matrix, template_mri.size(), align_corners=False)
            aligned_seg = F.grid_sample(template_seg, affine_grid, mode='nearest',
                                        padding_mode='border', align_corners=False)
            warped_seg = stn(aligned_seg, final_flow)
        else:
            warped_seg = stn(template_seg, final_flow)

        dice_per_class, mean_dice = compute_dice_score(warped_seg, sample_seg, num_classes)
        all_dice.append(mean_dice)
        for c in range(num_classes):
            all_per_class[c].append(dice_per_class[c])

    return np.mean(all_dice), [np.mean(scores) for scores in all_per_class]


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate with per-run fixed contrast augmentation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default="config.yaml",
                        help='Path to config.yaml')
    parser.add_argument('--device', type=str, default="cuda:6",
                        help='Device (e.g. cuda:6)')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of different "protocols" to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible protocol sampling')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    target_size = tuple(cfg['model']['target_size'])
    num_classes = cfg['model']['num_classes']
    class_names = cfg['visualization']['class_names']

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Num runs   : {args.num_runs}")
    print(f"Seed       : {args.seed}")
    print()

    # --- Model ---
    use_affine = cfg.get('affine', {}).get('enabled', False)
    model = MRIRegistrationNet(seg_channels=num_classes, use_affine=use_affine).to(device)
    stn = SpatialTransformer(size=target_size, device=device).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"best_dice {ckpt.get('best_dice', '?')})")
    print()

    # --- Base val dataset (no augmentation) ---
    val_ds_clean = MRIDataset(
        cfg['data']['val_txt'],
        cfg['data']['template_mri_path'],
        cfg['data']['template_seg_path'],
        target_size=target_size,
        contrast_augmentation=False,
    )

    # --- Baseline: identity contrast ---
    print("=" * 60)
    print("Baseline (no contrast augmentation)")
    print("=" * 60)
    loader_clean = DataLoader(val_ds_clean, batch_size=1, shuffle=False, num_workers=0)
    clean_dice, clean_per_class = evaluate(model, stn, loader_clean, device, num_classes)

    print(f"  Mean Dice : {clean_dice:.4f}")
    for i, name in enumerate(class_names):
        print(f"  {name:15s}: {clean_per_class[i]:.4f}")
    print()

    # --- Fixed-contrast runs ---
    print("=" * 60)
    print(f"Fixed-contrast evaluation ({args.num_runs} protocols)")
    print("=" * 60)

    rng = np.random.default_rng(args.seed)
    aug_cfg = cfg['augmentation']

    run_dices = []
    run_per_class = [[] for _ in range(num_classes)]

    for run in range(args.num_runs):
        params = sample_contrast_params(aug_cfg, rng)
        desc = describe_params(params)

        wrapped_ds = FixedContrastValDataset(val_ds_clean, params)
        loader = DataLoader(wrapped_ds, batch_size=1, shuffle=False, num_workers=0)

        dice, per_class = evaluate(model, stn, loader, device, num_classes)
        run_dices.append(dice)
        for c in range(num_classes):
            run_per_class[c].append(per_class[c])

        print(f"  Run {run+1:2d}/{args.num_runs}  Dice: {dice:.4f}  [{desc}]")

    mean_aug = np.mean(run_dices)
    std_aug = np.std(run_dices)
    worst = np.min(run_dices)
    best = np.max(run_dices)

    print()
    print("-" * 60)
    print(f"  Mean Dice  : {mean_aug:.4f} +/- {std_aug:.4f}")
    print(f"  Best run   : {best:.4f}")
    print(f"  Worst run  : {worst:.4f}")
    print()
    print("  Per-class (mean +/- std across protocols):")
    for i, name in enumerate(class_names):
        m = np.mean(run_per_class[i])
        s = np.std(run_per_class[i])
        print(f"    {name:15s}: {m:.4f} +/- {s:.4f}")

    # --- Summary ---
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Clean val Dice          : {clean_dice:.4f}")
    print(f"  Fixed-contrast val Dice : {mean_aug:.4f} +/- {std_aug:.4f}")
    print(f"  Worst protocol          : {worst:.4f}")
    delta = mean_aug - clean_dice
    print(f"  Delta (mean)            : {delta:+.4f}")


if __name__ == "__main__":
    main()
