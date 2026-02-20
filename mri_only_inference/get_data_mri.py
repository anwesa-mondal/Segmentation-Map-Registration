"""
MRI Dataset for Registration

Loads:
- Template MRI (1 channel) - grayscale brain scan
- Template Segmentation (5 channels) - one-hot encoded
- Sample MRI (1 channel) - target brain scan
- Sample Segmentation (5 channels) - for evaluation

Features:
- Contrast augmentation: Randomly varies MRI intensity distributions during training
  to simulate different MRI weightings (T1, T2, FLAIR, etc.), making the model
  contrast-agnostic and robust to different acquisition protocols.

File structure expected:
    subject_dir/
        brain.npy        - MRI scan (D, H, W) normalized to [0,1]
        seg4_onehot.npy  - Segmentation (5, D, H, W) one-hot

Configuration:
    Uses config.yaml for default paths and augmentation parameters.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
import yaml
from pathlib import Path


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def detect_and_correct_inversion(mri):
    """
    Detect and correct inverted MRI volumes.

    Brain MRI (any weighting) has a dark background that dominates the
    volume.  After [0,1] normalisation the median intensity should be
    well below 0.5.  If it is above, the image is inverted and we flip
    it back.  The threshold is intentionally generous -- a median above
    0.5 is unambiguous inversion for brain scans.

    Args:
        mri: Tensor of shape (1, D, H, W) in [0, 1].
    Returns:
        Corrected tensor (same shape), bool indicating whether inversion
        was applied.
    """
    median_val = mri.median().item()
    if median_val > 0.5:
        mri = 1.0 - mri
        return mri, True
    return mri, False


class MRIDataset(Dataset):
    """
    Dataset for MRI-based registration with segmentation guidance.
    
    Supports curriculum-based contrast augmentation: augmentation strength
    ramps from 0 to 1 over training via set_aug_intensity().
    
    Returns dict with:
        - template_mri: (1, D, H, W)
        - template_seg: (5, D, H, W)
        - sample_mri: (1, D, H, W)
        - sample_seg: (5, D, H, W)
    """
    
    def __init__(
        self, 
        data_list_file: str,
        template_mri_path: str,
        template_seg_path: str,
        target_size=(128, 128, 128),
        mri_filename="brain.npy",
        seg_filename="seg4_onehot.npy",
        contrast_augmentation=True,
        aug_config=None,
    ):
        """
        Args:
            data_list_file: Text file with paths to subject segmentation files
            template_mri_path: Path to template MRI .npy file
            template_seg_path: Path to template segmentation .npy file
            target_size: Target volume size
            mri_filename: MRI filename in each subject directory
            seg_filename: Segmentation filename in each subject directory
            contrast_augmentation: If True, apply random contrast augmentation
            aug_config: dict with augmentation ranges from config.yaml
        """
        with open(data_list_file, 'r') as f:
            seg_paths = f.read().splitlines()
        
        self.subject_dirs = [os.path.dirname(p) for p in seg_paths]
        
        self.mri_filename = mri_filename
        self.seg_filename = seg_filename
        self.target_size = target_size
        self.contrast_augmentation = contrast_augmentation

        # Augmentation parameters (defaults match tamed config values)
        ac = aug_config or {}
        gamma_range = ac.get('gamma_range', [0.7, 1.5])
        scale_range = ac.get('scale_range', [0.85, 1.15])
        offset_range = ac.get('offset_range', [-0.1, 0.1])
        hist_range = ac.get('histogram_alpha_range', [0.85, 1.15])
        self.gamma_lo, self.gamma_hi = gamma_range
        self.scale_lo, self.scale_hi = scale_range
        self.offset_lo, self.offset_hi = offset_range
        self.hist_lo, self.hist_hi = hist_range
        self.hist_shift_prob = ac.get('histogram_shift_prob', 0.3)
        self.inversion_prob = ac.get('inversion_prob', 0.1)

        # Curriculum intensity: 0.0 = no augmentation, 1.0 = full strength
        self._aug_intensity = 1.0
        
        self.template_mri = self._load_mri(template_mri_path, target_size)
        self.template_seg = self._load_seg(template_seg_path, target_size)

    def set_aug_intensity(self, intensity: float):
        """Set curriculum augmentation intensity in [0, 1]."""
        self._aug_intensity = max(0.0, min(1.0, intensity))
    
    def _load_mri(self, path, target_size):
        """Load and preprocess MRI volume."""
        mri = np.load(path)
        mri = torch.tensor(mri, dtype=torch.float32)
        
        if mri.ndim == 3:
            mri = mri.unsqueeze(0)
        
        mri_min, mri_max = mri.min(), mri.max()
        if mri_max - mri_min > 0:
            mri = (mri - mri_min) / (mri_max - mri_min)
        
        mri = F.interpolate(
            mri.unsqueeze(0),
            size=target_size,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
        
        mri, _ = detect_and_correct_inversion(mri)
        return mri
    
    def _load_seg(self, path, target_size):
        """Load and preprocess segmentation volume."""
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)
        
        seg = F.interpolate(
            seg.unsqueeze(0),
            size=target_size,
            mode='nearest'
        ).squeeze(0)
        
        return seg
    
    def _augment_contrast(self, mri):
        """
        Apply random contrast augmentation scaled by curriculum intensity.
        
        All augmentation ranges are interpolated between identity (intensity=0)
        and full range (intensity=1).
        """
        t = self._aug_intensity
        if t < 1e-6:
            return mri.clone()

        mri_aug = mri.clone()
        
        random_vals = torch.rand(4)

        # Interpolate ranges toward identity (gamma=1, scale=1, offset=0)
        gamma_lo = 1.0 + t * (self.gamma_lo - 1.0)
        gamma_hi = 1.0 + t * (self.gamma_hi - 1.0)
        gamma = random_vals[0].item() * (gamma_hi - gamma_lo) + gamma_lo

        scale_lo = 1.0 + t * (self.scale_lo - 1.0)
        scale_hi = 1.0 + t * (self.scale_hi - 1.0)
        scale = random_vals[1].item() * (scale_hi - scale_lo) + scale_lo

        offset_lo = t * self.offset_lo
        offset_hi = t * self.offset_hi
        offset = random_vals[2].item() * (offset_hi - offset_lo) + offset_lo

        mri_aug.pow_(gamma)
        mri_aug.mul_(scale).add_(offset)
        
        if random_vals[3].item() < self.hist_shift_prob * t:
            alpha_lo = 1.0 + t * (self.hist_lo - 1.0)
            alpha_hi = 1.0 + t * (self.hist_hi - 1.0)
            alpha = torch.rand(1).item() * (alpha_hi - alpha_lo) + alpha_lo
            mean_val = mri_aug.mean()
            mri_aug.sub_(mean_val).mul_(alpha).add_(mean_val)
        
        if torch.rand(1).item() < self.inversion_prob * t:
            mri_aug.neg_().add_(1.0)
        
        mri_min = mri_aug.min()
        mri_max = mri_aug.max()
        range_val = mri_max - mri_min
        
        if range_val > 1e-6:
            mri_aug.sub_(mri_min).div_(range_val)
        else:
            mri_aug.clamp_(0, 1)
        
        return mri_aug
    
    def __len__(self):
        return len(self.subject_dirs)
    
    def __getitem__(self, idx):
        subject_dir = self.subject_dirs[idx]
        
        mri_path = os.path.join(subject_dir, self.mri_filename)
        sample_mri = self._load_mri(mri_path, self.target_size)
        
        seg_path = os.path.join(subject_dir, self.seg_filename)
        sample_seg = self._load_seg(seg_path, self.target_size)
        
        template_mri = self.template_mri
        if self.contrast_augmentation:
            template_mri = self._augment_contrast(template_mri)
            sample_mri = self._augment_contrast(sample_mri)
            template_mri, _ = detect_and_correct_inversion(template_mri)
            sample_mri, _ = detect_and_correct_inversion(sample_mri)
        
        return {
            'template_mri': template_mri,
            'template_seg': self.template_seg,
            'sample_mri': sample_mri,
            'sample_seg': sample_seg,
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Load config for test paths
    config = load_config()
    
    if config is not None:
        data_list = config['data']['train_txt']
        template_mri = config['data']['template_mri_path']
        template_seg = config['data']['template_seg_path']
        target_size = tuple(config['model']['target_size'])
    else:
        raise Exception("Config file not found. Check paths in config.yaml or run convert_brain_mri.py first.")
    
    if os.path.exists(data_list) and os.path.exists(template_mri):
        print(f"Loading dataset from config.yaml...")
        print(f"  Data list: {data_list}")
        print(f"  Template MRI: {template_mri}")
        print(f"  Template Seg: {template_seg}")
        print(f"  Target size: {target_size}")
        print()
        
        dataset = MRIDataset(data_list, template_mri, template_seg, target_size=target_size)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Template MRI shape: {sample['template_mri'].shape}")
        print(f"Template Seg shape: {sample['template_seg'].shape}")
        print(f"Sample MRI shape: {sample['sample_mri'].shape}")
        print(f"Sample Seg shape: {sample['sample_seg'].shape}")
    else:
        print("Test data not found. Check paths in config.yaml or run convert_brain_mri.py first.")