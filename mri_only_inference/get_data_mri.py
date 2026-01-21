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


class MRIDataset(Dataset):
    """
    Dataset for MRI-based registration with segmentation guidance.
    
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
        contrast_augmentation=True
    ):
        """
        Args:
            data_list_file: Text file with paths to subject segmentation files
            template_mri_path: Path to template MRI .npy file
            template_seg_path: Path to template segmentation .npy file
            target_size: Target volume size
            mri_filename: MRI filename in each subject directory
            seg_filename: Segmentation filename in each subject directory
            contrast_augmentation: If True, apply random contrast augmentation to simulate
                                   different MRI weightings (T1, T2, FLAIR, etc.)
        """
        # Read subject paths
        with open(data_list_file, 'r') as f:
            seg_paths = f.read().splitlines()
        
        # Get subject directories from segmentation paths
        self.subject_dirs = [os.path.dirname(p) for p in seg_paths]
        
        self.mri_filename = mri_filename
        self.seg_filename = seg_filename
        self.target_size = target_size
        self.contrast_augmentation = contrast_augmentation
        
        # Load and preprocess template MRI
        self.template_mri = self._load_mri(template_mri_path, target_size)
        
        # Load and preprocess template segmentation
        self.template_seg = self._load_seg(template_seg_path, target_size)
    
    def _load_mri(self, path, target_size):
        """Load and preprocess MRI volume."""
        mri = np.load(path)
        mri = torch.tensor(mri, dtype=torch.float32)
        
        # Add channel dimension if needed
        if mri.ndim == 3:
            mri = mri.unsqueeze(0)  # (1, D, H, W)
        
        # Normalize to [0, 1]
        mri_min, mri_max = mri.min(), mri.max()
        if mri_max - mri_min > 0:
            mri = (mri - mri_min) / (mri_max - mri_min)
        
        # Resize
        mri = F.interpolate(
            mri.unsqueeze(0),
            size=target_size,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
        
        return mri
    
    def _load_seg(self, path, target_size):
        """Load and preprocess segmentation volume."""
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)
        
        # Resize using nearest neighbor (for discrete labels)
        seg = F.interpolate(
            seg.unsqueeze(0),
            size=target_size,
            mode='nearest'
        ).squeeze(0)
        
        return seg
    
    def _augment_contrast(self, mri):
        """
        Apply random contrast augmentation to simulate different MRI weightings.
        This makes the model contrast-agnostic by varying intensity distributions.
        
        Optimized implementation with:
        - In-place operations where possible
        - Reduced tensor allocations
        - Vectorized random sampling
        - Single renormalization pass
        
        Transformations applied:
        - Gamma correction (simulates different tissue contrasts)
        - Intensity scaling and shifting
        - Histogram shift (simulates different windowing)
        - Random inversion (simulates T1 vs T2 contrast differences)
        
        Args:
            mri: (1, D, H, W) normalized MRI tensor in [0, 1]
        
        Returns:
            Augmented MRI tensor, still in [0, 1] range
        """
        # Clone once at the start
        mri_aug = mri.clone()
        
        # Generate all random values at once (more efficient)
        random_vals = torch.rand(4)
        gamma = random_vals[0].item() * 1.5 + 0.5  # [0.5, 2.0]
        scale = random_vals[1].item() * 0.6 + 0.7  # [0.7, 1.3]
        offset = random_vals[2].item() * 0.4 - 0.2  # [-0.2, 0.2]
        
        # 1. Gamma correction (in-place)
        # Low gamma (<1): brightens dark regions, simulates T2-like contrast
        # High gamma (>1): darkens, simulates T1-like contrast
        mri_aug.pow_(gamma)
        
        # 2. Intensity scaling and offset (fused operation)
        # Simulates different scanner calibrations and acquisition parameters
        mri_aug.mul_(scale).add_(offset)
        
        # 3. Random histogram shift (50% probability)
        # Compress or expand histogram around mean
        if random_vals[3].item() < 0.5:
            alpha = torch.rand(1).item() * 0.4 + 0.8  # [0.8, 1.2]
            mean_val = mri_aug.mean()
            mri_aug.sub_(mean_val).mul_(alpha).add_(mean_val)
        
        # 4. Random inversion (20% probability)
        # Simulates T1 vs T2 contrast inversion
        if torch.rand(1).item() < 0.2:
            mri_aug.neg_().add_(1.0)
        
        # Single renormalization to [0, 1] range
        mri_min = mri_aug.min()
        mri_max = mri_aug.max()
        range_val = mri_max - mri_min
        
        if range_val > 1e-6:
            mri_aug.sub_(mri_min).div_(range_val)
        else:
            # Fallback for edge case: uniform intensity
            mri_aug.clamp_(0, 1)
        
        return mri_aug
    
    def __len__(self):
        return len(self.subject_dirs)
    
    def __getitem__(self, idx):
        subject_dir = self.subject_dirs[idx]
        
        # Load sample MRI
        mri_path = os.path.join(subject_dir, self.mri_filename)
        sample_mri = self._load_mri(mri_path, self.target_size)
        
        # Load sample segmentation
        seg_path = os.path.join(subject_dir, self.seg_filename)
        sample_seg = self._load_seg(seg_path, self.target_size)
        
        # Apply contrast augmentation if enabled (for training)
        template_mri = self.template_mri
        if self.contrast_augmentation:
            template_mri = self._augment_contrast(template_mri)
            sample_mri = self._augment_contrast(sample_mri)
        
        return {
            'template_mri': template_mri,        # (1, D, H, W)
            'template_seg': self.template_seg,   # (5, D, H, W)
            'sample_mri': sample_mri,            # (1, D, H, W)
            'sample_seg': sample_seg,            # (5, D, H, W)
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