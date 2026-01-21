"""
Visualize the effect of contrast augmentation on MRI scans.

This script demonstrates how the contrast augmentation makes the model
contrast-agnostic by showing multiple augmented versions of the same MRI scan.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from get_data_mri import MRIDataset
import os


def visualize_contrast_variations(dataset_with_aug, dataset_no_aug, num_samples=5, slice_idx=64):
    """
    Visualize original MRI and multiple augmented versions.
    
    Args:
        dataset_with_aug: Dataset with contrast_augmentation=True
        dataset_no_aug: Dataset with contrast_augmentation=False
        num_samples: Number of augmented samples to show
        slice_idx: Which slice to visualize (depth dimension)
    """
    # Get original sample (no augmentation)
    original_sample = dataset_no_aug[0]
    original_mri = original_sample['sample_mri'][0, slice_idx, :, :].numpy()
    
    # Get multiple augmented samples
    augmented_samples = []
    for _ in range(num_samples):
        aug_sample = dataset_with_aug[0]
        aug_mri = aug_sample['sample_mri'][0, slice_idx, :, :].numpy()
        augmented_samples.append(aug_mri)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Contrast Augmentation: Simulating Different MRI Weightings', fontsize=16)
    
    # Show original
    axes[0, 0].imshow(original_mri, cmap='gray')
    axes[0, 0].set_title('Original MRI', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Show augmented versions
    titles = [
        'Augmented (T1-like)',
        'Augmented (T2-like)',
        'Augmented (FLAIR-like)',
        'Augmented (Different Scanner)',
        'Augmented (Inverted Contrast)'
    ]
    
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for i, (aug_mri, title, pos) in enumerate(zip(augmented_samples, titles, positions)):
        axes[pos].imshow(aug_mri, cmap='gray')
        axes[pos].set_title(title, fontsize=12)
        axes[pos].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/shared/home/v_nishchay_nilabh/code/seg-seg-reg/Segmentation-Map-Registration/contrast_augmentation_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Also show histogram comparison
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle('Intensity Histograms: Original vs Augmented', fontsize=16)
    
    # Original histogram
    axes2[0, 0].hist(original_mri.flatten(), bins=50, color='blue', alpha=0.7)
    axes2[0, 0].set_title('Original Histogram')
    axes2[0, 0].set_xlabel('Intensity')
    axes2[0, 0].set_ylabel('Frequency')
    
    # Augmented histograms
    for i, (aug_mri, pos) in enumerate(zip(augmented_samples, positions)):
        axes2[pos].hist(aug_mri.flatten(), bins=50, color='red', alpha=0.7)
        axes2[pos].set_title(f'Augmented {i+1} Histogram')
        axes2[pos].set_xlabel('Intensity')
        axes2[pos].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    output_path2 = '/shared/home/v_nishchay_nilabh/code/seg-seg-reg/Segmentation-Map-Registration/contrast_histograms_demo.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Histogram visualization saved to: {output_path2}")
    
    plt.show()


if __name__ == "__main__":
    # Paths (adjust as needed)
    data_list = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/train.txt"
    template_mri = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/brain.npy"
    template_seg = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy"
    
    if not (os.path.exists(data_list) and os.path.exists(template_mri)):
        print("Error: Data files not found. Please adjust paths in the script.")
        print(f"Looking for:")
        print(f"  - {data_list}")
        print(f"  - {template_mri}")
        exit(1)
    
    print("Loading datasets...")
    
    # Create dataset with augmentation
    dataset_aug = MRIDataset(
        data_list, 
        template_mri, 
        template_seg,
        contrast_augmentation=True
    )
    
    # Create dataset without augmentation
    dataset_no_aug = MRIDataset(
        data_list, 
        template_mri, 
        template_seg,
        contrast_augmentation=False
    )
    
    print(f"Dataset loaded with {len(dataset_aug)} samples")
    print("Generating contrast augmentation visualizations...")
    
    # Visualize
    visualize_contrast_variations(dataset_aug, dataset_no_aug, num_samples=5, slice_idx=64)
    
    print("\nDone! The model will now be trained on diverse contrast variations,")
    print("making it robust to different MRI acquisition protocols.")
