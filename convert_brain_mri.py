"""
Convert brain MRI NIfTI files to NumPy arrays for training.

Usage:
    # Single subject test:
    python convert_brain_mri.py --test
    
    # Process all subjects:
    python convert_brain_mri.py --all
"""

import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse


# Configuration
MAIN_DIR = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans"
INPUT_TXT = os.path.join(MAIN_DIR, "subjects.txt")
INPUT_NIFTI = "norm.nii.gz"  # Use normalized MRI (alternatives: aligned_norm.nii.gz, orig.nii.gz)
OUTPUT_NPY = "brain.npy"


def convert_single_subject(subject_path, input_filename=INPUT_NIFTI, output_filename=OUTPUT_NPY):
    """
    Convert a single subject's brain MRI from NIfTI to NumPy.
    
    Args:
        subject_path: Path to subject directory
        input_filename: Name of input NIfTI file
        output_filename: Name of output .npy file
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    input_path = os.path.join(subject_path, input_filename)
    output_path = os.path.join(subject_path, output_filename)
    
    if not os.path.exists(input_path):
        return False, f"Input file not found: {input_path}"
    
    try:
        # Load NIfTI
        img = nib.load(input_path)
        brain_mri = img.get_fdata().astype(np.float32)
        
        # Normalize to [0, 1] range
        brain_min = brain_mri.min()
        brain_max = brain_mri.max()
        if brain_max - brain_min > 0:
            brain_mri = (brain_mri - brain_min) / (brain_max - brain_min)
        
        # Save as .npy
        np.save(output_path, brain_mri)
        
        return True, f"Saved: {output_path} | Shape: {brain_mri.shape}"
    
    except Exception as e:
        return False, f"Error processing {subject_path}: {str(e)}"


def process_all_subjects(subjects_txt=INPUT_TXT, main_dir=MAIN_DIR):
    """Process all subjects listed in the text file."""
    
    if not os.path.exists(subjects_txt):
        print(f"Error: Subjects file not found: {subjects_txt}")
        return
    
    with open(subjects_txt, 'r') as f:
        subjects = f.read().splitlines()
    
    print(f"Found {len(subjects)} subjects to process")
    print(f"Input: {INPUT_NIFTI} -> Output: {OUTPUT_NPY}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for subject in tqdm(subjects, desc="Converting brain MRIs"):
        subject_path = os.path.join(main_dir, subject)
        success, msg = convert_single_subject(subject_path)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            tqdm.write(f"FAILED: {msg}")
    
    print("-" * 60)
    print(f"Completed: {success_count} successful, {fail_count} failed")


def test_single_subject():
    """Test conversion on a single subject."""
    
    test_subject = "OASIS_OAS1_0406_MR1"
    subject_path = os.path.join(MAIN_DIR, test_subject)
    
    print("=" * 60)
    print("Testing brain MRI conversion")
    print("=" * 60)
    print(f"Subject: {test_subject}")
    print(f"Path: {subject_path}")
    print()
    
    # List available files
    print("Available files:")
    for f in sorted(os.listdir(subject_path)):
        print(f"  - {f}")
    print()
    
    # Load and inspect the MRI
    input_path = os.path.join(subject_path, INPUT_NIFTI)
    print(f"Loading: {INPUT_NIFTI}")
    
    img = nib.load(input_path)
    brain_mri = img.get_fdata().astype(np.float32)
    
    print(f"Shape: {brain_mri.shape}")
    print(f"Dtype: {brain_mri.dtype}")
    print(f"Min: {brain_mri.min():.4f}")
    print(f"Max: {brain_mri.max():.4f}")
    print(f"Mean: {brain_mri.mean():.4f}")
    print()
    
    # Normalize
    brain_min = brain_mri.min()
    brain_max = brain_mri.max()
    brain_mri_norm = (brain_mri - brain_min) / (brain_max - brain_min)
    
    print("After normalization to [0, 1]:")
    print(f"Min: {brain_mri_norm.min():.4f}")
    print(f"Max: {brain_mri_norm.max():.4f}")
    print()
    
    # Save to test location first
    test_output = "/tmp/brain_test.npy"
    np.save(test_output, brain_mri_norm)
    print(f"Test saved to: {test_output}")
    
    # Verify by reloading
    reloaded = np.load(test_output)
    print(f"Reload verification - Shape: {reloaded.shape}, Min: {reloaded.min():.4f}, Max: {reloaded.max():.4f}")
    print()
    
    # Ask before saving to actual location
    print("=" * 60)
    print(f"Ready to save to: {os.path.join(subject_path, OUTPUT_NPY)}")
    print("Run with --all flag to process all subjects")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert brain MRI NIfTI to NumPy")
    parser.add_argument("--test", action="store_true", help="Test on single subject")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--input", type=str, default=INPUT_NIFTI, help=f"Input NIfTI filename (default: {INPUT_NIFTI})")
    parser.add_argument("--output", type=str, default=OUTPUT_NPY, help=f"Output NPY filename (default: {OUTPUT_NPY})")
    
    args = parser.parse_args()
    
    INPUT_NIFTI = args.input
    OUTPUT_NPY = args.output
    
    if args.test:
        test_single_subject()
    elif args.all:
        process_all_subjects()
    else:
        # Default: run test
        test_single_subject()

