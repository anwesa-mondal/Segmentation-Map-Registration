# MRI-Guided Brain Registration

Deformable brain MRI registration using anatomical segmentation guidance. The model aligns brain MRI scans while respecting tissue boundaries through an Anatomical Correction Module (ACM).

**Performance**: Dice score = 0.89 with no self-intersection/folding/tearing.

## Project Structure

```
mri_only_inference/
├── config.yaml          # All configuration (paths, hyperparameters, loss weights)
├── convert_brain_mri.py # Step 1: Convert NIfTI to NumPy
├── train_mri.py         # Step 2: Train the model
├── get_data_mri.py      # Dataset loader
├── model_mri.py         # Network architecture
├── losses_mri.py        # Loss functions
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data Directory

Organize your data as follows:

```
data_root/
├── scans/
│   ├── subject_001/
│   │   ├── norm.nii.gz       # Normalized MRI (FreeSurfer output)
│   │   └── seg4_onehot.npy   # Segmentation (5, D, H, W) one-hot
│   ├── subject_002/
│   └── ...
│   └── subjects.txt          # List of subject folder names (for convert_brain_mri.py)
├── train.txt                 # Training paths (full paths to seg files)
└── val.txt                   # Validation paths
```

### 3. Configure Paths

Edit `config.yaml` with your paths:

```yaml
data:
  # For training (train_mri.py)
  train_txt: "/path/to/train.txt"
  val_txt: "/path/to/val.txt"
  template_mri_path: "/path/to/template/brain.npy"
  template_seg_path: "/path/to/template/seg4_onehot.npy"
  
  # For data conversion (convert_brain_mri.py)
  main_dir: "/path/to/scans/"
  subjects_txt: "/path/to/scans/subjects.txt"
```

## Running

### Step 1: Convert NIfTI to NumPy

Convert brain MRI scans from NIfTI format to NumPy arrays:

```bash
# Test on single subject first
python convert_brain_mri.py --test

# Process all subjects
python convert_brain_mri.py --all
```

This creates `brain.npy` in each subject directory.

### Step 2: Train Model

```bash
python train_mri.py
```

Training outputs are saved to timestamped directories:
```
output_dir/YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best_model.pth
│   └── checkpoint_epoch_XXX.pth
├── logs/
└── config.json
```

## File Descriptions

| File | Purpose |
|------|---------|
| `config.yaml` | Central configuration file. Contains all paths (data directories, template files), training hyperparameters (learning rate, batch size, epochs), loss function weights, and model settings. Edit this file to customize the pipeline. |
| `convert_brain_mri.py` | Data preprocessing script. Converts FreeSurfer `norm.nii.gz` files to normalized NumPy arrays (`brain.npy`). Run this first before training. |
| `train_mri.py` | Main training script. Loads config, initializes model/optimizer/scheduler, runs training loop with validation, saves checkpoints, and logs metrics. |
| `get_data_mri.py` | PyTorch Dataset class. Loads template MRI/segmentation and sample MRI/segmentation pairs. Handles resizing to target dimensions and optional contrast augmentation for robustness to different MRI weightings (T1/T2/FLAIR). |
| `model_mri.py` | Neural network architecture. Implements dual-stream encoder (MRI + segmentation), Segmentation Attention Module (SAM), Anatomical Correction Module (ACM), and multi-scale decoder with progressive flow refinement. Also contains the Spatial Transformer for warping volumes. |
| `losses_mri.py` | Loss functions. Includes NCC (intensity matching), Dice (segmentation overlap), boundary alignment, smoothness regularization, bending energy, Jacobian determinant (prevents folding), and lambda-based adaptive regularization. |
| `requirements.txt` | Python dependencies for the project. |
