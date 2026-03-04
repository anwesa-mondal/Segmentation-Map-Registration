

# Unsupervised Segmentation Map Registration

This repository implements an unsupervised deep learning framework for registering a surface-derived segmentation template to a target segmentation volume. The model learns a dense 3D deformation field using volumetric similarity and smoothness constraints, without requiring paired supervision. Experiments are conducted on medical segmentation datasets such as Neurite-OASIS.

---

## Dataset

We use the [Neurite-OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) brain MRI dataset.

Segmentations are converted into one-hot encoded `.npy` volumes using `convert_one_hot.py`.
Each volume has **5 channels** corresponding to:

* Background
* Cortex
* Subcortical Gray Matter
* White Matter
* CSF

All volumes are resized to `(128, 128, 128)` using **nearest-neighbor interpolation**.

---

## Setup

```bash
git clone https://github.com/karthiknm/Segmentation-Map-Registration.git
cd Segmentation-Map-Registration
pip install -r requirements.txt
```

### .gitignore

The repository includes a `.gitignore` that excludes:

* Large data files (`.npy`, `.nii.gz`)
* Model checkpoints and experiment outputs (`*.pth`, `weights*/`)
* WandB logs
* IDE and system files

---

## Model Overview

The core architecture is a **3D U-Net** that predicts a dense deformation field to warp a template segmentation to a target segmentation. Warping is performed using a differentiable **Spatial Transformer Network (STN)**.

### Input

* Template segmentation: 5 channels
* Target segmentation: 5 channels
* Concatenated input: **10 channels**

### Output

* Dense deformation field with **3 channels** (x, y, z displacement)

### Training Objective

The model is trained **unsupervised**, using:
**Strategy (λ-Regularized Deformable Registration)**

We extend the registration network to predict both:
- A dense deformation field to align the moving/template segmentation map to the fixed/target map, and
- A per-location **λ map** (regularization strength) via an additional output head.

Key idea: instead of using a single global smoothness weight, the model learns **where** deformation should be rigid vs flexible using the segmentation labels (2D label maps). Regions that should remain more rigid (e.g., white matter) are encouraged to have higher λ, while more deformable regions (e.g., ventricles/CSF) are encouraged to have lower λ.

Implementation notes:
- The λ head uses a **softplus** activation to keep λ positive and numerically stable.
- The classic smoothness regularizer is replaced by a **label-weighted λ-smoothness**, where λ scales the gradient penalties spatially.
- Training is end-to-end using both MRI/segmentation inputs and available segmentation labels for λ supervision; at test time, the model can run without labels and still outputs the deformation field (and λ map) for interpretation.

---

**File Overview**

* Dice overlap loss between warped template and target
* Cross-entropy loss
* Deformation smoothness (bending energy)
* Jacobian determinant regularization

No ground-truth deformation fields are used.

---

## Training Modes

The framework supports **three deformation strategies**:

### 1. `single` (Baseline)

A single-pass U-Net predicts one deformation field:

```
(template + target) → U-Net → deformation → warp(template)
```

### 2. `iter_shared` (Iterative, Shared Weights)

The same U-Net is applied repeatedly for refinement:

```
(template₀, target) → U-Net → warp → template₁
(template₁, target) → same U-Net → warp → template₂
...
```

* The U-Net weights are **shared** across iterations.
* Encourages consistent, progressive refinement.

### 3. `iter_unshared` (Iterative, Unshared Weights)
The main training script (`train.py`) uses a predefined `composite_loss` function from `losses.py`, which includes:

Core registration losses:
- Dice Loss: encourages overlap between warped moving labels and fixed labels.
- Cross Entropy Loss: provides dense label supervision complementing Dice, stabilizing early training.

Deformation regularization:
- (λ-)Smoothness / Bending Energy Loss: penalizes non-smooth deformations; in the λ-regularized version, this penalty is spatially weighted by the learned λ map (higher λ → stronger smoothing).
- Jacobian Determinant Loss: discourages folding/invalid warps by penalizing negative or extreme local volume changes and my removing self-intersection.

Additional λ-map objectives (strategy component):
- λ-Supervision Loss: uses anatomical regions from labels (e.g., ventricles/CSF vs white matter) to encourage lower λ in flexible regions and higher λ in rigid regions.
- λ-Prior Loss (Gaussian/Beta): keeps λ bounded, smooth, and interpretable.
- Displacement Magnitude Loss: discourages unnecessarily large deformations and improves stability.

Multiple U-Nets are applied sequentially:

```
(template₀, target) → U-Net₁ → warp → template₁
(template₁, target) → U-Net₂ → warp → template₂
...
```

* Each iteration has **independent parameters**.
* Increases model capacity at the cost of more parameters.

---

## File Overview

* `train.py`
  Training script supporting `single`, `iter_shared`, and `iter_unshared` modes.

* `model.py`
  Defines the 3D U-Net, iterative wrappers, and the `SpatialTransformer`.

* `losses.py`
  Dice loss, cross-entropy loss, bending energy loss, and Jacobian determinant loss.

* `get_data.py`
  Dataset loader and preprocessing utilities.

* `convert_one_hot.py`
  Converts `.nii.gz` segmentations into 5-channel `.npy` volumes.

* `test_model.py`
  Evaluation script that computes Dice, Jacobian statistics, and generates plots.

---

## Training

```bash
python train.py \
  --train_txt ./data/train_npy_list.txt \
  --template_path ./data/OASIS_OAS1_0406_MR1/seg5_onehot.npy \
  --mode iter_shared \
  --steps 2 \
  --epochs 50
```

### Key Arguments

* `--mode` : `single | iter_shared | iter_unshared`
* `--steps` : Number of deformation iterations (ignored for `single`)
* `--epochs` : Training epochs
* `--save_root` : Prefix for weight directories

Model weights are saved automatically as:

```
weights_<mode>_K<steps>_<timestamp>/final.pth
```

---

## Evaluation

Evaluation is performed on the training dataset using `test_model.py`.

```bash
python test_model.py
```

The script:

* Loads all trained models
* Computes Dice scores and Jacobian determinant statistics
* Logs results to WandB
* Saves plots and summaries in `test_viz/`

---

**Results**

- Dice score: **99.64**
- λ-map visualization: use the predicted λ map to interpret where the model prefers rigid vs flexible deformation.

Results (Google Drive): 
https://drive.google.com/file/d/1SKniB0R2gH20qeDjRafVpHuNmM7J7SsW/view?usp=sharing


