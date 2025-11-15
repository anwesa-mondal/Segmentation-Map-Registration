# Unsupervised Segmentation Map Registration

This repository implements an unsupervised deep learning framework for registering a surface-derived segmentation template to a target segmentation volume. The model learns a 3D deformation field using volumetric similarity and smoothness constraints, without requiring paired training data. It supports multiple surface-aware loss functions and operates on medical datasets such as OASIS.

---

**Dataset**

We use the [Neurite-OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) brain MRI dataset. The `.npz` files are preprocessed into one-hot encoded `.npy` volumes using `convert_one_hot.py`. Each one-hot volume has **5 channels**, corresponding to:  
- Background  
- Cortex  
- Subcortical GM  
- White Matter  
- CSF

---