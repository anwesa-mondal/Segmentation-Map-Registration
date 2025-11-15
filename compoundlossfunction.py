import torch
from losses import (
    dice_loss,
    cross_entropy_loss,
    lambda_weighted_smoothness_loss,
    lambda_prior_loss,
    displacement_magnitude_loss
)

def compound_loss(warped, fixed, flow, lambda_map, loss_weights=None):
    """
    Combines segmentation similarity + λ-regularization losses.

    Args:
        warped:       (B, C, D, H, W) warped moving image/segmentation
        fixed:        (B, C, D, H, W) fixed target image/segmentation
        flow:         (B, 3, D, H, W) deformation field
        lambda_map:   (B, 1, D, H, W) voxel-wise λ values
        loss_weights: dict with weights for each loss
    """
    if loss_weights is None:
        loss_weights = {
            "dice": 1.0,             # main segmentation alignment
            "cross_entropy": 0.0,    # optional, off by default
            "lambda_smoothness": 0.1,
            "lambda_prior": 0.05,
            "displacement": 0.01,
        }

    loss_dict = {}

    # --- Segmentation similarity losses ---
    loss_dict["dice"] = dice_loss(warped, fixed) * loss_weights["dice"]

    if loss_weights["cross_entropy"] > 0.0:
        loss_dict["cross_entropy"] = (
            cross_entropy_loss(warped, fixed) * loss_weights["cross_entropy"]
        )

    # --- λ-based adaptive regularization ---
    loss_dict["lambda_smoothness"] = (
        lambda_weighted_smoothness_loss(flow, lambda_map) * loss_weights["lambda_smoothness"]
    )
    loss_dict["lambda_prior"] = (
        lambda_prior_loss(lambda_map) * loss_weights["lambda_prior"]
    )

    # --- Displacement magnitude ---
    loss_dict["displacement"] = (
        displacement_magnitude_loss(flow) * loss_weights["displacement"]
    )

    # --- Total loss ---
    total_loss = sum(loss_dict.values())

    return total_loss, loss_dict
