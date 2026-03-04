import torch
import torch.nn.functional as F


def dice_loss(y_pred,y_true,smooth=1e-5):
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims+2))
    intersection = 2 * (y_true * y_pred).sum(dim=vol_axes)
    union = y_true.sum(dim=vol_axes) + y_pred.sum(dim=vol_axes)
    dice = (intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def smoothing_loss(deformation_field):
    dx = torch.abs(deformation_field[:, :, 1:, :, :] - deformation_field[:, :, :-1, :, :])
    dy = torch.abs(deformation_field[:, :, :, 1:, :] - deformation_field[:, :, :, :-1, :])
    dz = torch.abs(deformation_field[:, :, :, :, 1:] - deformation_field[:, :, :, :, :-1])
    
    return torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)# losses.py


def bending_energy_loss(flow):
    # Second-order derivatives for smoothness
    d2x = flow[:,:,2:] - 2*flow[:,:,1:-1] + flow[:,:,:-2]
    d2y = flow[:,:,:,2:] - 2*flow[:,:,:,1:-1] + flow[:,:,:,:-2]
    d2z = flow[:,:,:,:,2:] - 2*flow[:,:,:,:,1:-1] + flow[:,:,:,:,:-2]
    

    return torch.mean(d2x**2) + torch.mean(d2y**2) + torch.mean(d2z**2)

def jacobian_det_loss(flow):
    """Prevent folding through Jacobian analysis"""
    J = torch.stack(torch.gradient(flow, dim=(2,3,4)), dim=2) 
    det = torch.det(J.permute(0,3,4,5,1,2))  
    return torch.mean(F.relu(-det))  

def cross_entropy_loss(pred, target):
    return F.cross_entropy(pred, target.argmax(dim=1))

#  Penalizes multiple labels being mapped to the same voxel
def label_overlap_loss(warped):
    """
    warped: (B, C, D, H, W) — one-hot warped prediction
    penalizes voxels that belong to more than one class after warping
    """
    label_sum = torch.sum(warped, dim=1)  # (B, D, H, W)
    collision = torch.clamp(label_sum - 1, min=0)  # overlap error
    return torch.mean(collision)

#  Optional: smoothness via first-order gradient variation
def deformation_direction_variation(flow):
    """
    Penalizes rapid change in deformation direction.
    flow: (B, 3, D, H, W)
    """
    grad_x = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    grad_y = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    grad_z = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]


    loss_x = torch.mean(grad_x ** 2)
    loss_y = torch.mean(grad_y ** 2)
    loss_z = torch.mean(grad_z ** 2)

    return loss_x + loss_y + loss_z


def composite_loss(pred, target, flow):
    return (
        0.8 * dice_loss(pred, target) + 
        0.2 * cross_entropy_loss(pred, target) +
        0.1 * bending_energy_loss(flow) +
        0.01 * jacobian_det_loss(flow)
    )
