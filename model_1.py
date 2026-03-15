import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grid helper: correct identity grid for align_corners=False
# ---------------------------------------------------------------------------

def _identity_grid(size, device='cpu', dtype=torch.float32):
    """
    Build a (D, H, W, 3) identity sampling grid whose coordinates land on
    voxel centres under the align_corners=False convention used by
    grid_sample.

    align_corners=False maps voxel index i to:
        coord_i = 2*(i + 0.5)/N - 1  =  (2*i + 1)/N - 1

    so the range is  (-1 + 1/N,  1 - 1/N)  rather than  (-1, 1).
    """
    D, H, W = size
    lin_z = (2 * torch.arange(D, device=device, dtype=dtype) + 1) / D - 1
    lin_y = (2 * torch.arange(H, device=device, dtype=dtype) + 1) / H - 1
    lin_x = (2 * torch.arange(W, device=device, dtype=dtype) + 1) / W - 1
    zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')
    return torch.stack((xx, yy, zz), dim=-1)                   # (D, H, W, 3)


# ---------------------------------------------------------------------------
# Affine utilities: build a valid affine matrix from geometric parameters
# ---------------------------------------------------------------------------

def _angles_to_rotation_matrix(angles):
    """
    Convert 3 rotation angles (radians) to a 3x3 rotation matrix.
    Composes Rx @ Ry @ Rz (intrinsic XYZ convention).
    """
    cx, sx = torch.cos(angles[0]), torch.sin(angles[0])
    cy, sy = torch.cos(angles[1]), torch.sin(angles[1])
    cz, sz = torch.cos(angles[2]), torch.sin(angles[2])

    Rx = torch.stack([
        torch.stack([torch.ones_like(cx),  torch.zeros_like(cx), torch.zeros_like(cx)]),
        torch.stack([torch.zeros_like(cx), cx,                   sx]),
        torch.stack([torch.zeros_like(cx), -sx,                  cx]),
    ])
    Ry = torch.stack([
        torch.stack([cy,                   torch.zeros_like(cy), sy]),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy),  torch.zeros_like(cy)]),
        torch.stack([-sy,                  torch.zeros_like(cy), cy]),
    ])
    Rz = torch.stack([
        torch.stack([cz,                   sz,                   torch.zeros_like(cz)]),
        torch.stack([-sz,                  cz,                   torch.zeros_like(cz)]),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
    ])
    return Rx @ Ry @ Rz


def params_to_affine(translation, rotation, scale, shear):
    """
    Compose a 4x4 affine matrix from individual geometric components.
    All inputs are 1-D tensors on the same device.

    Parameters
    ----------
    translation : (3,)  shifts in x, y, z
    rotation    : (3,)  angles in *radians*
    scale       : (3,)  per-axis scale (passed through exp so network can output any real)
    shear       : (3,)  shear factors

    Returns
    -------
    (4, 4) affine matrix  =  T @ R @ Z @ S
    """
    device = translation.device
    dtype = translation.dtype

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, 3] = translation

    R = torch.eye(4, device=device, dtype=dtype)
    R[:3, :3] = _angles_to_rotation_matrix(rotation)

    Z = torch.diag(torch.cat([scale, torch.ones(1, device=device, dtype=dtype)]))

    S = torch.eye(4, device=device, dtype=dtype)
    S[0, 1] = shear[0]
    S[0, 2] = shear[1]
    S[1, 2] = shear[2]

    return T @ R @ Z @ S


def affine_to_dense_displacement(affine_4x4, size):
    """
    Convert a 4x4 affine matrix to a dense displacement field in
    normalised [-1, 1] coordinates (compatible with grid_sample).

    Parameters
    ----------
    affine_4x4 : (B, 4, 4)
    size        : (D, H, W)

    Returns
    -------
    disp : (B, 3, D, H, W)  displacement in normalised coords
    """
    B = affine_4x4.shape[0]
    D, H, W = size
    device = affine_4x4.device
    dtype = affine_4x4.dtype

    id_grid = _identity_grid(size, device=device, dtype=dtype)  # (D, H, W, 3)
    coords = id_grid.permute(3, 0, 1, 2)                       # (3, D, H, W)
    ones = torch.ones(1, D, H, W, device=device, dtype=dtype)
    homo = torch.cat([coords, ones], dim=0)                     # (4, D, H, W)
    homo_flat = homo.reshape(4, -1)                             # (4, N)

    A = affine_4x4[:, :3, :]                                   # (B, 3, 4)
    transformed = A @ homo_flat                                 # (B, 3, N)
    identity = coords.reshape(3, -1).unsqueeze(0).expand(B, -1, -1)
    disp = (transformed - identity).reshape(B, 3, D, H, W)
    return disp


# ---------------------------------------------------------------------------
# Affine network: predicts geometric parameters, composes valid affine
# ---------------------------------------------------------------------------

class AffineNet(nn.Module):
    """
    Predicts a *geometrically valid* 3-D affine transform from a pair of
    moving / fixed volumes.

    Instead of regressing 12 raw numbers, the network predicts:
      - 3 translation values
      - 3 rotation angles  (kept small via tanh scaling)
      - 3 log-scale values (exponentiated -> always positive)
      - 3 shear values     (kept small via tanh scaling)

    These are composed into a proper affine matrix:  T @ R @ Z @ S
    following the VoxelMorph parameterisation.
    """

    def __init__(self, in_channels=10, max_rotation_rad=0.17, max_shear=0.15):
        super().__init__()
        self.max_rotation_rad = max_rotation_rad
        self.max_shear = max_shear

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

        self.fc = nn.Linear(64, 12)

        # Initialise to identity: all zeros -> identity affine
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, moving, fixed):
        """
        Parameters
        ----------
        moving, fixed : (B, C, D, H, W)

        Returns
        -------
        affine_matrices : (B, 4, 4)  valid affine matrices
        """
        x = torch.cat([moving, fixed], dim=1)
        feat = self.encoder(x).view(x.size(0), -1)
        params = self.fc(feat)                                  # (B, 12)

        translation = params[:, 0:3]
        rotation = torch.tanh(params[:, 3:6]) * self.max_rotation_rad
        log_scale = params[:, 6:9]
        scale = torch.exp(log_scale)
        shear = torch.tanh(params[:, 9:12]) * self.max_shear

        matrices = []
        for i in range(params.size(0)):
            matrices.append(params_to_affine(
                translation[i], rotation[i], scale[i], shear[i]
            ))
        return torch.stack(matrices, dim=0)                     # (B, 4, 4)


# ---------------------------------------------------------------------------
# UNet (unchanged from test_1 branch, with dual heads: flow + lambda)
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels_flow=3, out_channels_lambda=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )

        def upsample_block(in_ch, out_ch):
            return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

        # Encoder
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(256, 512)

        # Decoder
        self.up4 = upsample_block(512, 256)
        self.dec4 = conv_block(512, 256)
        self.up3 = upsample_block(256, 128)
        self.dec3 = conv_block(256, 128)
        self.up2 = upsample_block(128, 64)
        self.dec2 = conv_block(128, 64)
        self.up1 = upsample_block(64, 32)
        self.dec1 = conv_block(64, 32)

        # Output heads
        self.flow_head = nn.Conv3d(32, out_channels_flow, kernel_size=1)
        self.lambda_head = nn.Sequential(
            nn.Conv3d(32, out_channels_lambda, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        up4 = self.up4(b)
        d4 = self.dec4(torch.cat((up4, e4), dim=1))
        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat((up3, e3), dim=1))
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat((up2, e2), dim=1))
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat((up1, e1), dim=1))

        deformation_field = self.flow_head(d1)           # (B, 3, D, H, W)
        lambda_map = self.lambda_head(d1)                 # (B, 1, D, H, W)

        return deformation_field, lambda_map


# ---------------------------------------------------------------------------
# Spatial Transformer (unchanged)
# ---------------------------------------------------------------------------

class SpatialTransformer(nn.Module):
    def __init__(self, size, device='cpu'):
        super().__init__()
        id_grid = _identity_grid(size, device=device)           # (D, H, W, 3)
        self.register_buffer('id_grid', id_grid.unsqueeze(0))   # (1, D, H, W, 3)

    def forward(self, moving, flow):
        B, C, D, H, W = moving.shape
        flow = flow.permute(0, 2, 3, 4, 1)
        grid = self.id_grid.expand(B, -1, -1, -1, -1)
        warped_grid = grid + flow
        warped = F.grid_sample(
            moving, warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        return warped
