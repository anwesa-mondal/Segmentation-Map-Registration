"""
MRI-Guided Registration Network with Anatomical Correction Module

Architecture:
- Input: Template MRI (1ch) + Template Seg (5ch) + Sample MRI (1ch)
- Dual-stream encoder: MRI stream + Segmentation attention stream
- Anatomical Correction Module (ACM) in decoder
- Multi-scale deformation with progressive refinement

Key difference from segmentation-only approach:
- MRI provides continuous intensity information
- Segmentation provides anatomical structure guidance
- Combined for robust registration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class AffineNet(nn.Module):
    """
    Predicts a 3x4 affine transformation matrix for coarse alignment.

    Takes concatenated [template_mri, sample_mri] as input and predicts
    an affine matrix initialized to identity. Used as a pre-alignment
    step before the deformable registration network.
    """
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(64, 12)
        # Initialize to identity transform
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
        ], dtype=torch.float))

    def forward(self, template_mri, sample_mri):
        """
        Args:
            template_mri: (B, 1, D, H, W)
            sample_mri: (B, 1, D, H, W)
        Returns:
            affine_matrix: (B, 3, 4)
        """
        x = torch.cat([template_mri, sample_mri], dim=1)
        features = self.conv(x).view(x.size(0), -1)
        affine_params = self.fc(features)
        return affine_params.view(-1, 3, 4)


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Basic conv block with InstanceNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class SegmentationAttentionModule(nn.Module):
    """
    Segmentation Attention Module (SAM)
    
    Uses template segmentation to generate attention weights that focus
    on anatomically important regions and boundaries.
    """
    def __init__(self, feature_channels, seg_channels):
        super().__init__()
        
        # Boundary detection from segmentation
        self.boundary_conv = nn.Sequential(
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(seg_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention generation
        self.attention_conv = nn.Sequential(
            nn.Conv3d(feature_channels + seg_channels, feature_channels, kernel_size=1),
            nn.InstanceNorm3d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Attention weights in [0, 1]
        )
        
        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features, seg_map):
        """
        Args:
            features: (B, C, D, H, W) - MRI feature map
            seg_map: (B, 5, D, H, W) - segmentation map
        Returns:
            refined_features, attention_weights
        """
        if seg_map.shape[2:] != features.shape[2:]:
            seg_map = F.interpolate(seg_map, size=features.shape[2:], mode='nearest')
        
        # Detect boundaries
        seg_boundaries = self.boundary_conv(seg_map)
        
        # Generate attention
        combined = torch.cat([features, seg_boundaries], dim=1)
        attention_weights = self.attention_conv(combined)
        
        # Apply attention and refine
        attended = features * attention_weights
        refined = self.refine_conv(attended)
        
        return refined, attention_weights


class AnatomicalCorrectionModule(nn.Module):
    """
    Anatomical Correction Module (ACM)
    
    Ensures deformations respect anatomical boundaries by:
    1. Incorporating segmentation structure into decoder features
    2. Encouraging smooth deformations within structures
    3. Allowing flexible deformations at boundaries
    """
    def __init__(self, channels, seg_channels=5):
        super().__init__()
        
        self.correction = nn.Sequential(
            nn.Conv3d(channels + seg_channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features, seg_map):
        """
        Args:
            features: (B, C, D, H, W) - decoder features
            seg_map: (B, 5, D, H, W) - template segmentation
        Returns:
            corrected_features: (B, C, D, H, W)
        """
        # Resize seg_map if needed
        if seg_map.shape[2:] != features.shape[2:]:
            seg_map = F.interpolate(seg_map, size=features.shape[2:], mode='nearest')
        
        combined = torch.cat([features, seg_map], dim=1)
        return self.correction(combined)


# =============================================================================
# Encoder
# =============================================================================

class DualStreamEncoder(nn.Module):
    """
    Dual-stream encoder:
    - MRI stream: processes [template_mri, sample_mri] (2 channels)
    - Segmentation attention at each scale using template_seg
    """
    def __init__(self, mri_channels=2, seg_channels=5):
        super().__init__()
        
        # MRI encoder stream
        self.enc1 = ConvBlock(mri_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        
        # Segmentation encoder (for attention)
        self.seg_enc1 = ConvBlock(seg_channels, 32)
        self.seg_enc2 = ConvBlock(32, 64)
        self.seg_enc3 = ConvBlock(64, 128)
        self.seg_enc4 = ConvBlock(128, 256)
        
        # Segmentation Attention Modules
        self.sam1 = SegmentationAttentionModule(32, 32)
        self.sam2 = SegmentationAttentionModule(64, 64)
        self.sam3 = SegmentationAttentionModule(128, 128)
        self.sam4 = SegmentationAttentionModule(256, 256)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)
    
    def forward(self, mri_input, seg_input):
        """
        Args:
            mri_input: (B, 2, D, H, W) - concatenated [template_mri, sample_mri]
            seg_input: (B, 5, D, H, W) - template segmentation
        Returns:
            skip_connections, bottleneck_features, attention_maps
        """
        skip_connections = []
        attention_maps = []
        
        # Scale 1
        mri_e1 = self.enc1(mri_input)
        seg_e1 = self.seg_enc1(seg_input)
        fused_e1, attn1 = self.sam1(mri_e1, seg_e1)
        skip_connections.append(fused_e1)
        attention_maps.append(attn1)
        
        # Scale 2
        mri_p1 = self.pool(fused_e1)
        seg_p1 = self.pool(seg_e1)
        mri_e2 = self.enc2(mri_p1)
        seg_e2 = self.seg_enc2(seg_p1)
        fused_e2, attn2 = self.sam2(mri_e2, seg_e2)
        skip_connections.append(fused_e2)
        attention_maps.append(attn2)
        
        # Scale 3
        mri_p2 = self.pool(fused_e2)
        seg_p2 = self.pool(seg_e2)
        mri_e3 = self.enc3(mri_p2)
        seg_e3 = self.seg_enc3(seg_p2)
        fused_e3, attn3 = self.sam3(mri_e3, seg_e3)
        skip_connections.append(fused_e3)
        attention_maps.append(attn3)
        
        # Scale 4
        mri_p3 = self.pool(fused_e3)
        seg_p3 = self.pool(seg_e3)
        mri_e4 = self.enc4(mri_p3)
        seg_e4 = self.seg_enc4(seg_p3)
        fused_e4, attn4 = self.sam4(mri_e4, seg_e4)
        skip_connections.append(fused_e4)
        attention_maps.append(attn4)
        
        # Bottleneck
        bottleneck_in = self.pool(fused_e4)
        bottleneck_out = self.bottleneck(bottleneck_in)
        
        return skip_connections, bottleneck_out, attention_maps


# =============================================================================
# Decoder with ACM
# =============================================================================

class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder with Anatomical Correction Module at each scale.
    Progressive deformation refinement from coarse to fine.
    """
    def __init__(self, seg_channels=5):
        super().__init__()
        
        # Upsampling + decoder blocks
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)  # 256 up + 256 skip
        self.acm4 = AnatomicalCorrectionModule(256, seg_channels)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)  # 128 up + 128 skip
        self.acm3 = AnatomicalCorrectionModule(128, seg_channels)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)   # 64 up + 64 skip
        self.acm2 = AnatomicalCorrectionModule(64, seg_channels)
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)    # 32 up + 32 skip
        self.acm1 = AnatomicalCorrectionModule(32, seg_channels)
        
        # Multi-scale flow outputs
        self.flow4 = nn.Conv3d(256, 3, kernel_size=3, padding=1)
        self.flow3 = nn.Conv3d(128, 3, kernel_size=3, padding=1)
        self.flow2 = nn.Conv3d(64, 3, kernel_size=3, padding=1)
        self.flow1 = nn.Conv3d(32, 3, kernel_size=3, padding=1)
        
        # Lambda maps for adaptive regularization
        self.lambda4 = nn.Sequential(nn.Conv3d(256, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.lambda3 = nn.Sequential(nn.Conv3d(128, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.lambda2 = nn.Sequential(nn.Conv3d(64, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.lambda1 = nn.Sequential(nn.Conv3d(32, 1, kernel_size=3, padding=1), nn.Sigmoid())
        
        # Initialize flow layers to near-zero
        for flow_layer in [self.flow1, self.flow2, self.flow3, self.flow4]:
            nn.init.normal_(flow_layer.weight, 0, 1e-3)
            nn.init.zeros_(flow_layer.bias)
    
    def forward(self, bottleneck, skip_connections, template_seg):
        """
        Args:
            bottleneck: (B, 512, D/16, H/16, W/16)
            skip_connections: [e1, e2, e3, e4] from encoder
            template_seg: (B, 5, D, H, W) - for ACM
        Returns:
            final_flow, intermediate_flows, lambda_maps
        """
        intermediate_flows = []
        lambda_maps = []
        
        # Decoder scale 4
        up4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([up4, skip_connections[3]], dim=1))
        d4 = self.acm4(d4, template_seg)
        flow4 = self.flow4(d4)
        lambda4 = self.lambda4(d4)
        intermediate_flows.append(flow4)
        lambda_maps.append(lambda4)
        
        # Decoder scale 3 (with progressive refinement)
        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat([up3, skip_connections[2]], dim=1))
        d3 = self.acm3(d3, template_seg)
        flow3 = self.flow3(d3) + F.interpolate(flow4, size=d3.shape[2:], mode='trilinear', align_corners=False)
        lambda3 = self.lambda3(d3)
        intermediate_flows.append(flow3)
        lambda_maps.append(lambda3)
        
        # Decoder scale 2
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat([up2, skip_connections[1]], dim=1))
        d2 = self.acm2(d2, template_seg)
        flow2 = self.flow2(d2) + F.interpolate(flow3, size=d2.shape[2:], mode='trilinear', align_corners=False)
        lambda2 = self.lambda2(d2)
        intermediate_flows.append(flow2)
        lambda_maps.append(lambda2)
        
        # Decoder scale 1 (final)
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat([up1, skip_connections[0]], dim=1))
        d1 = self.acm1(d1, template_seg)
        final_flow = self.flow1(d1) + F.interpolate(flow2, size=d1.shape[2:], mode='trilinear', align_corners=False)
        lambda1 = self.lambda1(d1)
        intermediate_flows.append(final_flow)
        lambda_maps.append(lambda1)
        
        return final_flow, intermediate_flows, lambda_maps


# =============================================================================
# Main Model
# =============================================================================

class MRIRegistrationNet(nn.Module):
    """
    MRI-Guided Registration Network

    Optionally includes an affine pre-alignment stage that coarsely aligns the
    template to the sample before predicting a dense deformation field.

    Input:
        - template_mri: (B, 1, D, H, W) - template MRI scan
        - template_seg: (B, 5, D, H, W) - template segmentation (for guidance)
        - sample_mri: (B, 1, D, H, W) - sample MRI to register to

    Output:
        - final_flow: (B, 3, D, H, W) - deformation field (in affine-aligned space when affine is used)
        - intermediate_flows: list of flows at each scale
        - lambda_maps: list of lambda maps for adaptive regularization
        - attention_maps: list of attention maps from SAM
        - affine_matrix: (B, 3, 4) or None - predicted affine matrix
    """
    def __init__(self, seg_channels=5, use_affine=False):
        super().__init__()

        self.use_affine = use_affine
        if use_affine:
            self.affine_net = AffineNet(in_channels=2)

        self.encoder = DualStreamEncoder(mri_channels=2, seg_channels=seg_channels)
        self.decoder = MultiScaleDecoder(seg_channels=seg_channels)

    def forward(self, template_mri, template_seg, sample_mri):
        """
        Args:
            template_mri: (B, 1, D, H, W)
            template_seg: (B, 5, D, H, W)
            sample_mri: (B, 1, D, H, W)
        """
        affine_matrix = None

        if self.use_affine:
            affine_matrix = self.affine_net(template_mri, sample_mri)
            affine_grid = F.affine_grid(affine_matrix, template_mri.size(), align_corners=False)
            template_mri = F.grid_sample(
                template_mri, affine_grid, mode='bilinear',
                padding_mode='border', align_corners=False
            )
            template_seg = F.grid_sample(
                template_seg, affine_grid, mode='nearest',
                padding_mode='border', align_corners=False
            )

        # Concatenate MRI inputs
        mri_input = torch.cat([template_mri, sample_mri], dim=1)  # (B, 2, D, H, W)

        # Encode
        skip_connections, bottleneck, attention_maps = self.encoder(mri_input, template_seg)

        # Decode with anatomical correction
        final_flow, intermediate_flows, lambda_maps = self.decoder(
            bottleneck, skip_connections, template_seg
        )

        return final_flow, intermediate_flows, lambda_maps, attention_maps, affine_matrix


# =============================================================================
# Spatial Transformer
# =============================================================================

class SpatialTransformer(nn.Module):
    """Spatial Transformer Network for warping volumes."""
    def __init__(self, size, device='cpu'):
        super().__init__()
        D, H, W = size

        # Pixel-centre coordinates under align_corners=False:  (2*i + 1)/N - 1
        lin_z = (2 * torch.arange(D, device=device).float() + 1) / D - 1
        lin_y = (2 * torch.arange(H, device=device).float() + 1) / H - 1
        lin_x = (2 * torch.arange(W, device=device).float() + 1) / W - 1
        zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')
        
        id_grid = torch.stack((xx, yy, zz), dim=-1)
        self.register_buffer('id_grid', id_grid.unsqueeze(0))
    
    def forward(self, moving, flow):
        """
        Args:
            moving: (B, C, D, H, W)
            flow: (B, 3, D, H, W)
        Returns:
            warped: (B, C, D, H, W)
        """
        B = moving.shape[0]
        flow = flow.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
        
        grid = self.id_grid.expand(B, -1, -1, -1, -1)
        warped_grid = grid + flow
        
        warped = F.grid_sample(
            moving, warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        return warped