import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetCore(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        def conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1),
                nn.InstanceNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, 3, padding=1),
                nn.InstanceNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        def up(in_c, out_c):
            return nn.ConvTranspose3d(in_c, out_c, 2, stride=2)

        self.enc1 = conv(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = conv(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = conv(256, 512)

        self.up4 = up(512, 256)
        self.dec4 = conv(512, 256)
        self.up3 = up(256, 128)
        self.dec3 = conv(256, 128)
        self.up2 = up(128, 64)
        self.dec2 = conv(128, 64)
        self.up1 = up(64, 32)
        self.dec1 = conv(64, 32)

        self.out = nn.Conv3d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.out(d1)


class IterativeRegNet(nn.Module):
    def __init__(self, in_channels, out_channels, steps=2, shared=True):
        super().__init__()
        self.steps = steps
        self.shared = shared

        if shared:
            self.net = UNetCore(in_channels, out_channels)
        else:
            self.nets = nn.ModuleList(
                [UNetCore(in_channels, out_channels) for _ in range(steps)]
            )

    def forward(self, moving, fixed, stn):
        warped = moving
        flow_total = torch.zeros(
            moving.shape[0], 3,
            moving.shape[2], moving.shape[3], moving.shape[4],
            device=moving.device
        )

        for k in range(self.steps):
            x = torch.cat([warped, fixed], dim=1)
            flow_k = self.net(x) if self.shared else self.nets[k](x)

            warped = stn(warped, flow_k)
            flow_total = flow_total + flow_k

        return flow_total


class SpatialTransformer(nn.Module):
    def __init__(self, size, device="cpu"):
        super().__init__()
        D, H, W = size
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        grid = torch.stack((xx, yy, zz), dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0))

    def forward(self, src, flow):
        flow = flow.permute(0, 2, 3, 4, 1)
        return F.grid_sample(
            src,
            self.grid + flow,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
