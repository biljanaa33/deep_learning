# paper-inspired model
# src/models/rt_monodepth.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.use_residual = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)

        if self.use_residual:
            out = out + x

        return out


class PointwiseProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x


class RTMonoDepth(nn.Module):
    def __init__(self, max_depth=10.0):
        super().__init__()

        self.max_depth = max_depth

        # Encoder
        self.enc1 = ConvBlock(3, 32, stride=2)      # H/2
        self.enc2 = ConvBlock(32, 64, stride=2)     # H/4
        self.enc3 = ConvBlock(64, 128, stride=2)    # H/8
        self.enc4 = ConvBlock(128, 256, stride=2)   # H/16

        # Decoder
        self.up3 = UpConvBlock(256, 128)
        self.up2 = UpConvBlock(128, 64)
        self.up1 = UpConvBlock(64, 32)
        self.fuse1 = ConvBlock(64, 32)

        self.up0 = UpConvBlock(32, 16)

        self.pred = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)

        d3 = self.up3(f4)
        d3 = d3 + f3

        d2 = self.up2(d3)
        d2 = d2 + f2

        d1 = self.up1(d2)
        d1 = torch.cat([d1, f1], dim=1)
        d1 = self.fuse1(d1)

        d0 = self.up0(d1)

        depth = self.pred(d0) * self.max_depth

        return depth


class RTMonoDepthLiteEncoder(nn.Module):
    def __init__(self, max_depth=10.0):
        super().__init__()

        self.max_depth = max_depth

        # Encoder: MobileNet-style depthwise separable blocks.
        self.stem = ConvBNAct(3, 16, stride=2)            # H/2
        self.enc1 = nn.Sequential(
            DepthwiseSeparableBlock(16, 32),
            DepthwiseSeparableBlock(32, 32),
        )
        self.enc2 = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=2),    # H/4
            DepthwiseSeparableBlock(64, 64),
        )
        self.enc3 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2),   # H/8
            DepthwiseSeparableBlock(128, 128),
        )
        self.enc4 = nn.Sequential(
            DepthwiseSeparableBlock(128, 256, stride=2),  # H/16
            DepthwiseSeparableBlock(256, 256),
        )

        # Decoder is intentionally the same style as the baseline model.
        self.up3 = UpConvBlock(256, 128)
        self.up2 = UpConvBlock(128, 64)
        self.up1 = UpConvBlock(64, 32)
        self.fuse1 = ConvBlock(64, 32)

        self.up0 = UpConvBlock(32, 16)

        self.pred = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f1 = self.enc1(self.stem(x))
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)

        d3 = self.up3(f4)
        d3 = d3 + f3

        d2 = self.up2(d3)
        d2 = d2 + f2

        d1 = self.up1(d2)
        d1 = torch.cat([d1, f1], dim=1)
        d1 = self.fuse1(d1)

        d0 = self.up0(d1)

        depth = self.pred(d0) * self.max_depth

        return depth


def _load_mobilenet_v3_large_features(pretrained_backbone):
    try:
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models import mobilenet_v3_large
    except ImportError as exc:
        raise ImportError(
            "The mobilenet_v3 model requires torchvision. Install it with: "
            "python -m pip install torchvision"
        ) from exc

    try:
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        backbone = mobilenet_v3_large(weights=weights)
    except TypeError:
        backbone = mobilenet_v3_large(pretrained=pretrained_backbone)

    return backbone.features


class RTMonoDepthMobileNetV3(nn.Module):
    def __init__(self, max_depth=10.0, pretrained_backbone=False):
        super().__init__()

        self.max_depth = max_depth
        self.encoder = _load_mobilenet_v3_large_features(pretrained_backbone)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # MobileNetV3-Large feature channels at H/2, H/4, H/8, and H/16.
        self.proj1 = PointwiseProjection(16, 32)
        self.proj2 = PointwiseProjection(24, 64)
        self.proj3 = PointwiseProjection(40, 128)
        self.proj4 = PointwiseProjection(112, 256)

        self.up3 = UpConvBlock(256, 128)
        self.up2 = UpConvBlock(128, 64)
        self.up1 = UpConvBlock(64, 32)
        self.fuse1 = ConvBlock(64, 32)

        self.up0 = UpConvBlock(32, 16)

        self.pred = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _normalize_for_backbone(self, x):
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(self, x):
        x = self._normalize_for_backbone(x)

        x = self.encoder[0](x)
        x = self.encoder[1](x)
        f1 = self.proj1(x)

        x = self.encoder[2](x)
        x = self.encoder[3](x)
        f2 = self.proj2(x)

        x = self.encoder[4](x)
        x = self.encoder[5](x)
        x = self.encoder[6](x)
        f3 = self.proj3(x)

        x = self.encoder[7](x)
        x = self.encoder[8](x)
        x = self.encoder[9](x)
        x = self.encoder[10](x)
        x = self.encoder[11](x)
        x = self.encoder[12](x)
        f4 = self.proj4(x)

        d3 = self.up3(f4)
        d3 = d3 + f3

        d2 = self.up2(d3)
        d2 = d2 + f2

        d1 = self.up1(d2)
        d1 = torch.cat([d1, f1], dim=1)
        d1 = self.fuse1(d1)

        d0 = self.up0(d1)

        depth = self.pred(d0) * self.max_depth

        return depth
