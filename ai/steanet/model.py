import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = F.gelu(x)
        return x

class STEANet(nn.Module):
    def __init__(self, in_channels=30, num_classes=4):
        super().__init__()
        self.metadata = {
            "model_type": "3d_cnn",
            "mesh_type": "geographic_grid"
        }
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                48,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 48),
            nn.GELU()
        )
        self.res1 = ResidualBlock3D(48)
        self.res2 = ResidualBlock3D(48)
        self.dilated = nn.Sequential(
            nn.Conv3d(
                48,
                64,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.res3 = ResidualBlock3D(64)
        self.head = nn.Sequential(
            nn.Conv3d(
                64,
                32,
                kernel_size=1,  # Based on state dict mismatch
                padding=0,
                bias=False
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(
                32,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.dilated(x)
        x = self.res3(x)
        return self.head(x)

    def predict(self, x):
        import os
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from ai.core.tensor_contract import HazardTensor
        
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return HazardTensor(data=probs, is_mesh=False)
