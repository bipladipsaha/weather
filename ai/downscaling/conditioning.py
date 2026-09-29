import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionEncoder(nn.Module):
    """
    Encodes the 28-km coarse forecast and conditioning (terrain, event info)
    into a latent representation suitable for cross-attention or concatenation
    in the diffusion UNet.
    """
    def __init__(self, c_in=4, cond_in=2, hidden=32):
        super().__init__()
        # We upsample the coarse forecast to match the fine grid size 
        # so it can be concatenated cleanly with the UNet noisy inputs.
        
        self.conv_in = nn.Conv2d(c_in + cond_in, hidden, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        
    def forward(self, coarse, cond, target_shape):
        """
        coarse: [B, C, T, H28, W28]
        cond:   [B, Cond_C, H28, W28]
        target_shape: (H_fine, W_fine)
        Returns spatial conditioning tensor aligned to [B, hidden, T, H_fine, W_fine]
        """
        B, C, T, H12, W12 = coarse.shape
        
        # Flatten time into batch for 2D convolutions
        coarse_2d = coarse.view(B * T, C, H12, W12)
        
        # Repeat conditioning across time
        cond_repeated = cond.unsqueeze(2).repeat(1, 1, T, 1, 1).view(B * T, cond.shape[1], H12, W12)
        
        # Concatenate
        x = torch.cat([coarse_2d, cond_repeated], dim=1)
        
        # Basic extraction
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_out(x))
        
        # Upsample to target resolution
        x_up = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        
        # Reshape back to [B, hidden, T, H_fine, W_fine]
        return x_up.view(B, T, -1, target_shape[0], target_shape[1]).permute(0, 2, 1, 3, 4)
