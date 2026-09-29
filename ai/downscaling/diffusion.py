import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SmallUNet(nn.Module):
    """
    A true conditional diffusion UNet that fits inside 4 GB VRAM.
    Accepts noisy state, timestep embedding, and conditioning features.
    """
    def __init__(self, c_in=4, c_cond=32, c_out=4, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        base_channels = 32
        
        self.conv0 = nn.Conv2d(c_in + c_cond, base_channels, 3, padding=1)
        
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1)
        self.te1 = nn.Linear(time_dim, base_channels*2)
        self.conv1 = nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1)
        
        self.bot1 = nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1)
        self.te_bot = nn.Linear(time_dim, base_channels*2)
        self.bot2 = nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1)
        
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1)
        self.te2 = nn.Linear(time_dim, base_channels)
        self.conv2 = nn.Conv2d(base_channels*2, base_channels, 3, padding=1)
        
        self.out = nn.Conv2d(base_channels, c_out, 1)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t)
        
        h = torch.cat([x, cond], dim=1)
        x0 = F.relu(self.conv0(h))
        
        x1 = self.down1(x0)
        x1 = x1 + self.te1(t_emb)[:, :, None, None]
        x1 = F.relu(self.conv1(x1))
        
        x2 = self.bot1(x1)
        x2 = x2 + self.te_bot(t_emb)[:, :, None, None]
        x2 = F.relu(self.bot2(x2))
        
        x_up = self.up1(x2)
        diffY = x0.size()[2] - x_up.size()[2]
        diffX = x0.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x_up = x_up + self.te2(t_emb)[:, :, None, None]
        
        x3 = torch.cat([x_up, x0], dim=1)
        x3 = F.relu(self.conv2(x3))
        
        return self.out(x3)

class ConditionalDiffusionModel(nn.Module):
    """
    Wrapper mapping the Time dimension into the UNet spatial denoiser.
    """
    def __init__(self, unet, condition_encoder):
        super().__init__()
        self.unet = unet
        self.cond_encoder = condition_encoder
        
    def forward(self, noisy_target, time_steps, coarse_input, cond_features):
        B, C, T, H5, W5 = noisy_target.shape
        
        # 1. Get conditioning
        cond_encoded = self.cond_encoder(coarse_input, cond_features, target_shape=(H5, W5))
        
        # 2. Reshape to run through 2D UNet
        noisy_target_2d = noisy_target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H5, W5)
        cond_2d = cond_encoded.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H5, W5)
        
        t_expanded = time_steps.repeat_interleave(T)
        
        # 3. Denoise
        noise_pred_2d = self.unet(noisy_target_2d, t_expanded, cond_2d)
        
        # 4. Reshape back
        noise_pred = noise_pred_2d.view(B, T, C, H5, W5).permute(0, 2, 1, 3, 4)
        
        return noise_pred
