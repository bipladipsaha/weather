import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DownscalingDataset(Dataset):
    """
    SYNTHETIC dataset for prototype architectural validation.
    No real paired 12km -> 5km data currently exists in the repository.
    
    Data Contract:
    coarse_input: [B, C, T, H12, W12] -> [B, 4, 9, 30, 26]
    conditioning: [B, Cond_C, H12, W12] -> [B, 2, 30, 26] (e.g. Terrain, Event Mask)
    target:       [B, C_out, T, H5, W5] -> [B, 4, 9, 72, 62]
    """
    def __init__(self, num_samples=10, c=4, t=9, h12=30, w12=26, h5=72, w5=62):
        self.num_samples = num_samples
        self.c = c
        self.t = t
        self.h12 = h12
        self.w12 = w12
        self.h5 = h5
        self.w5 = w5
        
        logger.warning("WARNING: Initializing SYNTHETIC DownscalingDataset. Paired 5km targets are MISSING from the repository.")
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # 12-km forecast probabilities (e.g., from STEA-Net/GNN)
        coarse = np.random.rand(self.c, self.t, self.h12, self.w12).astype(np.float32)
        
        # Terrain (static) and Event Mask (dynamic) as conditioning
        conditioning = np.random.rand(2, self.h12, self.w12).astype(np.float32)
        
        # 5-km high-res truth (MISSING in reality, mocked here)
        # Using a bicubic upsample + noise to mock a high-res structure
        coarse_tensor = torch.tensor(coarse)
        target = torch.nn.functional.interpolate(
            coarse_tensor, size=(self.h5, self.w5), mode='bicubic', align_corners=False
        )
        # Add high-frequency noise simulating local fine-scale extremes
        target += torch.randn_like(target) * 0.1
        target = torch.clamp(target, 0.0, 1.0)
        
        return torch.tensor(coarse), torch.tensor(conditioning), target.clone().detach()

def get_downscaling_dataloader(batch_size=2):
    ds = DownscalingDataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
