import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtremeAwareLoss(nn.Module):
    """
    Objective function combining standard diffusion loss with 
    explicit penalties for missing extreme events.
    """
    def __init__(self, lambda_recon=1.0, lambda_extreme=2.0, extreme_threshold=0.8):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_extreme = lambda_extreme
        self.extreme_threshold = extreme_threshold
        
    def forward(self, noise_pred, noise_target, clean_pred=None, clean_target=None, target_mask=None):
        """
        noise_pred, noise_target: Standard DDPM objective
        clean_pred, clean_target: Derived x_0 required for extreme loss (if applicable)
        target_mask: Boolean mask [B, 1, H, W] indicating valid pixels.
        """
        if target_mask is None:
            target_mask = torch.ones_like(noise_target, dtype=torch.bool)
            
        # Broadcast target_mask if necessary
        if target_mask.dim() < noise_target.dim():
            # Usually target_mask is [B, 1, T, H, W] or [1, T, H, W]. Let's just ensure it broadcasts
            pass

        # Calculate squared error
        se_diff = (noise_pred - noise_target) ** 2
        
        # Mean only over valid pixels
        loss_diff = se_diff[target_mask.expand_as(se_diff)].mean()
        
        loss_recon = 0.0
        loss_ext = 0.0
        
        if clean_pred is not None and clean_target is not None:
            # 2. Reconstruction Loss
            se_recon = (clean_pred - clean_target) ** 2
            loss_recon = se_recon[target_mask.expand_as(se_recon)].mean()
            
            # 3. Extreme-Aware Loss
            extreme_mask = (clean_target > self.extreme_threshold).float()
            underestimation = F.relu(clean_target - clean_pred)
            
            ext_vals = (underestimation * extreme_mask)
            loss_ext = ext_vals[target_mask.expand_as(ext_vals)].mean()
            
        total_loss = loss_diff + self.lambda_recon * loss_recon + self.lambda_extreme * loss_ext
        return total_loss
