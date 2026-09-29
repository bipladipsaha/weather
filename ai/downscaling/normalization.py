import json
import os
import torch
import numpy as np

class DataNormalizer:
    """
    Handles normalization for downscaling datasets.
    Calculates statistics ONLY on training data, explicitly masking invalid/NaN pixels.
    """
    def __init__(self, variables):
        self.variables = variables
        self.stats = {}
    
    def fit(self, dataset, indices):
        """
        Calculates mean and std ONLY from the given indices of the dataset.
        Masks out NaNs during calculation.
        """
        print(f"Calculating normalization statistics on {len(indices)} training samples...")
        
        # Accumulators for Welford's algorithm or just collecting for small dataset
        channel_data = {var: [] for var in self.variables}
        valid_counts = {var: 0 for var in self.variables}
        
        for idx in indices:
            sample = dataset[idx] # Raw unnormalized
            target = sample['fine'] # [C, T, H, W]
            mask = sample['target_mask'] # [1, T, H, W]
            
            for c, var in enumerate(self.variables):
                t_var = target[c:c+1] # [1, T, H, W]
                # Get valid pixels
                valid_pixels = t_var[mask]
                
                # Only store finite values, double check
                valid_pixels = valid_pixels[torch.isfinite(valid_pixels)]
                
                channel_data[var].append(valid_pixels)
                valid_counts[var] += valid_pixels.numel()
                
        for var in self.variables:
            if not channel_data[var]:
                continue
            all_pixels = torch.cat(channel_data[var])
            mean = all_pixels.mean().item()
            std = all_pixels.std().item()
            
            # Simple precipitation-safe handling: if it's precip, sometimes we log transform.
            # For now we stick to standard z-score but note it.
            transform_used = "z-score (mean/std)"
            
            self.stats[var] = {
                "mean": mean,
                "std": std,
                "valid_pixel_count": valid_counts[var],
                "units": "Native Units", 
                "transformation_used": transform_used,
                "dataset_split_used": "train"
            }
            
        print("Normalization statistics calculated successfully.")
            
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=4)
        print(f"Saved normalization stats to {filepath}")
            
    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Normalization stats file not found: {filepath}")
        with open(filepath, 'r') as f:
            self.stats = json.load(f)
            
    def transform(self, tensor, channel_names=None):
        """
        Normalizes tensor in-place. NaNs naturally propagate, remaining NaN.
        """
        out = tensor.clone()
        c_names = channel_names or self.variables
        
        for c, var in enumerate(c_names):
            if var not in self.stats:
                continue
            mean = self.stats[var]['mean']
            std = self.stats[var]['std']
            
            # Prevent division by zero
            safe_std = std if std > 1e-6 else 1.0
            
            if out.dim() == 5: # [B, C, T, H, W]
                out[:, c] = (out[:, c] - mean) / safe_std
            elif out.dim() == 4: # [C, T, H, W]
                out[c] = (out[c] - mean) / safe_std
                
        return out
        
    def inverse_transform(self, tensor, channel_names=None):
        """
        Inverses the normalization. NaNs remain NaN.
        """
        out = tensor.clone()
        c_names = channel_names or self.variables
        for c, var in enumerate(c_names):
            if var not in self.stats:
                continue
            mean = self.stats[var]['mean']
            std = self.stats[var]['std']
            
            safe_std = std if std > 1e-6 else 1.0
            
            if out.dim() == 5:
                out[:, c] = out[:, c] * safe_std + mean
            elif out.dim() == 4:
                out[c] = out[c] * safe_std + mean
                
        return out
