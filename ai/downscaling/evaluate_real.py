import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time

from diffusion import SmallUNet, ConditionalDiffusionModel
from conditioning import ConditionEncoder
from inference import DownscalingInference
from train_diffusion import DDPMNoiseScheduler
from normalization import DataNormalizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Real-Data Prototype on {device}...")

    variables = ['tp', 'u10', 'd2m', 't2m']
    
    # Normalizer
    stats_path = os.path.join(os.path.dirname(__file__), "normalization_stats.json")
    normalizer = DataNormalizer(variables)
    if os.path.exists(stats_path):
        normalizer.load(stats_path)
    
    # Load Model
    unet = SmallUNet(c_in=4, c_cond=32, c_out=4)
    cond_encoder = ConditionEncoder(c_in=4, cond_in=2, hidden=32)
    model = ConditionalDiffusionModel(unet, cond_encoder)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded trained model checkpoint successfully from {ckpt_path}.")
    else:
        print("WARNING: Checkpoint not found. Evaluating untrained model.")

    noise_scheduler = DDPMNoiseScheduler(device=device)
    infer = DownscalingInference(model, noise_scheduler, normalizer=normalizer, device=device)
    
    # 30-sample unseen test set
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    
    # We pass normalizer to dataset so we get normalized targets, but we denormalize for metrics!
    test_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="test", normalizer=normalizer)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    metrics = {var: {'bilinear_mse': 0.0, 'diffusion_mse': 0.0, 'bilinear_ext': 0.0, 'diffusion_ext': 0.0} for var in variables}
    total_valid_pixels = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            coarse = batch['coarse'].to(device)
            target_norm = batch['fine'].to(device)
            cond = batch['conditioning'].to(device)
            target_mask = batch['target_mask'].to(device)
            
            # True Target in physical units
            target_real = normalizer.inverse_transform(target_norm)
            
            # 1. Bilinear Baseline (Using raw unscaled coarse to output unscaled directly, or unscale the baseline)
            coarse_real = normalizer.inverse_transform(coarse)
            bilinear_out = infer.bilinear_baseline(coarse_real, target_h=451, target_w=401)
            
            # 2. Diffusion Inference
            # The inference method handles the reverse DDPM and denormalization automatically
            diffusion_out, _, _ = infer.downscale(coarse, cond, target_shape=(451, 401))
            
            # 3. Calculate Errors on Valid Pixels only
            for c, var in enumerate(variables):
                var_mask = target_mask[:, 0] # [B, T, H, W]
                if not var_mask.any(): continue
                
                t_val = target_real[:, c][var_mask]
                b_val = bilinear_out[:, c][var_mask]
                d_val = diffusion_out[:, c][var_mask]
                
                # Standard MSE
                metrics[var]['bilinear_mse'] += F.mse_loss(b_val, t_val, reduction='sum').item()
                metrics[var]['diffusion_mse'] += F.mse_loss(d_val, t_val, reduction='sum').item()
                
                # Extreme Preservation Check (Top 5% threshold defined per sample)
                if t_val.numel() > 0:
                    try:
                        q95 = torch.quantile(t_val, 0.95)
                        ext_mask = t_val >= q95
                        if ext_mask.any():
                            metrics[var]['bilinear_ext'] += F.mse_loss(b_val[ext_mask], t_val[ext_mask], reduction='sum').item()
                            metrics[var]['diffusion_ext'] += F.mse_loss(d_val[ext_mask], t_val[ext_mask], reduction='sum').item()
                    except:
                        pass
                
            total_valid_pixels += var_mask.sum().item()
            
    print("\n" + "="*50)
    print("      REAL-DATA TEST SET RESULTS (30 Sequences)      ")
    print("="*50)
    for var in variables:
        b_mse = metrics[var]['bilinear_mse'] / total_valid_pixels
        d_mse = metrics[var]['diffusion_mse'] / total_valid_pixels
        impr = ((b_mse - d_mse) / b_mse * 100) if b_mse > 0 else 0
        
        b_ext = metrics[var]['bilinear_ext'] / (total_valid_pixels * 0.05) # approx divisor
        d_ext = metrics[var]['diffusion_ext'] / (total_valid_pixels * 0.05)
        ext_impr = ((b_ext - d_ext) / b_ext * 100) if b_ext > 0 else 0
        
        print(f"[{var.upper()}] Overall MSE : Base = {b_mse:.4f} | Diff = {d_mse:.4f} | Impr = {impr:+.2f}%")
        print(f"[{var.upper()}] Extreme MSE : Base = {b_ext:.4f} | Diff = {d_ext:.4f} | Impr = {ext_impr:+.2f}%")
        print("-" * 50)
        
    print(f"Total evaluation time : {time.time() - start_time:.2f}s")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
