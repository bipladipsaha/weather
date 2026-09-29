import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset
from ai.downscaling.diffusion import SmallUNet, ConditionalDiffusionModel
from ai.downscaling.conditioning import ConditionEncoder
from ai.downscaling.inference import DownscalingInference
from ai.downscaling.train_diffusion import DDPMNoiseScheduler
from ai.downscaling.normalization import DataNormalizer

def run_evaluation_audit():
    print("========================================")
    print("      COMPLETE EVALUATION AUDIT         ")
    print("========================================\n")
    
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    variables = ['tp', 'u10', 'd2m', 't2m']
    units = {'tp': 'm', 'u10': 'm/s', 'd2m': 'K', 't2m': 'K'}
    
    print("3. Physical Units:")
    for v in variables:
        print(f"   {v}: {units[v]}")
        
    # --- 6. Verify Spatial Coordinates ---
    print("\n6. Verifying Spatial Coordinates...")
    try:
        import glob
        era5_file = glob.glob(os.path.join(era5_dir, "*.nc"))[0]
        era5_land_file = glob.glob(os.path.join(era5_land_dir, "*.nc"))[0]
        
        c_ds = xr.open_dataset(era5_file)
        f_ds = xr.open_dataset(era5_land_file)
        
        def print_spatial(ds, name):
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            lat_res = abs(lats[1] - lats[0]) if len(lats) > 1 else 0
            lon_res = abs(lons[1] - lons[0]) if len(lons) > 1 else 0
            print(f"   {name}: Lat [{lats.min():.2f}, {lats.max():.2f}] res {lat_res:.3f} | Lon [{lons.min():.2f}, {lons.max():.2f}] res {lon_res:.3f}")
            print(f"      Lat ordering: {'Desc' if lats[0] > lats[-1] else 'Asc'}, Lon ordering: {'Desc' if lons[0] > lons[-1] else 'Asc'}")
            
        print_spatial(c_ds, "Coarse ERA5")
        print_spatial(f_ds, "Fine ERA5-Land")
    except Exception as e:
        print(f"   Could not read raw coordinates: {e}")

    # Load Normalizer & Dataset
    stats_path = os.path.join(os.path.dirname(__file__), "normalization_stats.json")
    normalizer = DataNormalizer(variables)
    if os.path.exists(stats_path):
        normalizer.load(stats_path)
    
    test_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="test", normalizer=normalizer)
    
    # Check variables in dataset
    print("\n4 & 5. Variables match and Coarse field check:")
    print(f"   Variables requested: {variables}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    unet = SmallUNet(c_in=4, c_cond=32, c_out=4)
    cond_encoder = ConditionEncoder(c_in=4, cond_in=2, hidden=32)
    model = ConditionalDiffusionModel(unet, cond_encoder)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
    noise_scheduler = DDPMNoiseScheduler(device=device)
    infer = DownscalingInference(model, noise_scheduler, normalizer=normalizer, device=device)
    
    metrics = {var: {'bilinear_mse': 0.0, 'diffusion_mse': 0.0, 'trivial_mse': 0.0, 'tp_mae': 0.0} for var in variables}
    total_valid_pixels = 0
    
    print("\n1, 2, 7, 8. Analyzing 3 Unseen Test Samples...")
    
    for i in range(3):
        sample = test_ds[i]
        
        print(f"\n--- Sample {i} ---")
        meta = sample['metadata']
        print(f"   Split: {meta['split']}, Start Time: {meta['start_time']}")
        
        # We need raw timestamps to check alignment
        c_sample_raw = test_ds.coarse_ds.isel(time=slice(i, i+test_ds.sequence_length))
        f_sample_raw = test_ds.fine_ds.isel(time=slice(i, i+test_ds.sequence_length))
        print(f"   Coarse timestamp: {c_sample_raw.time.values[0]}")
        print(f"   Fine timestamp  : {f_sample_raw.time.values[0]}")
        diff = (c_sample_raw.time.values[0] - f_sample_raw.time.values[0]).astype('timedelta64[s]')
        print(f"   Time difference : {diff}")
        
        coarse = sample['coarse'].unsqueeze(0).to(device)
        target_norm = sample['fine'].unsqueeze(0).to(device)
        cond = sample['conditioning'].unsqueeze(0).to(device)
        target_mask = sample['target_mask'].unsqueeze(0).to(device)
        
        # Denormalize for physical units
        target_real = normalizer.inverse_transform(target_norm)
        coarse_real = normalizer.inverse_transform(coarse)
        
        # Check Coarse zeros/NaNs
        zeros_pct = (coarse_real == 0).float().mean().item() * 100
        nans_pct = torch.isnan(coarse_real).float().mean().item() * 100
        print(f"   Coarse field -> Zeros: {zeros_pct:.1f}%, NaNs: {nans_pct:.1f}%")
        
        # Bilinear baseline on UNNORMALIZED coarse
        bilinear_out = infer.bilinear_baseline(coarse_real, target_h=451, target_w=401)
        
        # Diffusion prediction
        with torch.no_grad():
            diffusion_out, _, _ = infer.downscale(coarse, cond, target_shape=(451, 401))
            
        print(f"   Stats (min / max / mean / std):")
        for c, var in enumerate(variables):
            c_vals = coarse_real[0, c].cpu().numpy()
            t_vals = target_real[0, c][target_mask[0, 0]].cpu().numpy()
            b_vals = bilinear_out[0, c][target_mask[0, 0]].cpu().numpy()
            d_vals = diffusion_out[0, c][target_mask[0, 0]].cpu().numpy()
            
            print(f"     [{var}] Coarse    : {c_vals.min():.4f} / {c_vals.max():.4f} / {c_vals.mean():.4f} / {c_vals.std():.4f}")
            print(f"     [{var}] Target    : {t_vals.min():.4f} / {t_vals.max():.4f} / {t_vals.mean():.4f} / {t_vals.std():.4f}")
            print(f"     [{var}] Bilinear  : {b_vals.min():.4f} / {b_vals.max():.4f} / {b_vals.mean():.4f} / {b_vals.std():.4f}")
            print(f"     [{var}] Diffusion : {d_vals.min():.4f} / {d_vals.max():.4f} / {d_vals.mean():.4f} / {d_vals.std():.4f}")

            # Plotting Sample 0
            if i == 0:
                plt.figure(figsize=(15, 4))
                
                plt.subplot(1, 4, 1)
                plt.title(f"{var} Coarse (ERA5)")
                plt.imshow(c_vals[0], cmap='viridis')
                plt.colorbar()
                
                plt.subplot(1, 4, 2)
                plt.title(f"{var} Bilinear Upsample")
                plt.imshow(bilinear_out[0, c, 0].cpu().numpy(), cmap='viridis')
                plt.colorbar()
                
                plt.subplot(1, 4, 3)
                plt.title(f"{var} Target (ERA5-Land)")
                # Mask out NaNs for plot
                t_plot = target_real[0, c, 0].cpu().numpy()
                t_plot[~target_mask[0, 0, 0].cpu().numpy()] = np.nan
                plt.imshow(t_plot, cmap='viridis')
                plt.colorbar()
                
                plt.subplot(1, 4, 4)
                plt.title(f"{var} Diffusion Prediction")
                d_plot = diffusion_out[0, c, 0].cpu().numpy()
                d_plot[~target_mask[0, 0, 0].cpu().numpy()] = np.nan
                plt.imshow(d_plot, cmap='viridis')
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(__file__), f"audit_plot_sample0_{var}.png"))
                plt.close()
                
    print("\n--- 12 & 13. Re-running Error Metrics properly ---")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in test_loader:
            coarse = batch['coarse'].to(device)
            target_norm = batch['fine'].to(device)
            cond = batch['conditioning'].to(device)
            target_mask = batch['target_mask'].to(device)
            
            target_real = normalizer.inverse_transform(target_norm)
            coarse_real = normalizer.inverse_transform(coarse)
            
            bilinear_out = infer.bilinear_baseline(coarse_real, target_h=451, target_w=401)
            diffusion_out, _, _ = infer.downscale(coarse, cond, target_shape=(451, 401))
            
            # Trivial baseline (mean of valid target pixels per batch)
            
            for c, var in enumerate(variables):
                var_mask = target_mask[:, 0]
                if not var_mask.any(): continue
                
                t_val = target_real[:, c][var_mask]
                b_val = bilinear_out[:, c][var_mask]
                d_val = diffusion_out[:, c][var_mask]
                
                trivial_mean = t_val.mean()
                trivial_pred = torch.full_like(t_val, trivial_mean)
                
                metrics[var]['bilinear_mse'] += F.mse_loss(b_val, t_val, reduction='sum').item()
                metrics[var]['diffusion_mse'] += F.mse_loss(d_val, t_val, reduction='sum').item()
                metrics[var]['trivial_mse'] += F.mse_loss(trivial_pred, t_val, reduction='sum').item()
                
                if var == 'tp':
                    metrics[var]['tp_mae'] += F.l1_loss(d_val, t_val, reduction='sum').item()
                
            total_valid_pixels += var_mask.sum().item()
            
    for var in variables:
        b_mse = metrics[var]['bilinear_mse'] / total_valid_pixels
        d_mse = metrics[var]['diffusion_mse'] / total_valid_pixels
        t_mse = metrics[var]['trivial_mse'] / total_valid_pixels
        
        print(f"[{var.upper()}] Trivial Mean MSE: {t_mse:.6f}")
        print(f"[{var.upper()}] Bilinear MSE    : {b_mse:.6f}")
        print(f"[{var.upper()}] Diffusion MSE   : {d_mse:.6f}")
        
        if var == 'tp':
            d_mae = metrics[var]['tp_mae'] / total_valid_pixels
            print(f"[TP] Diffusion MAE (Full Precision) : {d_mae:.10f}")
            
    print("\nEVALUATION AUDIT COMPLETE.")

if __name__ == "__main__":
    run_evaluation_audit()
