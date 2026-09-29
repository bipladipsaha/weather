import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset
from ai.downscaling.inference import DownscalingInference
from ai.downscaling.normalization import DataNormalizer

def compute_residuals_for_split(split, variables, era5_dir, era5_land_dir, normalizer, device):
    print(f"\n==============================================")
    print(f"      ANALYZING RESIDUALS: {split.upper()} SPLIT      ")
    print(f"==============================================")
    
    # Do NOT pass normalizer to dataset here so we can get raw unscaled values easily
    ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split=split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # We will instantiate DownscalingInference just for bilinear_baseline
    infer = DownscalingInference(model=None, noise_scheduler=None, normalizer=None, device=device)
    
    residuals = {var: [] for var in variables}
    
    with torch.no_grad():
        for batch in loader:
            # these are unnormalized because normalizer=None in dataset above
            coarse = batch['coarse'].to(device)
            target = batch['fine'].to(device)
            target_mask = batch['target_mask'].to(device)
            
            # compute baseline identically for all splits
            baseline = infer.bilinear_baseline(coarse, target_h=451, target_w=401)
            
            # calculate residuals only on valid pixels
            for c, var in enumerate(variables):
                var_mask = target_mask[:, 0]
                if not var_mask.any():
                    continue
                
                t_val = target[:, c][var_mask]
                b_val = baseline[:, c][var_mask]
                
                res = t_val - b_val
                residuals[var].append(res.cpu())
                
    # Aggregate and compute stats
    for var in variables:
        if not residuals[var]:
            print(f"[{var.upper()}] No data")
            continue
            
        all_res = torch.cat(residuals[var])
        
        res_mean = all_res.mean().item()
        res_std = all_res.std().item()
        res_mae = all_res.abs().mean().item()
        res_rmse = torch.sqrt((all_res ** 2).mean()).item()
        res_min = all_res.min().item()
        res_max = all_res.max().item()
        
        try:
            q95 = torch.quantile(all_res, 0.95).item()
            q99 = torch.quantile(all_res, 0.99).item()
        except:
            q95 = np.percentile(all_res.numpy(), 95)
            q99 = np.percentile(all_res.numpy(), 99)
            
        print(f"[{var.upper()}] Mean        : {res_mean:.6f}")
        print(f"[{var.upper()}] Std         : {res_std:.6f}")
        print(f"[{var.upper()}] MAE         : {res_mae:.6f}")
        print(f"[{var.upper()}] RMSE        : {res_rmse:.6f}")
        print(f"[{var.upper()}] Min / Max   : {res_min:.6f} / {res_max:.6f}")
        print(f"[{var.upper()}] 95th / 99th : {q95:.6f} / {q99:.6f}")
        print("-" * 46)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    variables = ['tp', 'u10', 'd2m', 't2m']
    
    normalizer = DataNormalizer(variables)
    stats_path = os.path.join(os.path.dirname(__file__), "normalization_stats.json")
    if os.path.exists(stats_path):
        normalizer.load(stats_path)
        
    for split in ['train', 'val', 'test']:
        compute_residuals_for_split(split, variables, era5_dir, era5_land_dir, normalizer, device)

if __name__ == "__main__":
    main()
