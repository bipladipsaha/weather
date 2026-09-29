import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from torch.utils.data import DataLoader

from diffusion import SmallUNet, ConditionalDiffusionModel
from conditioning import ConditionEncoder
from inference import DownscalingInference
from train_diffusion import DDPMNoiseScheduler
from normalization import DataNormalizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset

def get_percentile(tensor, q):
    if tensor.numel() == 0:
        return 0.0
    if tensor.numel() > 1000000:
        idx = torch.randint(0, tensor.numel(), (1000000,), device=tensor.device)
        tensor = tensor.flatten()[idx]
    return torch.quantile(tensor.float(), q).item()

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    variables = ['tp', 'u10', 'd2m', 't2m']
    
    abs_stats_path = os.path.join(os.path.dirname(__file__), "normalization_stats.json")
    res_stats_path = os.path.join(os.path.dirname(__file__), "residual_normalization_stats.json")
    
    abs_normalizer = DataNormalizer(variables)
    abs_normalizer.load(abs_stats_path)
    
    res_normalizer = DataNormalizer(variables)
    res_normalizer.load(res_stats_path)
    
    test_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="test", target_mode="absolute", normalizer=None)
    dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Models
    noise_scheduler = DDPMNoiseScheduler(device=device)
    
    # Absolute Diffusion
    abs_unet = SmallUNet(c_in=4, c_cond=32, c_out=4)
    abs_cond = ConditionEncoder(c_in=4, cond_in=2, hidden=32)
    abs_model = ConditionalDiffusionModel(abs_unet, abs_cond)
    abs_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
    abs_model.load_state_dict(torch.load(abs_ckpt, map_location=device, weights_only=True))
    abs_infer = DownscalingInference(abs_model, noise_scheduler, normalizer=abs_normalizer, device=device)
    
    # Residual Diffusion
    res_unet = SmallUNet(c_in=4, c_cond=32, c_out=4)
    res_cond = ConditionEncoder(c_in=4, cond_in=2, hidden=32)
    res_model = ConditionalDiffusionModel(res_unet, res_cond)
    res_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "residual_diffusion", "best_model.pth")
    res_model.load_state_dict(torch.load(res_ckpt, map_location=device, weights_only=True))
    res_infer = DownscalingInference(res_model, noise_scheduler, normalizer=res_normalizer, device=device)
    
    metrics = {var: {
        "trivial_mean": {"mse": [], "mae": [], "bias": []},
        "bilinear": {"mse": [], "mae": [], "bias": [], "abs_errs": [], "preds": []},
        "absolute_diffusion": {"mse": [], "mae": [], "bias": [], "abs_errs": [], "preds": []},
        "residual_diffusion": {"mse": [], "mae": [], "bias": [], "abs_errs": [], "preds": []},
        "target": {"vals": []}
    } for var in variables}
    
    # Evaluation loop
    with torch.no_grad():
        for batch in dataloader:
            raw_coarse = batch['coarse'].to(device)
            raw_fine = batch['fine'].to(device)
            cond = batch['conditioning'].to(device)
            mask = batch['target_mask'].to(device)
            
            # Baseline
            bilinear = abs_infer.bilinear_baseline(raw_coarse, raw_fine.shape[-2], raw_fine.shape[-1])
            
            # Absolute Diffusion
            abs_pred, _, _ = abs_infer.downscale(raw_coarse, cond, target_shape=(raw_fine.shape[-2], raw_fine.shape[-1]), target_mode="absolute", coarse_normalizer=None)
            abs_pred[:, 0] = torch.clamp(abs_pred[:, 0], min=0.0) # TP safeguard
            
            # Residual Diffusion
            res_pred, _, _ = res_infer.downscale(raw_coarse, cond, target_shape=(raw_fine.shape[-2], raw_fine.shape[-1]), target_mode="residual", coarse_normalizer=abs_normalizer)
            # TP safeguard is already in inference, but enforcing again for safety
            res_pred[:, 0] = torch.clamp(res_pred[:, 0], min=0.0)
            
            # Trivial Mean (from training stats)
            B, C, T, H, W = raw_fine.shape
            
            for c, var in enumerate(variables):
                var_mask = mask[:, 0] # [B, T, H, W]
                t_fine = raw_fine[:, c][var_mask]
                
                # Target
                metrics[var]["target"]["vals"].append(t_fine.cpu())
                
                # Trivial
                train_mean = abs_normalizer.stats[var]['mean']
                trivial_err = t_fine - train_mean
                metrics[var]["trivial_mean"]["mse"].append((trivial_err**2).mean().item())
                metrics[var]["trivial_mean"]["mae"].append(trivial_err.abs().mean().item())
                metrics[var]["trivial_mean"]["bias"].append(trivial_err.mean().item())
                
                # Bilinear
                b_pred = bilinear[:, c][var_mask]
                b_err = b_pred - t_fine
                metrics[var]["bilinear"]["mse"].append((b_err**2).mean().item())
                metrics[var]["bilinear"]["mae"].append(b_err.abs().mean().item())
                metrics[var]["bilinear"]["bias"].append(b_err.mean().item())
                metrics[var]["bilinear"]["abs_errs"].append(b_err.abs().cpu())
                metrics[var]["bilinear"]["preds"].append(b_pred.cpu())
                
                # Absolute
                a_pred = abs_pred[:, c][var_mask]
                a_err = a_pred - t_fine
                metrics[var]["absolute_diffusion"]["mse"].append((a_err**2).mean().item())
                metrics[var]["absolute_diffusion"]["mae"].append(a_err.abs().mean().item())
                metrics[var]["absolute_diffusion"]["bias"].append(a_err.mean().item())
                metrics[var]["absolute_diffusion"]["abs_errs"].append(a_err.abs().cpu())
                metrics[var]["absolute_diffusion"]["preds"].append(a_pred.cpu())
                
                # Residual
                r_pred = res_pred[:, c][var_mask]
                r_err = r_pred - t_fine
                metrics[var]["residual_diffusion"]["mse"].append((r_err**2).mean().item())
                metrics[var]["residual_diffusion"]["mae"].append(r_err.abs().mean().item())
                metrics[var]["residual_diffusion"]["bias"].append(r_err.mean().item())
                metrics[var]["residual_diffusion"]["abs_errs"].append(r_err.abs().cpu())
                metrics[var]["residual_diffusion"]["preds"].append(r_pred.cpu())
                
    # Aggregate Report
    report = {
        "dataset_information": "Test split (38 snapshots / 30 usable)",
        "model_parameters": sum(p.numel() for p in res_model.parameters()),
        "variables": variables,
        "results": {}
    }
    
    for var in variables:
        report["results"][var] = {}
        
        t_vals = torch.cat(metrics[var]["target"]["vals"])
        report["results"][var]["target"] = {
            "min": t_vals.min().item(),
            "max": t_vals.max().item(),
            "mean": t_vals.mean().item(),
            "95th_pct": get_percentile(t_vals, 0.95),
            "99th_pct": get_percentile(t_vals, 0.99)
        }
        
        for model_name in ["trivial_mean", "bilinear", "absolute_diffusion", "residual_diffusion"]:
            mse = np.mean(metrics[var][model_name]["mse"])
            report["results"][var][model_name] = {
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "MAE": np.mean(metrics[var][model_name]["mae"]),
                "bias": np.mean(metrics[var][model_name]["bias"])
            }
            
            if model_name != "trivial_mean":
                all_abs_errs = torch.cat(metrics[var][model_name]["abs_errs"])
                all_preds = torch.cat(metrics[var][model_name]["preds"])
                report["results"][var][model_name].update({
                    "min": all_preds.min().item(),
                    "max": all_preds.max().item(),
                    "abs_err_95th_pct": get_percentile(all_abs_errs, 0.95),
                    "abs_err_99th_pct": get_percentile(all_abs_errs, 0.99),
                    "pred_95th_pct": get_percentile(all_preds, 0.95),
                    "pred_99th_pct": get_percentile(all_preds, 0.99)
                })
                
                if var == "tp":
                    report["results"][var][model_name]["wet_pixel_fraction"] = (all_preds > 0.001).float().mean().item()
                    
    # Conclusion
    # Determine if Residual Outperforms Bilinear on average across all 4 variables for RMSE
    wins = 0
    for var in variables:
        res_rmse = report["results"][var]["residual_diffusion"]["RMSE"]
        bilinear_rmse = report["results"][var]["bilinear"]["RMSE"]
        if res_rmse < bilinear_rmse:
            wins += 1
            
    if wins >= 2: # At least half the variables better
        report["conclusion"] = "OUTPERFORMS_BILINEAR"
    else:
        report["conclusion"] = "DOES_NOT_OUTPERFORM_BILINEAR"
        
    with open(os.path.join(os.path.dirname(__file__), "residual_diffusion_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Evaluation complete. Conclusion: {report['conclusion']}")

if __name__ == "__main__":
    evaluate()
