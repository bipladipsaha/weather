import torch
import json
import os
import sys

from diffusion import SmallUNet, ConditionalDiffusionModel
from conditioning import ConditionEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset

def main():
    print("--- 1. Data Split Verification ---")
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    variables = ['tp', 'u10', 'd2m', 't2m']
    
    train_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="train")
    val_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="val")
    test_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="test")
    
    print(f"Train samples: {len(train_ds)}")
    if len(train_ds) > 0: print(f"Train times: {train_ds.times[0]} to {train_ds.times[-1]}")
    print(f"Val samples: {len(val_ds)}")
    if len(val_ds) > 0: print(f"Val times: {val_ds.times[0]} to {val_ds.times[-1]}")
    print(f"Test samples: {len(test_ds)}")
    if len(test_ds) > 0: print(f"Test times: {test_ds.times[0]} to {test_ds.times[-1]}")
    
    print("\n--- 2. Normalization Stats ---")
    stats_path = "normalization_stats.json"
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            print(json.dumps(stats, indent=2))
    else:
        print("normalization_stats.json not found!")
        
    print("\n--- 9. Parameter Count ---")
    unet = SmallUNet(c_in=4, c_cond=32, c_out=4)
    cond_encoder = ConditionEncoder(c_in=4, cond_in=2, hidden=32)
    model = ConditionalDiffusionModel(unet, cond_encoder)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    
    print("\n--- 10. Deterministic Reproducibility Test ---")
    torch.manual_seed(42)
    B, C, T, H5, W5 = 1, 4, 9, 451, 401
    noisy_target = torch.randn(B, C, T, H5, W5)
    time_steps = torch.tensor([500])
    coarse = torch.randn(B, 4, 9, 181, 161)
    cond = torch.randn(B, 2, 181, 161)
    
    model.eval()
    with torch.no_grad():
        out1 = model(noisy_target, time_steps, coarse, cond)
        out2 = model(noisy_target, time_steps, coarse, cond)
        
        diff = torch.max(torch.abs(out1 - out2)).item()
        print(f"Max difference between two identical runs: {diff}")
        if diff < 1e-6:
            print("Reproducibility check PASSED.")
        else:
            print("Reproducibility check FAILED.")

if __name__ == "__main__":
    main()
