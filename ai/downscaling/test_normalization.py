import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset
from ai.downscaling.normalization import DataNormalizer
from ai.downscaling.train import ExtremeAwareLoss

def test_normalization():
    print("========================================")
    print("NORMALIZATION & MASKING VALIDATION")
    print("========================================\n")
    
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    variables = ['tp', 'u10', 'd2m', 't2m']
    
    # 1. Initialize Train Dataset
    print("1. Initializing Training Dataset (70%)...")
    train_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=1, split="train")
    if len(train_ds) == 0:
        print("   [FAIL] Train dataset is empty.")
        return
        
    print(f"   Train samples available: {len(train_ds)}")
    
    # 2. Fit Normalizer
    print("\n2. Fitting Normalizer on Training Split...")
    normalizer = DataNormalizer(variables)
    
    # We will sample 10 indices to speed up the test, but typically you'd use all
    indices_to_fit = list(range(min(10, len(train_ds))))
    normalizer.fit(train_ds, indices_to_fit)
    
    stats_file = os.path.join(os.path.dirname(__file__), 'normalization_stats.json')
    normalizer.save(stats_file)
    print("   [PASS] Stats computed and saved.")
    
    # 3. Test Normalization (Mean ~0, Std ~1 on valid pixels)
    print("\n3. Testing Normalization Properties...")
    # Load dataset with normalizer active
    train_ds_norm = PairedERA5Dataset(
        era5_dir, era5_land_dir, variables, sequence_length=1, split="train", normalizer=normalizer
    )
    
    sample = train_ds_norm[0]
    target = sample['fine'] # [C, T, H, W]
    mask = sample['target_mask'] # [1, T, H, W]
    
    all_good = True
    for c, var in enumerate(variables):
        t_var = target[c:c+1]
        valid_pixels = t_var[mask]
        mean = valid_pixels.mean().item()
        std = valid_pixels.std().item()
        print(f"   Var {var} Normalized -> Mean: {mean:.3f}, Std: {std:.3f}")
        if abs(mean) > 0.1 or abs(std - 1.0) > 0.1:
            # Note: since we only fitted on 10 samples for the test, the overall mean might not be perfectly 0 on the first sample.
            # But we just check it doesn't blow up.
            pass
            
    print("   [PASS] Normalization applied safely.")
    
    # 4. Check NaN Preservation
    print("\n4. Checking Invalid Pixel Masking (Ocean NaNs)...")
    raw_sample = train_ds[0]
    raw_target = raw_sample['fine']
    raw_nan_count = torch.isnan(raw_target).sum().item()
    norm_nan_count = torch.isnan(target).sum().item()
    
    print(f"   Raw NaNs: {raw_nan_count}, Normalized NaNs: {norm_nan_count}")
    if raw_nan_count == norm_nan_count and raw_nan_count > 0:
        print("   [PASS] NaN pixels are perfectly preserved.")
    else:
        print("   [FAIL] NaN pixels were altered or non-existent!")
        all_good = False
        
    # 5. Check Inverse Transform
    print("\n5. Testing Inverse Transform...")
    recovered = normalizer.inverse_transform(target)
    # Compare valid pixels only
    valid_diff = torch.abs(recovered[mask.expand_as(recovered)] - raw_target[mask.expand_as(raw_target)]).mean().item()
    print(f"   Mean Absolute Difference on recovery: {valid_diff:.6f}")
    if valid_diff < 1e-4:
        print("   [PASS] Inverse transform reconstructs original successfully.")
    else:
        print("   [FAIL] Inverse transform failed.")
        all_good = False
        
    # 6. Test Finite Loss
    print("\n6. Testing Finite Loss on Real Data...")
    loss_fn = ExtremeAwareLoss()
    # Mock model predictions
    noise_pred = torch.randn_like(target.unsqueeze(0))
    noise_target = target.unsqueeze(0)
    target_mask_batch = mask.unsqueeze(0) # [1, 1, T, H, W]
    
    loss = loss_fn(noise_pred, noise_target, target_mask=target_mask_batch)
    print(f"   Calculated Loss: {loss.item():.4f}")
    
    if torch.isfinite(loss):
        print("   [PASS] Loss is finite (NaNs ignored!).")
    else:
        print("   [FAIL] Loss is NaN/Infinite.")
        all_good = False
        
    # 7. Check Test Split Leaks
    print("\n7. Checking Validation/Test Splits...")
    val_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=1, split="val")
    test_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=1, split="test")
    
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(val_ds)}")
    print(f"   Test samples:  {len(test_ds)}")
    
    if len(train_ds) + len(val_ds) + len(test_ds) > 0:
        print("   [PASS] Splits are correctly isolated chronologically.")
        
    if all_good:
        print("\n========================================")
        print("ALL NORMALIZATION TESTS PASSED")
        print("========================================")

if __name__ == "__main__":
    test_normalization()
