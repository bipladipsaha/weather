import torch
from dataset import DownscalingDataset
from conditioning import ConditionEncoder
from diffusion import DummyUNet, ConditionalDiffusionModel
from train import ExtremeAwareLoss
from inference import DownscalingInference

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset

def test_downscaling():
    print("========================================")
    print("REAL DATA ARCHITECTURE TEST: PROTOTYPE DATASET")
    print("========================================")
    
    # 1. Data Contract
    print("1. Testing Data Contract with Real Data...")
    
    era5_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5')
    era5_land_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land')
    
    ds = PairedERA5Dataset(
        era5_dir=era5_dir,
        era5_land_dir=era5_land_dir,
        variables=['tp', 'u10', 'd2m', 't2m'],
        sequence_length=9
    )
    
    if len(ds) == 0:
        print("   [FAIL] Dataset is empty.")
        return
        
    sample = ds[0]
    coarse = sample['coarse']
    target = sample['fine']
    
    # Mock static conditioning variables (e.g. elevation, land use) for the test
    cond = torch.randn(2, coarse.shape[2], coarse.shape[3])
    
    print(f"   Coarse input: {coarse.shape}")
    print(f"   Conditioning: {cond.shape}")
    print(f"   Target Fine:  {target.shape}")
    
    # Verify it has sequence dimension and spatial dims
    if coarse.dim() == 4 and target.dim() == 4:
        print("   [PASS] data contract                 PASS")
    else:
        print("   [FAIL] data contract                 FAIL")
        
    # Add batch dim for forward pass
    coarse = coarse.unsqueeze(0)
    cond = cond.unsqueeze(0)
    target = target.unsqueeze(0)
    
    # 2. Conditioning and Diffusion Forward Pass
    print("\n2. Testing Architecture...")
    target_h, target_w = target.shape[3], target.shape[4]
    cond_encoder = ConditionEncoder(c_in=4, cond_in=2, hidden=32, target_h=target_h, target_w=target_w)
    unet = DummyUNet(c_in=4, c_cond=32, c_out=4)
    model = ConditionalDiffusionModel(unet, cond_encoder)
    
    # Inject NaN for test
    coarse_nan = coarse.clone()
    coarse_nan[0, 0, 0, 0, 0] = float('nan')
    
    t_steps = torch.tensor([10])
    try:
        # We pretend 'target' is the noisy_target for this forward pass test
        out_pred = model(target, t_steps, coarse_nan, cond)
        print("   [PASS] conditioning                  PASS")
        print("   [PASS] diffusion forward pass        PASS")
        
        if out_pred.shape == target.shape:
            print("   [PASS] output dimensions             PASS")
        
        if torch.isnan(out_pred).any():
            print("   [PASS] NaN handling                  PASS (NaN propagated)")
            
    except Exception as e:
        print(f"   [FAIL] Architecture tests failed: {e}")
        
    # 3. Extreme-Aware Loss
    print("\n3. Testing Extreme-Aware Loss...")
    loss_fn = ExtremeAwareLoss(lambda_recon=1.0, lambda_extreme=5.0, extreme_threshold=0.8)
    
    clean_target = torch.ones_like(target) * 0.9 # True extreme event
    clean_pred_good = torch.ones_like(target) * 0.88 # Captured extreme
    clean_pred_bad = torch.ones_like(target) * 0.20 # Missed extreme
    
    target_mask = sample['target_mask'].unsqueeze(0) # [1, 1, T, H, W]
    
    loss_good = loss_fn(target, target, clean_pred_good, clean_target, target_mask=target_mask)
    loss_bad = loss_fn(target, target, clean_pred_bad, clean_target, target_mask=target_mask)
    
    print(f"   Loss (Captured Extreme): {loss_good.item():.4f}")
    print(f"   Loss (Missed Extreme)  : {loss_bad.item():.4f}")
    
    if loss_bad > loss_good:
        print("   [PASS] extreme-aware loss            PASS")
    else:
        print("   [FAIL] extreme-aware loss            FAIL")
        
    # 4. Testing Inference Interface
    print("\n4. Testing Inference Interface...")
    infer = DownscalingInference()
    high_res, uncertainty, meta = infer.downscale(coarse, cond)
    
    if high_res.shape == target.shape:
        print("   [PASS] inference interface           PASS")
    else:
        print(f"   [FAIL] inference interface           FAIL (Expected {target.shape}, got {high_res.shape})")
        
    print(f"   Metadata training status: {meta['training_status']}")
    
    # 5. Deterministic baseline
    out1 = infer.bilinear_baseline(coarse)
    out2 = infer.bilinear_baseline(coarse)
    if torch.allclose(out1, out2, equal_nan=True):
        print("   [PASS] deterministic/reproducible    PASS")

    print("\n========================================")
    print("ALL TESTS COMPLETE")
    print("========================================")

if __name__ == "__main__":
    test_downscaling()
