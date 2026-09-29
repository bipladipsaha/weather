import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset
from normalization import DataNormalizer

def test_reconstruction():
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    variables = ['tp', 'u10', 'd2m', 't2m']
    
    # Load dataset in residual mode with NO normalizers to test raw reconstruction
    dataset = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=1, split="train", target_mode="residual")
    
    sample = dataset[0]
    raw_fine = sample['fine'] # This is actually the residual target because of target_mode="residual"! Wait!
    # If no normalizer is used, sample['fine'] is exactly the residual.
    # We also get sample['baseline']
    # But wait, we need the original fine to verify it!
    # Let's load an absolute dataset too.
    abs_dataset = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=1, split="train", target_mode="absolute")
    abs_sample = abs_dataset[0]
    
    original_fine = abs_sample['fine']
    
    baseline = sample['baseline']
    residual = sample['fine']
    mask = sample['target_mask']
    
    # In paired_dataset.py, we did:
    # fine_tensor = torch.nan_to_num(fine_tensor, nan=0.0)
    # target_tensor = fine_tensor - baseline
    # target_tensor = target_tensor * target_mask
    
    # Let's reconstruct
    reconstructed = baseline + residual
    
    original_fine = torch.nan_to_num(original_fine, nan=0.0)
    
    # Mask out invalid pixels for comparison
    reconstructed = reconstructed * mask
    original_fine = original_fine * mask
    
    diff = (reconstructed - original_fine).abs().max().item()
    print(f"Max reconstruction difference: {diff}")
    if diff < 1e-4:
        print("Reconstruction verified successfully: final = baseline + residual")
    else:
        print("Reconstruction failed!")
        
if __name__ == "__main__":
    test_reconstruction()
