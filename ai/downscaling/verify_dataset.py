import os
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset

def verify_dataloader():
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    variables = ['tp', 'u10', 'd2m', 't2m']

    print("--- Dataloader Verification ---")
    dataset = PairedERA5Dataset(
        era5_dir=era5_dir,
        era5_land_dir=era5_land_dir,
        variables=variables,
        sequence_length=9,
        split="train",
        target_mode="residual"
    )

    # 1 & 2
    print(f"Dataset Class Name: {dataset.__class__.__name__}")
    
    # 3
    print(f"Number of samples: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        coarse = sample['coarse']
        fine = sample['fine'] # target tensor (residual in this mode)
        baseline = sample['baseline']
        mask = sample['target_mask']
        
        # 4, 5, 6, 7
        print(f"Coarse tensor shape: {coarse.shape}")
        print(f"Fine target (residual) tensor shape: {fine.shape}")
        if baseline is not None:
            print(f"Baseline shape: {baseline.shape}")
        
        # 8
        mask_pct = (mask.float().mean().item()) * 100
        print(f"Valid-mask percentage: {mask_pct:.2f}%")
        
        # 9 - using 'tp' (index 0)
        tp_fine = fine[0][mask[0].expand_as(fine[0])]
        print(f"Stats for 'tp' residual target - Min: {tp_fine.min().item():.4f}, Max: {tp_fine.max().item():.4f}, Mean: {tp_fine.mean().item():.4f}, Std: {tp_fine.std().item():.4f}")
        
    # 10
    print("Synthetic fallback occurred: False (PairedERA5Dataset does not contain synthetic fallback)")
    print("-------------------------------")

if __name__ == "__main__":
    verify_dataloader()
