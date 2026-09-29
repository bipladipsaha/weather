import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import glob

class PairedERA5Dataset(Dataset):
    """
    PyTorch Dataset for pairing coarse ERA5 (31km) and fine ERA5-Land (9km) data.
    Aligns the datasets temporally and provides paired (coarse, fine) samples.
    """
    def __init__(self, era5_dir, era5_land_dir, variables, time_range=None, 
                 sequence_length=1, split="all", normalizer=None, target_mode="absolute", coarse_normalizer=None):
        self.era5_dir = era5_dir
        self.era5_land_dir = era5_land_dir
        self.variables = variables
        self.sequence_length = sequence_length
        self.split = split
        self.normalizer = normalizer
        self.target_mode = target_mode
        self.coarse_normalizer = coarse_normalizer
        
        self.coarse_ds = self._load_and_filter(era5_dir, time_range)
        self.fine_ds = self._load_and_filter(era5_land_dir, time_range)
        
        if self.coarse_ds is None or self.fine_ds is None:
            self.num_samples = 0
            return
            
        try:
            self.coarse_ds, self.fine_ds = xr.align(
                self.coarse_ds, self.fine_ds, 
                join="inner", 
                exclude=["latitude", "longitude", "lat", "lon"]
            )
        except Exception:
            pass
            
        if 'time' in self.coarse_ds.coords:
            all_times = self.coarse_ds.time.values
            n_times = len(all_times)
            
            # Chronological Split
            if split == "train":
                self.times = all_times[:int(n_times * 0.7)]
            elif split == "val":
                self.times = all_times[int(n_times * 0.7):int(n_times * 0.85)]
            elif split == "test":
                self.times = all_times[int(n_times * 0.85):]
            else:
                self.times = all_times
                
            # Keep only the subset in the datasets
            self.coarse_ds = self.coarse_ds.sel(time=self.times)
            self.fine_ds = self.fine_ds.sel(time=self.times)
            
            self.num_samples = max(0, len(self.times) - self.sequence_length + 1)
        else:
            self.num_samples = 0

    def _load_and_filter(self, data_dir, time_range):
        pattern = os.path.join(data_dir, "*.nc")
        files = glob.glob(pattern)
        if not files: return None
        try:
            ds = xr.open_mfdataset(pattern, engine='netcdf4', combine='by_coords')
            if 'valid_time' in ds.coords and 'time' not in ds.coords:
                ds = ds.rename({'valid_time': 'time'})
            if time_range:
                ds = ds.sel(time=slice(time_range[0], time_range[1]))
            return ds
        except Exception:
            return None

    def __len__(self):
        return getattr(self, 'num_samples', 0)

    def __getitem__(self, idx):
        if not hasattr(self, 'num_samples') or self.num_samples == 0:
            raise RuntimeError("Dataset is empty.")

        time_slice = slice(idx, idx + self.sequence_length)
        coarse_sample = self.coarse_ds.isel(time=time_slice)
        fine_sample = self.fine_ds.isel(time=time_slice)
        
        coarse_tensors, fine_tensors = [], []
        
        for var in self.variables:
            c_val = coarse_sample[var].values if var in coarse_sample else np.zeros_like(coarse_sample[self.variables[0]].values)
            f_val = fine_sample[var].values if var in fine_sample else np.zeros_like(fine_sample[self.variables[0]].values)
            coarse_tensors.append(c_val)
            fine_tensors.append(f_val)
                
        coarse_tensor = torch.tensor(np.stack(coarse_tensors, axis=0), dtype=torch.float32)
        fine_tensor = torch.tensor(np.stack(fine_tensors, axis=0), dtype=torch.float32)
        
        # Build validity mask (True where target data is valid finite number)
        # We assume invalid ocean pixels are represented as NaN in ERA5-Land
        target_mask = torch.isfinite(fine_tensor[0:1]) # [1, T, H, W]
        
        # Replace NaNs with zeros in fine_tensor to avoid NaNs propagating in computation
        fine_tensor = torch.nan_to_num(fine_tensor, nan=0.0)

        # Baseline and residual
        baseline = None
        target_tensor = fine_tensor
        if self.target_mode == "residual":
            import torch.nn.functional as F
            # coarse_tensor is [C, T, H, W] -> permute to [T, C, H, W] for interpolation
            C, T, H, W = coarse_tensor.shape
            _, _, fH, fW = fine_tensor.shape
            coarse_2d = coarse_tensor.permute(1, 0, 2, 3)
            baseline_2d = F.interpolate(coarse_2d, size=(fH, fW), mode='bilinear', align_corners=False)
            baseline = baseline_2d.permute(1, 0, 2, 3) # [C, T, fH, fW]
            target_tensor = fine_tensor - baseline
            # Make sure residual is 0 where mask is invalid
            target_tensor = target_tensor * target_mask
            
        # Normalize coarse tensor
        if self.coarse_normalizer:
            coarse_tensor = self.coarse_normalizer.transform(coarse_tensor.unsqueeze(0), self.variables).squeeze(0)
        elif self.normalizer and self.target_mode == "absolute":
            # Backward compatibility: use same normalizer
            coarse_tensor = self.normalizer.transform(coarse_tensor.unsqueeze(0), self.variables).squeeze(0)
            
        # Normalize target tensor (either absolute or residual)
        if self.normalizer:
            target_tensor = self.normalizer.transform(target_tensor.unsqueeze(0), self.variables).squeeze(0)
            target_tensor = target_tensor * target_mask # Ensure mask remains enforced
            
        # Conditioning mock for now (could be static elevation etc)
        conditioning = torch.zeros(2, coarse_tensor.shape[2], coarse_tensor.shape[3])
        
        out = {
            'coarse': coarse_tensor, 
            'fine': target_tensor,
            'conditioning': conditioning,
            'target_mask': target_mask,
            'metadata': {
                'split': self.split,
                'start_time': str(self.times[idx]),
                'variables': self.variables
            }
        }
        if self.target_mode == "residual":
            out['baseline'] = baseline
        return out

if __name__ == "__main__":
    # Quick test to verify instantiation
    print("Testing PairedERA5Dataset class...")
    # This will fail gracefully if directories don't have data yet
    dummy_dir = os.path.dirname(__file__)
    dataset = PairedERA5Dataset(
        era5_dir=os.path.join(dummy_dir, "raw_era5"),
        era5_land_dir=os.path.join(dummy_dir, "raw_era5_land"),
        variables=["2m_temperature", "total_precipitation"],
        sequence_length=1
    )
