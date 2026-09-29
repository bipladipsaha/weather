import os
import sys
import torch
import numpy as np
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset

def get_spatial_res(ds):
    lats = ds.latitude.values
    lons = ds.longitude.values
    d_lat = abs(lats[1] - lats[0]) if len(lats) > 1 else 0
    d_lon = abs(lons[1] - lons[0]) if len(lons) > 1 else 0
    return d_lat, d_lon

def format_time(t):
    return str(t)[:16].replace('T', ' ')

def main():
    print("========================================")
    print("REAL DATA PAIRING VALIDATION")
    print("========================================\n")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    era5_dir = os.path.join(base_dir, 'ai', 'data', 'raw_era5')
    era5_land_dir = os.path.join(base_dir, 'ai', 'data', 'raw_era5_land')

    print(f"Loading Coarse ERA5 from: {era5_dir}")
    print(f"Loading Fine ERA5-Land from: {era5_land_dir}\n")

    # 1. Load Raw Datasets
    try:
        coarse_ds = xr.open_mfdataset(os.path.join(era5_dir, "*.nc"), engine='netcdf4', combine='by_coords')
        if 'valid_time' in coarse_ds.coords and 'time' not in coarse_ds.coords:
            coarse_ds = coarse_ds.rename({'valid_time': 'time'})
    except Exception as e:
        raise AssertionError(f"Failed to load Coarse ERA5 data: {e}")

    try:
        fine_ds = xr.open_mfdataset(os.path.join(era5_land_dir, "*.nc"), engine='netcdf4', combine='by_coords')
        if 'valid_time' in fine_ds.coords and 'time' not in fine_ds.coords:
            fine_ds = fine_ds.rename({'valid_time': 'time'})
    except Exception as e:
        raise AssertionError(f"Failed to load Fine ERA5-Land data: {e}")

    # 2. Extract Metadata & Validate
    c_lat_res, c_lon_res = get_spatial_res(coarse_ds)
    f_lat_res, f_lon_res = get_spatial_res(fine_ds)

    print("--- COARSE ERA5 (Native) ---")
    print(f"Dimensions: {coarse_ds.dims}")
    print(f"Spatial Res: d_lat={c_lat_res:.2f}, d_lon={c_lon_res:.2f} (approx {c_lat_res*111:.0f}km)")
    print(f"Lat Range: {coarse_ds.latitude.values.min()} to {coarse_ds.latitude.values.max()}")
    print(f"Lon Range: {coarse_ds.longitude.values.min()} to {coarse_ds.longitude.values.max()}")
    print(f"Time Range: {format_time(coarse_ds.time.values.min())} to {format_time(coarse_ds.time.values.max())}")
    print(f"Variables: {list(coarse_ds.data_vars.keys())}\n")

    print("--- FINE ERA5-LAND (Native) ---")
    print(f"Dimensions: {fine_ds.dims}")
    print(f"Spatial Res: d_lat={f_lat_res:.2f}, d_lon={f_lon_res:.2f} (approx {f_lat_res*111:.0f}km)")
    print(f"Lat Range: {fine_ds.latitude.values.min()} to {fine_ds.latitude.values.max()}")
    print(f"Lon Range: {fine_ds.longitude.values.min()} to {fine_ds.longitude.values.max()}")
    print(f"Time Range: {format_time(fine_ds.time.values.min())} to {format_time(fine_ds.time.values.max())}")
    print(f"Variables: {list(fine_ds.data_vars.keys())}\n")

    # 3. Verify Overlap
    c_times = set(coarse_ds.time.values)
    f_times = set(fine_ds.time.values)
    overlap_times = sorted(list(c_times.intersection(f_times)))

    print(f"Temporal Overlap: {len(overlap_times)} timesteps match exactly.")
    if len(overlap_times) == 0:
        raise AssertionError("CRITICAL ERROR: No overlapping time periods between ERA5 and ERA5-Land.")
    
    overlapping_vars = [v for v in coarse_ds.data_vars if v in fine_ds.data_vars]
    print(f"Variable Overlap: {overlapping_vars}")
    if len(overlapping_vars) == 0:
        raise AssertionError("CRITICAL ERROR: No overlapping variables found (check naming conventions, e.g., t2m vs 2m_temperature).")

    # 4. Use PairedERA5Dataset
    print("\n--- PAIRED DATASET LOADER ---")
    dataset = PairedERA5Dataset(
        era5_dir=era5_dir,
        era5_land_dir=era5_land_dir,
        variables=overlapping_vars, # Only request overlapping variables for the test
        sequence_length=1
    )
    print(f"Total Paired Samples Available: {len(dataset)}")

    if len(dataset) == 0:
        raise AssertionError("Dataset is empty. Paired loader failed to align data.")

    sample = dataset[0]
    coarse_tensor = sample['coarse']
    fine_tensor = sample['fine']
    t_val = sample['time']

    print(f"Sample 0 Timestamp: {format_time(t_val)}")
    print(f"Coarse Tensor Shape: {coarse_tensor.shape} (C, T, H, W)")
    print(f"Fine Tensor Shape:   {fine_tensor.shape} (C, T, H, W)")

    if torch.isnan(coarse_tensor).all() or torch.isnan(fine_tensor).all():
        raise AssertionError("CRITICAL ERROR: Tensors are entirely NaN. Check data values.")

    # 5. Generate Audit Document
    audit_path = os.path.join(base_dir, 'ai', 'downscaling', 'DATASET_AUDIT.md')
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Audit: High-Resolution ERA5 Prototype Dataset\n\n")
        f.write("> **Note**: This dataset is a prototype for validating the neural downscaling pipeline. It represents a downscaling task from approx ")
        f.write(f"{c_lat_res*111:.0f}km (native ERA5) to {f_lat_res*111:.0f}km (native ERA5-Land). ")
        f.write("It should **not** be called the final 12 km → 5 km dataset.\n\n")

        f.write("## 1. Coarse Input (ERA5)\n")
        f.write(f"- **Native Resolution**: {c_lat_res:.2f}° (approx {c_lat_res*111:.0f} km)\n")
        f.write(f"- **Temporal Range**: {format_time(coarse_ds.time.values.min())} to {format_time(coarse_ds.time.values.max())}\n")
        f.write(f"- **Spatial Range**: Lat {coarse_ds.latitude.values.min()} to {coarse_ds.latitude.values.max()}, Lon {coarse_ds.longitude.values.min()} to {coarse_ds.longitude.values.max()}\n")
        f.write(f"- **Variables**: {', '.join(coarse_ds.data_vars.keys())}\n\n")

        f.write("## 2. Fine Target (ERA5-Land)\n")
        f.write(f"- **Native Resolution**: {f_lat_res:.2f}° (approx {f_lat_res*111:.0f} km)\n")
        f.write(f"- **Temporal Range**: {format_time(fine_ds.time.values.min())} to {format_time(fine_ds.time.values.max())}\n")
        f.write(f"- **Spatial Range**: Lat {fine_ds.latitude.values.min()} to {fine_ds.latitude.values.max()}, Lon {fine_ds.longitude.values.min()} to {fine_ds.longitude.values.max()}\n")
        f.write(f"- **Variables**: {', '.join(fine_ds.data_vars.keys())}\n\n")

        f.write("## 3. Paired Dataset Alignment\n")
        f.write(f"- **Total Valid Paired Samples (Overlapping Time)**: {len(overlap_times)}\n")
        f.write(f"- **Variables Paired**: {', '.join(overlapping_vars)}\n")
        f.write(f"- **Coarse Tensor Shape**: `{coarse_tensor.shape}` (Channels, Time, Height, Width)\n")
        f.write(f"- **Fine Tensor Shape**: `{fine_tensor.shape}` (Channels, Time, Height, Width)\n\n")

        f.write("## 4. Limitations & Validation\n")
        f.write("- Time alignment is strict inner-join to avoid fabricating data.\n")
        f.write("- Variable names must match exactly across the two datasets to be loaded automatically.\n")
        f.write("- ERA5 and ERA5-Land use the same core atmospheric physics model but at different resolutions, making them an excellent pair for testing super-resolution/diffusion architectures.\n")

    print(f"\nAudit complete! Wrote details to {audit_path}")
    print("SUCCESS: Real-data pipeline is valid.")

if __name__ == "__main__":
    main()
