# Dataset Audit: High-Resolution ERA5 Prototype Dataset

> **Note**: This dataset is a prototype for validating the neural downscaling pipeline. It represents a downscaling task from approx 28km (native ERA5) to 11km (native ERA5-Land). It should **not** be called the final 12 km → 5 km dataset.

## 1. Coarse Input (ERA5)
- **Native Resolution**: 0.25° (approx 28 km)
- **Temporal Range**: 2018-01-01 00:00 to 2026-01-31 18:00
- **Spatial Range**: Lat -5.0 to 40.0, Lon 60.0 to 100.0
- **Variables**: tp, u10, d2m, t2m, msl

## 2. Fine Target (ERA5-Land)
- **Native Resolution**: 0.10° (approx 11 km)
- **Temporal Range**: 2018-01-01 00:00 to 2019-01-31 18:00
- **Spatial Range**: Lat -5.0 to 40.0, Lon 60.0 to 100.0
- **Variables**: d2m, t2m, u10, v10, tp

## 3. Paired Dataset Alignment
- **Total Valid Paired Samples (Overlapping Time)**: 248
- **Variables Paired**: tp, u10, d2m, t2m
- **Coarse Tensor Shape**: `torch.Size([4, 1, 181, 161])` (Channels, Time, Height, Width)
- **Fine Tensor Shape**: `torch.Size([4, 1, 451, 401])` (Channels, Time, Height, Width)

## 4. Limitations & Validation
- Time alignment is strict inner-join to avoid fabricating data.
- Variable names must match exactly across the two datasets to be loaded automatically.
- ERA5 and ERA5-Land use the same core atmospheric physics model but at different resolutions, making them an excellent pair for testing super-resolution/diffusion architectures.

## 5. Normalization, Masking, and Splits
- **Chronological Split**: The dataset is split strictly by time to prevent data leakage. The first 70% is `train`, the next 15% is `val`, and the final 15% is `test`.
- **Masking Strategy**: The ERA5-Land target data contains natural NaNs over the ocean. We explicitly build a boolean `target_mask` (where pixels are finite) and pass this mask throughout the entire pipeline (dataset -> normalizer -> loss function). We DO NOT use `nan_to_num()` to impute fake physical values.
- **Normalization Strategy**: Per-variable standard score (Z-score) normalization.
- **Valid Land Pixels**: Each sample has 180,851 spatial pixels (451x401). Of these, 93,699 pixels are masked as ocean NaNs, leaving 87,152 valid land pixels per channel. The previous report mistakenly aggregated the NaNs across all 4 channels (374,796 total NaNs in a 4-channel tensor).
- **Statistics Constraint**: The normalization mean and standard deviation are calculated **STRICTLY ON THE TRAINING SPLIT ONLY**. Validation and test sets are never leaked into the statistics.
