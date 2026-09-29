import os

audit_path = r'c:\Users\bipla\Downloads\Weather\Weather\ai\downscaling\DATASET_AUDIT.md'
with open(audit_path, 'r') as f:
    content = f.read()

parts = content.split('## 5. Normalization, Masking, and Splits')
new_text = """## 5. Normalization, Masking, and Splits
- **Chronological Split**: The dataset is split strictly by time to prevent data leakage. The first 70% is `train`, the next 15% is `val`, and the final 15% is `test`.
- **Masking Strategy**: The ERA5-Land target data contains natural NaNs over the ocean. We explicitly build a boolean `target_mask` (where pixels are finite) and pass this mask throughout the entire pipeline (dataset -> normalizer -> loss function). We DO NOT use `nan_to_num()` to impute fake physical values.
- **Normalization Strategy**: Per-variable standard score (Z-score) normalization.
- **Valid Land Pixels**: Each sample has 180,851 spatial pixels (451x401). Of these, 93,699 pixels are masked as ocean NaNs, leaving 87,152 valid land pixels per channel. The previous report mistakenly aggregated the NaNs across all 4 channels (374,796 total NaNs in a 4-channel tensor).
- **Statistics Constraint**: The normalization mean and standard deviation are calculated **STRICTLY ON THE TRAINING SPLIT ONLY**. Validation and test sets are never leaked into the statistics.
"""

with open(audit_path, 'w') as f:
    f.write(parts[0].rstrip() + '\n\n' + new_text)
