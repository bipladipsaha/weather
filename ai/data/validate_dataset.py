import xarray as xr
import numpy as np

class DatasetValidator:
    def __init__(self, expected_vars=None, min_spatial_res_km=25, required_dims=None):
        self.expected_vars = expected_vars or ['temperature_2m', 'total_precipitation']
        self.min_spatial_res_km = min_spatial_res_km
        self.required_dims = required_dims or ['time', 'latitude', 'longitude']
        
    def validate(self, ds, is_ensemble=False, is_climatology=False):
        """
        Validates the dataset against required scientific schemas.
        """
        report = {
            "status": "PASS",
            "errors": [],
            "warnings": []
        }
        
        # 1. Dimensions
        dims = list(ds.dims.keys())
        for r_dim in self.required_dims:
            if r_dim not in dims:
                # check aliases
                aliases = {'latitude': ['lat'], 'longitude': ['lon'], 'time': ['valid_time']}
                found = False
                if r_dim in aliases:
                    for alias in aliases[r_dim]:
                        if alias in dims:
                            found = True
                            break
                if not found:
                    report["errors"].append(f"Missing required dimension: {r_dim}")
                    report["status"] = "FAIL"
                    
        # 2. Variable availability
        for var in self.expected_vars:
            if var not in ds.data_vars:
                report["errors"].append(f"Missing required variable: {var}")
                report["status"] = "FAIL"
                
        # 3. Missing values check
        for var in self.expected_vars:
            if var in ds.data_vars:
                # check a small sample to avoid huge computation
                if ds[var].isnull().sum().values > (ds[var].size * 0.1): # more than 10% missing
                    report["errors"].append(f"Excessive missing values in variable: {var}")
                    report["status"] = "FAIL"
                    
        # 4. Ensemble Check
        if is_ensemble:
            if 'member' not in dims and 'number' not in dims:
                report["errors"].append("Ensemble dataset missing member/number dimension.")
                report["status"] = "FAIL"
                
        # 5. Coordinate ordering and duplicates
        time_dim = 'time' if 'time' in ds.dims else 'valid_time' if 'valid_time' in ds.dims else None
        if time_dim:
            time_vals = ds[time_dim].values
            if len(time_vals) != len(np.unique(time_vals)):
                report["errors"].append("Duplicate timestamps found.")
                report["status"] = "FAIL"
            
            # Check ordering
            if not np.all(np.diff(time_vals).astype(float) > 0):
                report["errors"].append("Time dimension is not strictly monotonically increasing.")
                report["status"] = "FAIL"

        return report
