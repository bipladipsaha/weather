import numpy as np

class TrueEFIEngine:
    def __init__(self, data_adapter):
        self.adapter = data_adapter

    def calculate_efi(self, forecast_samples, climatology_samples, variable="temperature"):
        """
        Calculates the True EFI given forecast samples and climatology samples.
        forecast_samples: [member, lead_time, lat, lon] or [member] (for single point tests)
        climatology_samples: [sample, lat, lon] or [sample] (for single point tests)
        Returns: EFI tensor of shape [lead_time, lat, lon] or scalar
        """
        # Add safety to avoid empty samples
        if forecast_samples.size == 0 or climatology_samples.size == 0:
            return np.full(forecast_samples.shape[1:], np.nan)
            
        # Ensure we operate on numpy arrays
        f_samp = np.asarray(forecast_samples)
        c_samp = np.asarray(climatology_samples)
        
        # Flatten spatial/lead dimensions for vectorized computation or loop
        # Since EFI is calculated per grid point and lead time, it's easier to reshape,
        # apply along axis, and reshape back.
        
        # Determine shape:
        if f_samp.ndim == 1:
            # Single point test
            return self._calculate_efi_point(f_samp, c_samp, variable)
            
        # shape: [member, lead, lat, lon]
        num_members, num_leads, lats, lons = f_samp.shape
        out_efi = np.zeros((num_leads, lats, lons))
        
        # Vectorized or looped EFI computation per grid point
        # For simplicity and robustness in numpy, we loop over spatial dims.
        # A fully vectorized approach is possible but complex for precipitation zero-mass logic.
        for l in range(num_leads):
            for i in range(lats):
                for j in range(lons):
                    f = f_samp[:, l, i, j]
                    c = c_samp[:, i, j]
                    out_efi[l, i, j] = self._calculate_efi_point(f, c, variable)
                    
        return out_efi

    def _calculate_efi_point(self, f_samp, c_samp, variable):
        """
        f_samp: 1D array of ensemble members
        c_samp: 1D array of climatology samples
        """
        # Handle NaNs
        f_samp = f_samp[~np.isnan(f_samp)]
        c_samp = c_samp[~np.isnan(c_samp)]
        
        if len(f_samp) == 0 or len(c_samp) < 5:
            return np.nan
            
        # Sort climatology to build CDF
        c_sorted = np.sort(c_samp)
        n = len(c_sorted)
        
        # Calculate p_i = i / n
        p = np.arange(1, n + 1) / n
        
        # Forecast CDF evaluated at climatology points
        # F_f(x) = fraction of forecast members <= x
        # Use searchsorted to find how many forecast members are <= each c_sorted value
        f_sorted = np.sort(f_samp)
        m = len(f_sorted)
        
        F_f = np.searchsorted(f_sorted, c_sorted, side='right') / m
        
        # Precipitation handling: mass at zero
        if variable == "precipitation":
            # If multiple climatology values are exactly 0, they share the same x.
            # In EFI, we typically start integration after the zero mass, or average over the jump.
            # To be mathematically stable, we compute dp = p_i - p_{i-1}
            # For identical values, dp might be effectively merged.
            pass # Standard integral handles identical values if dp is computed correctly

        # EFI Integral: sum_{i=1}^n (p_i - F_f(x_i)) / sqrt(p_i(1-p_i)) * dp
        # However, p_i can be 1, so p_i(1-p_i) can be 0.
        # We must clip p_i to avoid division by zero.
        # And p_i - F_f(x_i) is the difference. Wait, standard formula uses:
        # EFI = (2/pi) * sum_{i=1}^n (p_i - F_f(x_i)) / sqrt(p_i * (1-p_i)) * (1/n)
        
        # Clip p to [eps, 1-eps] to avoid division by zero
        eps = 1e-6
        p_clipped = np.clip(p, eps, 1 - eps)
        
        weight = 1.0 / np.sqrt(p_clipped * (1 - p_clipped))
        
        # The true ECMWF formula:
        # EFI = (2/pi) * sum (p - F_f(p)) / sqrt(p(1-p)) * dp
        # Since p_i = i/n, dp = 1/n
        dp = 1.0 / n
        
        efi = (2.0 / np.pi) * np.sum((p - F_f) * weight) * dp
        
        # Bound between -1 and 1
        return np.clip(efi, -1.0, 1.0)
        
    def get_metadata(self):
        ready_info = self.adapter.check_readiness()
        if ready_info["true_efi_ready"]:
            return {
                "efi_method": "true",
                "ensemble_status": "real",
                "climatology_status": "real"
            }
        else:
            return {
                "efi_method": "unavailable",
                "ensemble_status": "unavailable",
                "climatology_status": "insufficient"
            }
