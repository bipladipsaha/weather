import numpy as np
import xarray as xr

class ExtremeAnomalyDetector:
    def __init__(self, climatology_engine):
        """
        climatology_engine: Instance of ClimatologyEngine with loaded historical data
        """
        self.clim = climatology_engine
        
    def calculate_standardized_anomaly(self, forecast_da, variable, valid_times):
        """
        Z-score anomaly: (Forecast - Mean) / StdDev
        """
        clim_matched = self.clim.get_climatology_for_forecast(valid_times)
        
        # Assuming clim_matched has 'mean' and 'std' for the variable
        # Or if the variables are just named the same, we assume it's the mean
        # Let's handle standard WB2 climatology which provides the mean
        
        clim_mean = clim_matched[variable]
        
        # If std isn't provided directly, we might need a fallback. 
        # The user's metadata mentioned surface_std_2016_2020.nc for STEA-Net
        # Here we just assume std is available or mock it for prototype safely
        clim_std = clim_matched.get(f"{variable}_std", xr.ones_like(clim_mean))
        
        anomaly = (forecast_da - clim_mean) / clim_std
        return anomaly
        
    def calculate_threshold_exceedance(self, forecast_ensemble_da, variable, q=0.95):
        """
        P(Forecast > Historical q-th Percentile)
        forecast_ensemble_da: DataArray with a 'member' dimension (e.g. from NEPS-G)
        """
        if 'member' not in forecast_ensemble_da.dims:
            raise ValueError("Threshold exceedance requires an ensemble forecast with a 'member' dimension.")
            
        extreme_threshold = self.clim.get_percentile(variable, q)
        
        # Boolean mask where forecast exceeds historical extreme, preserving NaNs
        exceeds = (forecast_ensemble_da > extreme_threshold).where(forecast_ensemble_da.notnull())
        
        # Probability = mean across ensemble members
        probability = exceeds.mean(dim='member')
        
        return probability
        
    def calculate_efi_proxy(self, forecast_ensemble_da, variable, valid_times):
        """
        Proxy for Extreme Forecast Index (EFI) based on threshold exceedance.
        Used until full 30-year climatological CDFs are available.
        """
        if 'member' not in forecast_ensemble_da.dims:
            raise ValueError("EFI_PROXY calculation requires an ensemble forecast (NEPS-G).")
            
        prob_extreme = self.calculate_threshold_exceedance(forecast_ensemble_da, variable, q=0.95)
        proxy_efi = prob_extreme
        return proxy_efi
        
    def calculate_true_efi(self, forecast_ensemble_da, variable, valid_times):
        """
        Scientifically defensible Extreme Forecast Index (EFI) formulation.
        EFI = (2/pi) * integral_0_1 (F_f(p) - p) / sqrt(p * (1-p)) dp
        """
        raise NotImplementedError("TRUE EFI calculation requires a full 30-year climatological CDF, which is currently NOT AVAILABLE in this prototype repository. Please use calculate_efi_proxy() for now.")
