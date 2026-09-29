import logging

logger = logging.getLogger(__name__)

class AnomalyPipeline:
    def __init__(self, efi_calculator, true_efi_engine=None):
        self.efi_calc = efi_calculator
        self.true_efi_engine = true_efi_engine
        
    def detect_extremes(self, forecast_ensemble_ds, variables):
        """
        Runs the full EFI and Anomaly pipeline over a dataset.
        forecast_ensemble_ds: xarray Dataset containing multiple variables with a 'member' dimension.
        """
        results = {}
        
        valid_times = forecast_ensemble_ds['valid_time'].values if 'valid_time' in forecast_ensemble_ds else forecast_ensemble_ds['time'].values
        
        for var in variables:
            logger.info(f"Detecting extremes for {var}...")
            
            # 1. Standardized Anomaly (using ensemble mean)
            ens_mean = forecast_ensemble_ds[var].mean(dim='member')
            z_score = self.efi_calc.calculate_standardized_anomaly(ens_mean, var, valid_times)
            
            # 2. Extreme Probability (e.g. > 95th percentile)
            prob_95 = self.efi_calc.calculate_threshold_exceedance(forecast_ensemble_ds[var], var, q=0.95)
            prob_99 = self.efi_calc.calculate_threshold_exceedance(forecast_ensemble_ds[var], var, q=0.99)
            
            # 3. EFI Proxy and True EFI
            efi_proxy = self.efi_calc.calculate_efi_proxy(forecast_ensemble_ds[var], var, valid_times)
            
            # Check for True EFI
            if self.true_efi_engine is not None:
                meta = self.true_efi_engine.get_metadata()
                if meta["efi_method"] == "true":
                    # For integration, pretend we extract samples here
                    f_samp = self.true_efi_engine.adapter.get_ensemble_samples(var, valid_times)
                    c_samp = self.true_efi_engine.adapter.get_climatology_samples(var, valid_times)
                    true_efi_val = self.true_efi_engine.calculate_efi(f_samp, c_samp, variable=var)
                    
                    results[var] = {
                        'standardized_anomaly': z_score,
                        'prob_extreme_95': prob_95,
                        'prob_extreme_99': prob_99,
                        'efi_proxy': efi_proxy, # Preserve proxy but distinguish it
                        'true_efi': true_efi_val,
                        'metadata': {
                            'climatology_period': self.efi_calc.clim.metadata.get('baseline_period', 'Unknown'),
                            'ensemble_available': 'member' in forecast_ensemble_ds.dims,
                            'efi_method': 'true',
                            'percentile_source': 'empirical/true',
                            'baseline_resolution': self.efi_calc.clim.metadata.get('spatial_grid', 'Unknown')
                        }
                    }
                    continue
            
            # Fallback to proxy
            results[var] = {
                'standardized_anomaly': z_score,
                'prob_extreme_95': prob_95,
                'prob_extreme_99': prob_99,
                'efi_proxy': efi_proxy,
                'true_efi': None,
                'metadata': {
                    'climatology_period': self.efi_calc.clim.metadata.get('baseline_period', 'Unknown'),
                    'ensemble_available': 'member' in forecast_ensemble_ds.dims,
                    'efi_method': 'proxy',
                    'percentile_source': 'empirical/mocked',
                    'baseline_resolution': self.efi_calc.clim.metadata.get('spatial_grid', 'Unknown')
                }
            }
            
        return results
