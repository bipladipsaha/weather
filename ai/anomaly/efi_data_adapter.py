import numpy as np

class EFIDataAdapter:
    """
    Interface for providing empirical CDFs for EFI calculation.
    Separates software correctness from data availability.
    """
    def __init__(self, ensemble_dataset=None, climatology_dataset=None):
        self.ensemble_ds = ensemble_dataset
        self.climatology_ds = climatology_dataset
        
    def check_readiness(self):
        """
        Returns a data-readiness report.
        """
        has_ens = self.ensemble_ds is not None and getattr(self.ensemble_ds, "is_real_nepsg", False)
        has_clim = self.climatology_ds is not None and getattr(self.climatology_ds, "is_real_imdaa", False)
        clim_years = getattr(self.climatology_ds, "years_coverage", 0) if has_clim else 5
        
        ready = has_ens and (has_clim and clim_years >= 30)
        reason = "Ready" if ready else "Real NEPS-G and sufficiently long climatological distribution unavailable"
        
        return {
            "ensemble_available": has_ens,
            "climatology_available": has_clim,
            "climatology_years": clim_years,
            "true_efi_ready": ready,
            "reason": reason
        }

    def get_climatology_samples(self, variable, valid_times):
        """
        Returns a continuous empirical sample from climatology for a given valid time period.
        Expected shape: [sample, lat, lon]
        """
        if not self.check_readiness()["true_efi_ready"]:
            raise ValueError("Cannot extract real climatology samples: True EFI data not available.")
        # Logic to extract historical samples based on valid_times (e.g. +/- 15 days for a 30 year period)
        # return self.climatology_ds.extract(...)
        return None

    def get_ensemble_samples(self, variable, valid_times):
        """
        Returns the ensemble forecast samples.
        Expected shape: [member, lead_time, lat, lon]
        """
        if not self.check_readiness()["true_efi_ready"]:
            raise ValueError("Cannot extract real ensemble samples: True EFI data not available.")
        # return self.ensemble_ds.extract(...)
        return None
