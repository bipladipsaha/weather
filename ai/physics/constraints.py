import torch
import numpy as np

class PhysicalConstraints:
    """
    Defines physically defensible limits and relationships for meteorological variables.
    These are not arbitrarily invented, but based on Earth climatological bounds.
    """
    
    # Absolute physical limits (Hard thresholds)
    LIMITS = {
        "temperature_K": {"min": 180.0, "max": 340.0},     # Vostok min to Furnace Creek max + margin
        "precipitation_mm": {"min": 0.0, "max": 2000.0},   # Cherrapunji extreme + margin
        "pressure_hPa": {"min": 850.0, "max": 1090.0},     # Typhoon Tip min to Agata max
        "wind_speed_ms": {"min": 0.0, "max": 120.0}        # Barrow Island max + margin
    }
    
    # Gradient limits (Soft diagnostic thresholds, triggering WARN not FAIL)
    # These are configurable diagnostic flags, not universal physical laws.
    # dx/dy differences in adjacent 5km cells
    GRADIENTS = {
        "temperature_K": 15.0,     # e.g., >15K change over 5km is flagged for review (could be valid fire/front)
        "pressure_hPa": 10.0,      
        "wind_speed_ms": 30.0      
    }
    
    def __init__(self, gradient_thresholds=None):
        if gradient_thresholds:
            self.GRADIENTS.update(gradient_thresholds)

    @staticmethod
    def check_finite(tensor):
        """Returns True if no NaNs or Infs are present."""
        return torch.isfinite(tensor).all().item()

    @staticmethod
    def check_bounds(tensor, var_name):
        """Checks if tensor values fall within physical climatological bounds."""
        if var_name not in PhysicalConstraints.LIMITS:
            return True, "No bounds defined"
            
        limits = PhysicalConstraints.LIMITS[var_name]
        min_valid = (tensor >= limits["min"]).all().item()
        max_valid = (tensor <= limits["max"]).all().item()
        
        if not min_valid or not max_valid:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            msg = f"Values out of bounds for {var_name}. Found [{min_val:.2f}, {max_val:.2f}], allowed [{limits['min']}, {limits['max']}]."
            return False, msg
        return True, "Valid bounds"

    @staticmethod
    def check_wind_consistency(u, v, speed, tolerance=1e-3):
        """
        Verifies speed = sqrt(u^2 + v^2)
        """
        derived_speed = torch.sqrt(u**2 + v**2)
        max_diff = torch.max(torch.abs(speed - derived_speed)).item()
        
        if max_diff > tolerance:
            return False, f"Wind speed inconsistent with U/V components. Max difference: {max_diff:.4f} m/s"
        return True, "Consistent"

    def check_spatial_gradient(self, tensor, var_name):
        """
        Computes adjacent spatial differences and flags suspicious checkerboards or spikes.
        """
        if var_name not in PhysicalConstraints.GRADIENTS:
            return "PASS", "No gradient limit defined"
            
        threshold = self.GRADIENTS.get(var_name, None)
        if threshold is None:
            return "PASS", "No diagnostic gradient limit defined"
        
        # Calculate gradients using simple differences (assuming [B, C, H, W] or [H, W])
        if tensor.dim() >= 2:
            dy = torch.abs(tensor[..., 1:, :] - tensor[..., :-1, :])
            dx = torch.abs(tensor[..., :, 1:] - tensor[..., :, :-1])
            max_grad = max(dy.max().item(), dx.max().item())
            
            if max_grad > threshold:
                return "WARN", f"Diagnostic flag: Spatial gradient {max_grad:.2f} exceeds threshold {threshold} (Requires review)"
                
        return "PASS", "Gradients stable"
