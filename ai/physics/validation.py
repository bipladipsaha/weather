import torch
import logging
try:
    from constraints import PhysicalConstraints
except ImportError:
    from .constraints import PhysicalConstraints

class PhysicsValidationReport:
    def __init__(self):
        self.results = {}
        
    def add_result(self, check_name, status, message):
        self.results[check_name] = {"status": status, "message": message}
        
    def get_overall_status(self):
        if any(res["status"] == "FAIL" for res in self.results.values()):
            return "FAIL"
        if any(res["status"] == "WARN" for res in self.results.values()):
            return "WARN"
        return "PASS"
        
    def print_report(self):
        print("\n--- Physics Validation Report ---")
        for check, data in self.results.items():
            print(f"[{data['status']}] {check}: {data['message']}")
        print(f">>> OVERALL STATUS: {self.get_overall_status()} <<<\n")


class PhysicsValidator:
    def __init__(self):
        self.constraints = PhysicalConstraints()

    def validate(self, downscaled_vars, coarse_vars=None):
        """
        downscaled_vars: Dict of variable names to tensors [B, ..., H, W]
        coarse_vars: Dict of variable names to tensors (for extreme preservation check)
        """
        report = PhysicsValidationReport()
        
        # 1. Finite Values Check
        finite = True
        for name, tensor in downscaled_vars.items():
            if not self.constraints.check_finite(tensor):
                finite = False
                report.add_result(f"finite_values_{name}", "FAIL", f"NaN or Inf found in {name}")
                
        if finite:
            report.add_result("finite_values", "PASS", "All variables finite")

        # 2. Range Checks
        for name, tensor in downscaled_vars.items():
            valid, msg = self.constraints.check_bounds(tensor, name)
            if not valid:
                report.add_result(f"range_{name}", "FAIL", msg)
            else:
                report.add_result(f"range_{name}", "PASS", msg)
                
            # Specifically check non-negative precip
            if "precipitation" in name:
                if (tensor < 0).any():
                    report.add_result("non_negative_precipitation", "FAIL", "Negative precipitation detected")
                else:
                    report.add_result("non_negative_precipitation", "PASS", "Precipitation >= 0")

        # 3. Wind Consistency
        if "u_wind" in downscaled_vars and "v_wind" in downscaled_vars and "wind_speed_ms" in downscaled_vars:
            valid, msg = self.constraints.check_wind_consistency(
                downscaled_vars["u_wind"], 
                downscaled_vars["v_wind"], 
                downscaled_vars["wind_speed_ms"]
            )
            status = "PASS" if valid else "FAIL"
            report.add_result("wind_consistency", status, msg)

        # 4. Spatial Gradient Check
        for name, tensor in downscaled_vars.items():
            status, msg = self.constraints.check_spatial_gradient(tensor, name)
            report.add_result(f"spatial_gradient_{name}", status, msg)

        # 5. Extreme Preservation
        if coarse_vars is not None:
            for name, high_tensor in downscaled_vars.items():
                if name in coarse_vars:
                    coarse_tensor = coarse_vars[name]
                    max_coarse = coarse_tensor.max().item()
                    max_high = high_tensor.max().item()
                    
                    if max_coarse > 0:
                        ratio = max_high / max_coarse
                    else:
                        ratio = 1.0
                        
                    # If the high-res model severely blurs/suppresses the extreme
                    if ratio < 0.7:
                        msg = f"Extreme suppression: coarse max={max_coarse:.2f}, 5km max={max_high:.2f} (ratio {ratio:.2f})"
                        report.add_result(f"extreme_preservation_{name}", "WARN", msg)
                    else:
                        msg = f"Extreme preserved: coarse max={max_coarse:.2f}, 5km max={max_high:.2f} (ratio {ratio:.2f})"
                        report.add_result(f"extreme_preservation_{name}", "PASS", msg)
                        
        return report

def get_physics_loss(u, v, speed):
    """
    Optional differentiable physics loss to enforce u/v/speed consistency during diffusion training.
    """
    derived_speed = torch.sqrt(u**2 + v**2 + 1e-8)
    return torch.nn.functional.mse_loss(derived_speed, speed)
