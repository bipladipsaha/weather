import torch
from validation import PhysicsValidator, get_physics_loss

def test_physics_framework():
    print("========================================")
    print("TESTING PHYSICS VALIDATION FRAMEWORK")
    print("========================================")
    print("NOTE: Physics validation framework verified on synthetic fields, ")
    print("not as proof that the weather downscaling model is physically accurate.\n")
    
    validator = PhysicsValidator()
    
    # Generate Synthetic Normal Data (5km grid: 72x62)
    H, W = 72, 62
    
    # 1. Normal Temp (300K)
    temp_valid = torch.ones((1, H, W)) * 300.0
    
    # 2. Impossible Temp (500K)
    temp_invalid = torch.ones((1, H, W)) * 500.0
    
    # 3. Valid Precip
    precip_valid = torch.ones((1, H, W)) * 10.0
    
    # 4. Negative Precip
    precip_invalid = torch.ones((1, H, W)) * -5.0
    
    # 5. Valid Wind
    u_valid = torch.ones((1, H, W)) * 3.0
    v_valid = torch.ones((1, H, W)) * 4.0
    speed_valid = torch.ones((1, H, W)) * 5.0 # sqrt(3^2 + 4^2) = 5
    
    # 6. Inconsistent Wind
    speed_invalid = torch.ones((1, H, W)) * 10.0 # Should fail
    
    # 7. NaN Field
    temp_nan = temp_valid.clone()
    temp_nan[0, 0, 0] = float('nan')
    
    # 8. Suspicious Spike
    temp_spike = temp_valid.clone()
    temp_spike[0, 10, 10] = 340.0 # 40K gradient in 1 cell
    
    # 9. Coarse Extreme Preservation
    # Coarse Max = 320, Downscaled Max = 325 (Preserved)
    coarse_temp = torch.ones((1, 30, 26)) * 300.0
    coarse_temp[0, 15, 15] = 320.0
    
    downscaled_temp_preserved = torch.ones((1, H, W)) * 300.0
    downscaled_temp_preserved[0, 35, 30] = 325.0
    
    # 10. Coarse Extreme Suppressed
    # Downscaled Max = 210 (Suppressed 320, Ratio < 0.7)
    downscaled_temp_suppressed = torch.ones((1, H, W)) * 210.0
    
    # Run Validations
    print("Running Tests...\n")
    
    # Test Normal
    r_normal = validator.validate({"temperature_K": temp_valid})
    assert r_normal.results["range_temperature_K"]["status"] == "PASS"
    print("   [PASS] normal temperature field             PASS")
    
    # Test Impossible
    r_imp = validator.validate({"temperature_K": temp_invalid})
    assert r_imp.results["range_temperature_K"]["status"] == "FAIL"
    print("   [PASS] impossible temperature               FAIL")
    
    # Test Precip
    r_precip = validator.validate({"precipitation_mm": precip_valid})
    assert r_precip.results["non_negative_precipitation"]["status"] == "PASS"
    print("   [PASS] valid precipitation                  PASS")
    
    r_nprecip = validator.validate({"precipitation_mm": precip_invalid})
    assert r_nprecip.results["non_negative_precipitation"]["status"] == "FAIL"
    print("   [PASS] negative precipitation               FAIL")
    
    # Test Wind
    r_wind = validator.validate({"u_wind": u_valid, "v_wind": v_valid, "wind_speed_ms": speed_valid})
    assert r_wind.results["wind_consistency"]["status"] == "PASS"
    print("   [PASS] valid u/v wind                       PASS")
    
    r_bwind = validator.validate({"u_wind": u_valid, "v_wind": v_valid, "wind_speed_ms": speed_invalid})
    assert r_bwind.results["wind_consistency"]["status"] == "FAIL"
    print("   [PASS] inconsistent wind speed              FAIL")
    
    # Test NaN
    r_nan = validator.validate({"temperature_K": temp_nan})
    assert r_nan.results["finite_values_temperature_K"]["status"] == "FAIL"
    print("   [PASS] NaN/Inf input                        FAIL")
    
    # Test Spike
    r_spike = validator.validate({"temperature_K": temp_spike})
    assert r_spike.results["spatial_gradient_temperature_K"]["status"] == "WARN"
    print("   [PASS] suspicious spatial spike             WARN/FAIL")
    
    # Test Extreme Preservation
    r_pres = validator.validate({"temperature_K": downscaled_temp_preserved}, coarse_vars={"temperature_K": coarse_temp})
    assert r_pres.results["extreme_preservation_temperature_K"]["status"] == "PASS"
    print("   [PASS] preserved extreme                    PASS")
    
    r_supp = validator.validate({"temperature_K": downscaled_temp_suppressed}, coarse_vars={"temperature_K": coarse_temp})
    assert r_supp.results["extreme_preservation_temperature_K"]["status"] == "WARN"
    print("   [PASS] suppressed extreme                   WARN/FAIL")
    
    # Physics Loss Test
    loss = get_physics_loss(u_valid, v_valid, speed_invalid)
    assert loss.item() > 0
    print("   [PASS] Physics loss                         IMPLEMENTED")
    
    print("\n========================================")
    print("ALL TESTS COMPLETE")
    print("========================================")

if __name__ == "__main__":
    test_physics_framework()
