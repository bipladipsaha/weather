import unittest
import numpy as np
from efi_data_adapter import EFIDataAdapter
from true_efi import TrueEFIEngine

class TestTrueEFI(unittest.TestCase):
    def setUp(self):
        self.adapter = EFIDataAdapter() # Not ready by default
        self.engine = TrueEFIEngine(self.adapter)
        
    def test_case_a_identical_distributions(self):
        # Forecast matches climatology
        c_samp = np.random.normal(20, 5, 1000)
        f_samp = c_samp.copy()
        
        efi = self.engine.calculate_efi(f_samp, c_samp)
        self.assertAlmostEqual(efi, 0.0, places=1)
        
    def test_case_b_warmer_forecast(self):
        c_samp = np.random.normal(20, 5, 1000)
        f_samp = np.random.normal(25, 5, 50) # Shifted warmer
        
        efi = self.engine.calculate_efi(f_samp, c_samp)
        self.assertGreater(efi, 0.2) # Positive EFI
        
    def test_case_c_colder_forecast(self):
        c_samp = np.random.normal(20, 5, 1000)
        f_samp = np.random.normal(15, 5, 50) # Shifted colder
        
        efi = self.engine.calculate_efi(f_samp, c_samp)
        self.assertLess(efi, -0.2) # Negative EFI
        
    def test_case_d_precip_above_climatology(self):
        # Precip has zero mass
        c_samp = np.concatenate([np.zeros(500), np.random.exponential(5, 500)])
        f_samp = np.concatenate([np.zeros(10), np.random.exponential(15, 40)]) # Much wetter
        
        efi = self.engine.calculate_efi(f_samp, c_samp, variable="precipitation")
        self.assertGreater(efi, 0.1)
        
    def test_case_e_precip_below_climatology(self):
        c_samp = np.concatenate([np.zeros(500), np.random.exponential(5, 500)])
        f_samp = np.zeros(50) # Completely dry forecast
        
        efi = self.engine.calculate_efi(f_samp, c_samp, variable="precipitation")
        self.assertLess(efi, -0.1)
        
    def test_case_f_nan_handling(self):
        c_samp = np.array([np.nan, np.nan, np.nan])
        f_samp = np.array([20, 21, 22])
        
        efi = self.engine.calculate_efi(f_samp, c_samp)
        self.assertTrue(np.isnan(efi))
        
    def test_metadata_readiness(self):
        meta = self.engine.get_metadata()
        self.assertEqual(meta["efi_method"], "unavailable")
        
        # Mocking real data
        class MockDS:
            pass
        ens_ds = MockDS()
        ens_ds.is_real_nepsg = True
        
        clim_ds = MockDS()
        clim_ds.is_real_imdaa = True
        clim_ds.years_coverage = 30
        
        ready_adapter = EFIDataAdapter(ens_ds, clim_ds)
        ready_engine = TrueEFIEngine(ready_adapter)
        
        meta_ready = ready_engine.get_metadata()
        self.assertEqual(meta_ready["efi_method"], "true")
        self.assertEqual(meta_ready["climatology_status"], "real")

if __name__ == '__main__':
    unittest.main()
