import os
import sys
import unittest
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.nwp_loader import NWPLoader
from ai.data.validate_dataset import DatasetValidator
from ai.climatology.climatology import ClimatologyBuilder

class TestRealDataPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data_tmp'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 1. Create a synthetic DETERMINISTIC NetCDF file
        times = pd.date_range("2026-09-01", periods=10, freq="6h")
        lats = np.linspace(8, 38, 30)
        lons = np.linspace(68, 98, 26)
        
        self.det_ds = xr.Dataset(
            {
                "temperature_2m": (["time", "latitude", "longitude"], np.random.rand(10, 30, 26) * 30 + 273.15),
                "total_precipitation": (["time", "latitude", "longitude"], np.random.rand(10, 30, 26) * 5)
            },
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
            attrs={"model": "Test-Deterministic"}
        )
        self.det_path = os.path.join(self.test_dir, "det.nc")
        self.det_ds.to_netcdf(self.det_path)

        # 2. Create synthetic ENSEMBLE members
        self.ens_paths = []
        for i in range(3):
            ens_ds = xr.Dataset(
                {
                    "temperature_2m": (["time", "latitude", "longitude"], np.random.rand(10, 30, 26) * 30 + 273.15),
                    "total_precipitation": (["time", "latitude", "longitude"], np.random.rand(10, 30, 26) * 5)
                },
                coords={
                    "time": times,
                    "latitude": lats,
                    "longitude": lons,
                },
                attrs={"model": "Test-Ensemble"}
            )
            path = os.path.join(self.test_dir, f"ens_mem_{i}.nc")
            ens_ds.to_netcdf(path)
            self.ens_paths.append(path)
            
        # 3. Create a short climatology archive (2016-2020)
        clim_times = pd.date_range("2016-01-01", "2020-12-31", freq="D")
        self.clim_ds = xr.Dataset(
            {
                "temperature_2m": (["time", "latitude", "longitude"], np.random.rand(len(clim_times), 30, 26) * 30 + 273.15),
            },
            coords={
                "time": clim_times,
                "latitude": lats,
                "longitude": lons,
            }
        )
        self.clim_path = os.path.join(self.test_dir, "clim_2016_2020.nc")
        self.clim_ds.to_netcdf(self.clim_path)

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.det_path): os.remove(self.det_path)
        for p in self.ens_paths:
            if os.path.exists(p): os.remove(p)
        if os.path.exists(self.clim_path): os.remove(self.clim_path)
        if os.path.exists(self.test_dir): os.rmdir(self.test_dir)

    def test_deterministic_load(self):
        loader = NWPLoader()
        ds, meta = loader.load_forecast(self.det_path, is_ensemble=False)
        self.assertEqual(meta["ensemble_members"], 1)
        
        # Should raise error if trying to load as ensemble
        with self.assertRaises(ValueError):
            loader.load_forecast(self.det_path, is_ensemble=True)

    def test_ensemble_load(self):
        loader = NWPLoader()
        ds, meta = loader.load_ensemble_from_files(self.ens_paths)
        self.assertEqual(meta["ensemble_members"], 3)
        self.assertIn("member", ds.dims)
        
        # Test tensor conversion
        tensor = loader.to_tensor(ds, meta)
        # Expected shape: [B=3, C=2 (expected vars found), Lead=10, Lat=30, Lon=26]
        self.assertEqual(tensor.shape, (3, 2, 10, 30, 26))

    def test_validation(self):
        validator = DatasetValidator()
        loader = NWPLoader()
        ds, meta = loader.load_ensemble_from_files(self.ens_paths)
        
        report = validator.validate(ds, is_ensemble=True)
        self.assertEqual(report["status"], "PASS")

        # Test failure on missing variable
        ds_bad = ds.drop_vars("temperature_2m")
        report_bad = validator.validate(ds_bad, is_ensemble=True)
        self.assertEqual(report_bad["status"], "FAIL")
        self.assertTrue(any("temperature_2m" in err for err in report_bad["errors"]))

    def test_climatology(self):
        builder = ClimatologyBuilder(self.test_dir)
        ds = builder.load_archive([self.clim_path])
        
        meta = builder.get_baseline_metadata()
        self.assertEqual(meta["years_available"], 5)
        self.assertEqual(meta["target_years"], 30)
        self.assertIn("PARTIAL", meta["status"])
        
        # Quick stat calculation
        results = builder.calculate_climatology(ds, ["temperature_2m"], quantiles=[0.90])
        self.assertIn("temperature_2m", results)
        self.assertIn("mean", results["temperature_2m"])
        self.assertIn("quantiles", results["temperature_2m"])

if __name__ == '__main__':
    unittest.main()
