import unittest
import torch
import numpy as np
import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.core.schema import CanonicalEvent
from ai.core.metadata import ForecastMetadata, ModelMetadata
from ai.core.tensor_contract import HazardTensor

from ai.steanet.model import STEANet
from ai.gnn.model import SphericalGNN
from ai.gnn.icosahedron import IcosahedronMesh
from ai.gnn.icosahedral_graph import IcosahedralGraphConverter
from ai.gnn.icosahedral_model import IcosahedralGNN
from ai.tracking.tracker import SpatioTemporalTracker

class TestContract(unittest.TestCase):
    def setUp(self):
        self.B, self.C, self.L, self.Lat, self.Lon = 1, 5, 9, 30, 26
        self.dummy_nwp = torch.randn(self.B, self.C, self.L, self.Lat, self.Lon)

    def test_steanet_compatibility(self):
        model = STEANet(in_channels=self.C)
        self.assertEqual(model.metadata["model_type"], "3d_cnn")
        out = model.predict(self.dummy_nwp)
        self.assertIsInstance(out, HazardTensor)
        self.assertFalse(out.is_mesh)
        self.assertEqual(out.to_grid().shape, (self.B, 4, self.L, self.Lat, self.Lon))

    def test_spherical_gnn_compatibility(self):
        model = SphericalGNN(in_channels=self.C)
        self.assertEqual(model.metadata["model_type"], "spherical_knn_gnn")
        out = model.predict(self.dummy_nwp)
        self.assertIsInstance(out, HazardTensor)
        self.assertFalse(out.is_mesh)

    def test_icosahedral_gnn_compatibility(self):
        mesh = IcosahedronMesh(subdivision_level=1)
        lats = np.linspace(8, 38, self.Lat)
        lons = np.linspace(68, 98, self.Lon)
        converter = IcosahedralGraphConverter(mesh, lats, lons)
        model = IcosahedralGNN(mesh, in_channels=self.C, out_channels=4)
        
        self.assertEqual(model.metadata["model_type"], "icosahedral_gnn")
        
        mesh_input = converter.grid_to_mesh(self.dummy_nwp)
        out = model.predict(mesh_input)
        
        self.assertIsInstance(out, HazardTensor)
        self.assertTrue(out.is_mesh)
        
        # Validate mesh conversion back to grid via tracker logic
        out.converter = converter
        grid_out = out.to_grid()
        self.assertEqual(grid_out.shape, (self.B, 4, self.L, self.Lat, self.Lon))

    def test_tracker_compatibility(self):
        tracker = SpatioTemporalTracker(prob_threshold=0.8)
        
        # Test with direct grid HazardTensor
        grid_tensor = HazardTensor(torch.rand(self.B, 4, self.L, self.Lat, self.Lon), is_mesh=False)
        events_grid = tracker.track_events(grid_tensor)
        self.assertIsInstance(events_grid, list)
        
    def test_metadata_integrity(self):
        meta = ForecastMetadata(
            source="Test", model="Test", initialization_time="2026",
            valid_times=["2026"], lead_times_hours=[0], ensemble_members=1,
            latitude_range=[0, 1], longitude_range=[0, 1], resolution_km=12,
            variables=["temp"], units={"temp": "K"}, mesh_type="icosahedral",
            training_status="NOT_TRAINED", ensemble_status="ensemble", # Conflict
            downscaling_method="bilinear", efi_method="proxy"
        )
        
        with self.assertRaises(ValueError):
            meta.validate_integrity()
            
        meta.ensemble_status = "deterministic"
        setattr(meta, "production_model", True) # Conflict with NOT_TRAINED
        with self.assertRaises(ValueError):
            meta.validate_integrity()

if __name__ == '__main__':
    unittest.main()
