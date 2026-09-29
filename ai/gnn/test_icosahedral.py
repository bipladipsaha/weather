import unittest
import torch
import numpy as np
from icosahedron import IcosahedronMesh
from icosahedral_graph import IcosahedralGraphConverter
from icosahedral_model import IcosahedralGNN

class TestIcosahedralGNN(unittest.TestCase):
    def setUp(self):
        self.level = 2 # subdivision level
        self.mesh = IcosahedronMesh(subdivision_level=self.level)
        self.lats = np.linspace(-5, 40, 30)
        self.lons = np.linspace(60, 100, 26)
        self.converter = IcosahedralGraphConverter(self.mesh, self.lats, self.lons)
        self.model = IcosahedralGNN(self.mesh, in_channels=5, out_channels=4)

    def test_mesh_generation(self):
        stats = self.mesh.get_stats()
        self.assertTrue(stats["euler_valid"], f"Euler characteristic V-E+F != 2. Got: {stats['euler_characteristic']}")
        self.assertEqual(stats["level"], self.level)
        
        # Verify normalization (all vertices on unit sphere)
        norms = np.linalg.norm(self.mesh.vertices, axis=1)
        self.assertTrue(np.allclose(norms, 1.0))
        
        # Degree distribution: Base icosahedron vertices have degree 5, others 6
        self.assertGreaterEqual(stats["min_degree"], 5)
        self.assertLessEqual(stats["max_degree"], 6)

    def test_grid_mesh_conversion(self):
        B, C, T = 2, 5, 3
        Lat, Lon = 30, 26
        grid_tensor = torch.randn((B, C, T, Lat, Lon))
        
        # grid -> mesh
        mesh_tensor = self.converter.grid_to_mesh(grid_tensor)
        self.assertEqual(mesh_tensor.shape, (B, C, T, len(self.mesh.vertices)))
        
        # mesh -> grid
        reconstructed = self.converter.mesh_to_grid(mesh_tensor)
        self.assertEqual(reconstructed.shape, (B, C, T, Lat, Lon))
        
        # NaN handling check: inject NaN into the entire first sample/channel/time
        grid_tensor[0, 0, 0, :, :] = float('nan')
        mesh_tensor_nan = self.converter.grid_to_mesh(grid_tensor)
        reconstructed_nan = self.converter.mesh_to_grid(mesh_tensor_nan)
        self.assertTrue(torch.isnan(reconstructed_nan[0, 0, 0, 0, 0]))

    def test_gnn_forward(self):
        B, C, T = 2, 5, 3
        N = len(self.mesh.vertices)
        x_mesh = torch.randn((B, C, T, N))
        
        out_mesh = self.model(x_mesh)
        self.assertEqual(out_mesh.shape, (B, 4, T, N))
        
        # Check metadata
        self.assertEqual(self.model.metadata["mesh_type"], "icosahedral")

    def test_pipeline_integration(self):
        B, C, T = 2, 5, 3
        Lat, Lon = 30, 26
        x_grid = torch.randn((B, C, T, Lat, Lon))
        
        # Full pipeline
        x_mesh = self.converter.grid_to_mesh(x_grid)
        out_mesh = self.model(x_mesh)
        out_grid = self.converter.mesh_to_grid(out_mesh)
        
        self.assertEqual(out_grid.shape, (B, 4, T, Lat, Lon))

if __name__ == '__main__':
    print("========================================")
    print("TESTING ICOSAHEDRAL MESH GNN PIPELINE")
    print("========================================")
    unittest.main(verbosity=2)
