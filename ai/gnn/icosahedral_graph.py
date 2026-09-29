import numpy as np
import torch
from scipy.spatial import cKDTree

def latlon_to_cartesian(lat_deg, lon_deg):
    """
    Converts latitude/longitude to 3D Cartesian coordinates on a unit sphere.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)

class IcosahedralGraphConverter:
    """
    Handles mapping between regular lat/lon grids and the icosahedral mesh.
    Uses nearest-neighbor interpolation on the sphere.
    """
    def __init__(self, mesh, lats, lons):
        self.mesh = mesh
        self.lats = lats
        self.lons = lons
        
        # Create grid coordinates
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        self.grid_shape = lon_grid.shape # (30, 26)
        
        # Flatten grid coordinates and convert to Cartesian
        self.grid_coords_3d = latlon_to_cartesian(lat_grid.flatten(), lon_grid.flatten())
        
        # Create KD-Trees for fast nearest-neighbor search
        self.mesh_tree = cKDTree(self.mesh.vertices)
        self.grid_tree = cKDTree(self.grid_coords_3d)
        
        # Precompute mapping indices
        # From Grid -> Mesh: for each mesh vertex, find the nearest grid point
        _, self.mesh_to_grid_idx = self.grid_tree.query(self.mesh.vertices)
        
        # From Mesh -> Grid: for each grid point, find the nearest mesh vertex
        _, self.grid_to_mesh_idx = self.mesh_tree.query(self.grid_coords_3d)

    def grid_to_mesh(self, grid_tensor):
        """
        grid_tensor shape: [B, C, T, Lat, Lon]
        Returns mesh_tensor shape: [B, C, T, N] where N is number of mesh vertices.
        """
        B, C, T, Lat, Lon = grid_tensor.shape
        # Flatten spatial dims: [B, C, T, Lat*Lon]
        flat_grid = grid_tensor.view(B, C, T, -1)
        
        # Gather along spatial dimension using nearest neighbors
        # We want to populate N mesh vertices from Lat*Lon grid points.
        # But wait, grid_to_mesh conceptually means we sample the grid AT the mesh vertices.
        # So for each mesh vertex, we take the value of the nearest grid point.
        idx = torch.tensor(self.mesh_to_grid_idx, dtype=torch.long, device=grid_tensor.device)
        
        # Expand idx for batch, channel, time
        idx = idx.view(1, 1, 1, -1).expand(B, C, T, -1)
        
        mesh_tensor = torch.gather(flat_grid, dim=3, index=idx)
        return mesh_tensor

    def mesh_to_grid(self, mesh_tensor):
        """
        mesh_tensor shape: [B, C, T, N]
        Returns grid_tensor shape: [B, C, T, Lat, Lon]
        """
        B, C, T, N = mesh_tensor.shape
        
        # We sample the mesh AT the grid points.
        # For each grid point, we take the value of the nearest mesh vertex.
        idx = torch.tensor(self.grid_to_mesh_idx, dtype=torch.long, device=mesh_tensor.device)
        idx = idx.view(1, 1, 1, -1).expand(B, C, T, -1)
        
        flat_grid = torch.gather(mesh_tensor, dim=3, index=idx)
        grid_tensor = flat_grid.view(B, C, T, self.grid_shape[0], self.grid_shape[1])
        return grid_tensor
