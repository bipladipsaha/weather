import numpy as np
import torch
from scipy.spatial import cKDTree

class SphericalGraphBuilder:
    def __init__(self, lats, lons, k_neighbors=6):
        """
        lats: 1D array of latitude points (degrees)
        lons: 1D array of longitude points (degrees)
        k_neighbors: number of neighbors for each node in the k-NN graph
        """
        self.lats = lats
        self.lons = lons
        self.k_neighbors = k_neighbors
        self.num_nodes = len(lats) * len(lons)
        
        # Build the graph
        self.edge_index, self.edge_weight = self._build_spherical_graph()
        
    def _latlon_to_cartesian(self, lat, lon, radius=6371.0):
        """
        Converts lat/lon in degrees to 3D Cartesian coordinates on a sphere.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        x = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)
        
        return x, y, z

    def _build_spherical_graph(self):
        """
        Constructs a graph by projecting the regional grid onto a sphere,
        then connecting nodes using a k-Nearest Neighbors approach.
        """
        # Create meshgrid
        lon_mesh, lat_mesh = np.meshgrid(self.lons, self.lats)
        lat_flat = lat_mesh.flatten()
        lon_flat = lon_mesh.flatten()
        
        # Convert to 3D Cartesian for accurate spherical distance
        x, y, z = self._latlon_to_cartesian(lat_flat, lon_flat)
        coords = np.column_stack((x, y, z))
        
        # Build KD-Tree for efficient neighbor search
        tree = cKDTree(coords)
        
        # Query k nearest neighbors (k+1 because a node is its own nearest neighbor)
        distances, indices = tree.query(coords, k=self.k_neighbors + 1)
        
        source_nodes = []
        target_nodes = []
        edge_weights = []
        
        for i in range(self.num_nodes):
            for j in range(1, self.k_neighbors + 1):  # Skip self-loop (index 0)
                neighbor_idx = indices[i, j]
                dist = distances[i, j]
                
                source_nodes.append(i)
                target_nodes.append(neighbor_idx)
                
                # Convert distance to a similarity weight (e.g., Gaussian RBF)
                weight = np.exp(- (dist ** 2) / (2.0 * 100000.0)) # Scale heuristic
                edge_weights.append(weight)
                
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        return edge_index, edge_weight
        
    def get_graph_tensors(self):
        return self.edge_index, self.edge_weight
