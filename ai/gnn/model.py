import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGraphConv(nn.Module):
    """
    A lightweight, pure-PyTorch implementation of a Graph Convolution Layer.
    Avoids heavy dependencies like torch_geometric for easy deployment.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x, adj_matrix):
        """
        x: [B, Leads, Nodes, Channels]
        adj_matrix: Sparse tensor [Nodes, Nodes]
        """
        B, L, N, C = x.shape
        
        # Apply linear transformation: [B, L, N, C] -> [B, L, N, C_out]
        x_transformed = self.weight(x)
        
        # Flatten batch and leads to apply sparse matrix multiplication
        # [B*L, N, C_out] -> [N, B*L*C_out] (for sparse.mm)
        x_flat = x_transformed.view(B * L, N, -1).permute(1, 0, 2).reshape(N, -1)
        
        # Message passing: A * X
        out_flat = torch.sparse.mm(adj_matrix, x_flat)
        
        # Reshape back to [B, L, N, C_out]
        out = out_flat.view(N, B * L, -1).permute(1, 0, 2).view(B, L, N, -1)
        
        return out + self.bias

class SphericalGNN(nn.Module):
    def __init__(self, in_channels=30, num_classes=4, num_leads=9, lat_size=30, lon_size=26):
        super().__init__()
        self.metadata = {
            "model_type": "spherical_knn_gnn",
            "mesh_type": "spherical_knn"
        }
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_leads = num_leads
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.num_nodes = lat_size * lon_size
        
        # GNN Layers
        self.gc1 = SimpleGraphConv(in_channels, 64)
        self.gc2 = SimpleGraphConv(64, 128)
        self.gc3 = SimpleGraphConv(128, 64)
        
        # Temporal convolution to process Lead dimension
        self.temporal_conv = nn.Conv1d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=3, 
            padding=1
        )
        
        # Final classification head
        self.head = nn.Linear(64, num_classes)
        
    def _build_sparse_adj(self, edge_index, edge_weight, device):
        """
        Converts edge_index to a normalized sparse adjacency matrix.
        """
        # Add self-loops
        self_loops = torch.arange(self.num_nodes, device=device)
        loop_index = torch.stack([self_loops, self_loops], dim=0)
        loop_weight = torch.ones(self.num_nodes, device=device)
        
        full_index = torch.cat([edge_index, loop_index], dim=1)
        full_weight = torch.cat([edge_weight, loop_weight], dim=0)
        
        # Create sparse tensor
        adj = torch.sparse_coo_tensor(
            full_index, 
            full_weight, 
            (self.num_nodes, self.num_nodes)
        )
        
        # Normalize adjacency (D^-0.5 A D^-0.5) proxy for prototype
        # In a real GCN, we precisely normalize, but for this prototype, simple A is fine.
        return adj
        
    def forward(self, x, edge_index, edge_weight):
        """
        x: [B, Channels, Leads, Lat, Lon]
        """
        B, C, L, Lat, Lon = x.shape
        device = x.device
        
        # 1. Feature Preprocessing (Reshape to nodes)
        # [B, C, L, Lat, Lon] -> [B, L, Lat*Lon, C]
        x_nodes = x.permute(0, 2, 3, 4, 1).view(B, L, self.num_nodes, C)
        
        # 2. Build Adjacency
        adj = self._build_sparse_adj(edge_index.to(device), edge_weight.to(device), device)
        
        # 3. Spatial Message Passing
        x_g = F.relu(self.gc1(x_nodes, adj))
        x_g = F.relu(self.gc2(x_g, adj))
        x_g = F.relu(self.gc3(x_g, adj)) # [B, L, N, 64]
        
        # 4. Temporal Processing
        # [B, L, N, 64] -> [B*N, 64, L]
        x_temp = x_g.permute(0, 2, 3, 1).reshape(B * self.num_nodes, 64, L)
        x_temp = F.relu(self.temporal_conv(x_temp))
        # Back to [B, L, N, 64]
        x_g = x_temp.view(B, self.num_nodes, 64, L).permute(0, 3, 1, 2)
        
        # 5. Prediction Head
        # [B, L, N, 4]
        out_nodes = self.head(x_g)
        
        # 6. Reconstruct Spatial Grid
        # [B, L, Lat*Lon, 4] -> [B, 4, L, Lat, Lon]
        out_grid = out_nodes.view(B, L, Lat, Lon, 4).permute(0, 4, 1, 2, 3)
        
        # Return probabilities via sigmoid (as multi-label extreme classification)
        return torch.sigmoid(out_grid)

    def predict(self, x, edge_index=None, edge_weight=None):
        import os
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from ai.core.tensor_contract import HazardTensor
        
        if edge_index is None or edge_weight is None:
            import numpy as np
            from ai.gnn.graph_builder import SphericalGraphBuilder
            lats = np.linspace(-5, 40, self.lat_size)
            lons = np.linspace(60, 100, self.lon_size)
            gb = SphericalGraphBuilder(lats, lons)
            edge_index = gb.edge_index
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)
            
        probs = self.forward(x, edge_index, edge_weight)
        return HazardTensor(data=probs, is_mesh=False)
