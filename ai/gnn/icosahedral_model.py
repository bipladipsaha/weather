import torch
import torch.nn as nn

class IcosahedralGNN(nn.Module):
    """
    Genuine Icosahedral Spherical GNN using message passing along mesh edges.
    """
    def __init__(self, mesh, in_channels, out_channels, hidden_channels=32, num_layers=3):
        super(IcosahedralGNN, self).__init__()
        self.metadata = {
            "model_type": "icosahedral_gnn",
            "mesh_type": "icosahedral"
        }
        self.mesh = mesh
        
        # Build adjacency matrix for message passing
        # Source to destination edges
        src = []
        dst = []
        for u, v in self.mesh.edges:
            src.extend([u, v])
            dst.extend([v, u])
            
        # Include self-loops
        for i in range(len(self.mesh.vertices)):
            src.append(i)
            dst.append(i)
            
        self.register_buffer('edge_index', torch.tensor([src, dst], dtype=torch.long))
        
        # Simple GNN model
        self.layers = nn.ModuleList()
        self.layers.append(SimpleMeshConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(SimpleMeshConv(hidden_channels, hidden_channels))
        self.layers.append(SimpleMeshConv(hidden_channels, out_channels))
        
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, T, N]
        """
        B, C, T, N = x.shape
        
        # Merge B and T into batch dimension for spatial GNN processing
        x = x.permute(0, 2, 3, 1).contiguous() # [B, T, N, C]
        x = x.view(-1, N, C) # [B*T, N, C]
        
        for i, layer in enumerate(self.layers):
            x = layer(x, self.edge_index)
            if i < len(self.layers) - 1:
                x = self.act(x)
                
        x = self.out_act(x) # [B*T, N, OutC]
        
        # Reshape back to [B, OutC, T, N]
        x = x.view(B, T, N, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def predict(self, x):
        import os
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from ai.core.tensor_contract import HazardTensor
        
        probs = self.forward(x)
        return HazardTensor(data=probs, is_mesh=True)

class SimpleMeshConv(nn.Module):
    """
    A basic graph convolution layer adapted for batch processing over a fixed mesh.
    """
    def __init__(self, in_channels, out_channels):
        super(SimpleMeshConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        """
        x: [B_total, N, C_in]
        edge_index: [2, E]
        """
        B_total, N, C_in = x.shape
        src, dst = edge_index
        
        # Gather source node features
        # x_src: [B_total, E, C_in]
        x_src = x[:, src, :]
        
        # Aggregate messages at destination nodes using scatter_add
        # We need to reshape dst for scatter_add
        out = torch.zeros((B_total, N, C_in), dtype=x.dtype, device=x.device)
        dst_expanded = dst.view(1, -1, 1).expand(B_total, -1, C_in)
        out.scatter_add_(1, dst_expanded, x_src)
        
        # Calculate degree for normalization
        ones = torch.ones((1, len(src), 1), dtype=x.dtype, device=x.device)
        deg = torch.zeros((1, N, 1), dtype=x.dtype, device=x.device)
        deg.scatter_add_(1, dst.view(1, -1, 1), ones)
        deg = deg.clamp(min=1)
        
        out = out / deg
        out = self.linear(out)
        return out
