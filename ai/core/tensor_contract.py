import torch
from ai.gnn.icosahedral_graph import IcosahedralGraphConverter

class HazardTensor:
    """
    Standardizes the canonical hazard output.
    Grid format: [B, hazard_classes, time, latitude, longitude]
    Mesh format: [B, hazard_classes, time, nodes]
    """
    def __init__(self, data: torch.Tensor, is_mesh=False, converter: IcosahedralGraphConverter=None):
        self.data = data
        self.is_mesh = is_mesh
        self.converter = converter
        
        self.validate()
        
    def validate(self):
        shape = self.data.shape
        if self.is_mesh:
            if len(shape) != 4:
                raise ValueError(f"Mesh tensor must be [B, C, T, Nodes], got {shape}")
            if shape[1] != 4:
                raise ValueError(f"Mesh tensor must have 4 hazard classes, got {shape[1]}")
        else:
            if len(shape) != 5:
                raise ValueError(f"Grid tensor must be [B, C, T, Lat, Lon], got {shape}")
            if shape[1] != 4:
                raise ValueError(f"Grid tensor must have 4 hazard classes, got {shape[1]}")

    def to_grid(self) -> torch.Tensor:
        """
        Explicitly converts to geographic grid if currently in mesh format.
        """
        if not self.is_mesh:
            return self.data
            
        if self.converter is None:
            raise ValueError("Cannot convert mesh to grid without IcosahedralGraphConverter")
            
        return self.converter.mesh_to_grid(self.data)
        
    def to_mesh(self) -> torch.Tensor:
        """
        Explicitly converts to mesh format if currently in grid format.
        """
        if self.is_mesh:
            return self.data
            
        if self.converter is None:
            raise ValueError("Cannot convert grid to mesh without IcosahedralGraphConverter")
            
        return self.converter.grid_to_mesh(self.data)
