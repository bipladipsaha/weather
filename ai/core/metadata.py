from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ForecastMetadata:
    source: str
    model: str
    initialization_time: str
    valid_times: List[str]
    lead_times_hours: List[int]
    ensemble_members: int
    latitude_range: List[float]
    longitude_range: List[float]
    resolution_km: float
    variables: List[str]
    units: Dict[str, str]
    mesh_type: str
    training_status: str
    ensemble_status: str
    downscaling_method: str
    efi_method: str

    def validate_integrity(self):
        """
        Prevents metadata lies.
        """
        if self.ensemble_members == 1 and self.ensemble_status == "ensemble":
            raise ValueError("Integrity Error: ensemble_members=1 but ensemble_status='ensemble'")
            
        if self.training_status == "NOT_TRAINED" and getattr(self, "production_model", False):
            raise ValueError("Integrity Error: NOT_TRAINED but reported as production_model")
            
        if self.mesh_type == "spherical_knn" and getattr(self, "reported_mesh_type", self.mesh_type) == "icosahedral":
            raise ValueError("Integrity Error: mesh_type='spherical_knn' reported as 'icosahedral'")
            
        if self.downscaling_method == "bilinear" and getattr(self, "reported_downscaling", self.downscaling_method) == "diffusion":
            raise ValueError("Integrity Error: downscaling='bilinear' reported as 'diffusion'")
            
        if self.efi_method == "proxy" and getattr(self, "reported_efi", self.efi_method) == "true_efi":
            raise ValueError("Integrity Error: EFI method='proxy' reported as 'true_efi'")
        
        return True

@dataclass
class ModelMetadata:
    model_type: str
    training_status: str = "NOT_TRAINED"
    
    def __post_init__(self):
        valid_types = ["3d_cnn", "spherical_knn_gnn", "icosahedral_gnn"]
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
