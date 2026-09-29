from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union

class HazardClass(Enum):
    HEAT = 0
    COLD = 1
    RAIN = 2
    CYCLONE = 3

@dataclass
class CanonicalEvent:
    event_id: str
    hazard_type: str
    lead_time: int
    valid_time: str
    centroid: tuple
    probability: float
    severity: str
    area: float
    trajectory: Optional[str] = "STATIONARY"
    speed: Optional[float] = 0.0
    direction: Optional[str] = "STATIONARY"
    anomaly: Optional[float] = None
    efi_proxy: Optional[float] = None
    physics_status: Optional[str] = None
    downscaling_status: Optional[str] = None
    
    def to_dict(self):
        return {
            "event_id": self.event_id,
            "hazard_type": self.hazard_type,
            "forecast": {
                "lead_time_hours": self.lead_time,
                "valid_time": self.valid_time,
                "latitude": self.centroid[0],
                "longitude": self.centroid[1]
            },
            "severity": self.severity,
            "probability": self.probability,
            "area": self.area,
            "tracking": {
                "trajectory": self.trajectory,
                "speed": self.speed,
                "direction": self.direction
            },
            "anomaly": self.anomaly,
            "efi_proxy": self.efi_proxy,
            "physics": {
                "status": self.physics_status
            },
            "downscaling": {
                "status": self.downscaling_status,
                "method": self.downscaling_status  # For legacy compat
            }
        }
