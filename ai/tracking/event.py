from dataclasses import dataclass, field
from typing import List, Tuple
import datetime

@dataclass
class EventOccurrence:
    """Represents an event detected at a specific forecast lead time."""
    lead_time: int              # Hours ahead (e.g. 12, 24, 72)
    timestamp: datetime.datetime
    centroid_lat: float
    centroid_lon: float
    area: float                 # Estimated area (e.g. grid cell count or km^2)
    intensity: float            # e.g., anomaly magnitude or max probability
    probability: float          # Model output probability
    severity: str               # Computed dynamically
    confidence: float           # Confidence level
    affected_grid_cells: List[Tuple[int, int]] # (y, x) or (lat_idx, lon_idx)

class EventTrack:
    """Represents a persistently tracked event across multiple lead times."""
    def __init__(self, event_id: str, event_type: str):
        self.event_id = event_id
        self.event_type = event_type
        self.occurrences: List[EventOccurrence] = []
        
        # Track-level properties
        self.direction = "STATIONARY"
        self.speed = 0.0 # km/h
        self.merged_from = []
        self.split_to = []

    def add_occurrence(self, occurrence: EventOccurrence):
        self.occurrences.append(occurrence)
        self.occurrences.sort(key=lambda o: o.lead_time)
        self._update_kinematics()

    def _update_kinematics(self):
        """Calculates trajectory direction and speed based on the latest points."""
        if len(self.occurrences) < 2:
            self.direction = "STATIONARY"
            self.speed = 0.0
            return
            
        # Simplified speed/direction between first and last for prototype
        start = self.occurrences[0]
        end = self.occurrences[-1]
        
        time_diff_hours = end.lead_time - start.lead_time
        if time_diff_hours == 0:
            return
            
        # Haversine distance for speed (simplified)
        import math
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
        dist_km = haversine(start.centroid_lat, start.centroid_lon, end.centroid_lat, end.centroid_lon)
        self.speed = dist_km / time_diff_hours
        
        # Simple cardinal direction
        dlat = end.centroid_lat - start.centroid_lat
        dlon = end.centroid_lon - start.centroid_lon
        
        if dist_km < 10.0:  # If it moved less than 10km, consider stationary
            self.direction = "STATIONARY"
        else:
            ns = "N" if dlat > 0 else "S" if dlat < 0 else ""
            ew = "E" if dlon > 0 else "W" if dlon < 0 else ""
            self.direction = ns + ew
            
    def get_latest(self) -> EventOccurrence:
        return self.occurrences[-1]
