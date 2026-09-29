import numpy as np
from scipy.ndimage import label, center_of_mass
try:
    from event import EventOccurrence, EventTrack
except ImportError:
    from .event import EventOccurrence, EventTrack
import datetime
import uuid

class SpatioTemporalTracker:
    def __init__(self, prob_threshold=0.5, lats=None, lons=None):
        self.prob_threshold = prob_threshold
        # STEA-Net Default Grid
        self.lats = lats if lats is not None else np.linspace(-5, 40, 30)
        self.lons = lons if lons is not None else np.linspace(60, 100, 26)
        
        self.class_names = {
            0: "EXTREME_HEAT",
            1: "EXTREME_COLD",
            2: "EXTREME_RAIN",
            3: "EXTREME_CYCLONE"
        }
        
        self.lead_hours = [0, 12, 24, 36, 48, 60, 72, 96, 120]
        self.active_tracks = []
        self.event_counter = 1
        
    def _generate_event_id(self):
        eid = f"EVT-{self.event_counter:04d}"
        self.event_counter += 1
        return eid
        
    def _calculate_severity(self, prob):
        if prob > 0.90:
            return "EXTREME"
        elif prob > 0.70:
            return "SEVERE"
        else:
            return "MODERATE"
            
    def _detect_spatial_events(self, prob_map_2d, lead_time, class_idx, base_time):
        """Detects isolated blobs of high probability in a 2D map."""
        binary_map = prob_map_2d > self.prob_threshold
        labeled_array, num_features = label(binary_map)
        
        occurrences = []
        for i in range(1, num_features + 1):
            mask = labeled_array == i
            
            # Filter out tiny artifacts (e.g., single pixel noise)
            if mask.sum() < 2:
                continue
                
            coords = np.argwhere(mask)
            y_mean, x_mean = center_of_mass(mask)
            
            # Map to lat/lon
            centroid_lat = float(np.interp(y_mean, range(len(self.lats)), self.lats))
            centroid_lon = float(np.interp(x_mean, range(len(self.lons)), self.lons))
            
            max_prob = float(prob_map_2d[mask].max())
            mean_prob = float(prob_map_2d[mask].mean())
            
            occ = EventOccurrence(
                lead_time=lead_time,
                timestamp=base_time + datetime.timedelta(hours=lead_time),
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                area=float(mask.sum()),
                intensity=max_prob,
                probability=mean_prob,
                severity=self._calculate_severity(max_prob),
                confidence=mean_prob,
                affected_grid_cells=[(int(r), int(c)) for r, c in coords]
            )
            occurrences.append(occ)
            
        return occurrences
        
    def _associate_occurrences(self, new_occurrences, event_type):
        """Associates new occurrences with existing tracks or creates new ones."""
        # Simple tracking: distance-based association
        # In a robust system, we would also use IoU, but for centroids moving, distance works well.
        MAX_DIST = 10.0 # degrees roughly
        
        # Get active tracks for this type
        type_tracks = [t for t in self.active_tracks if t.event_type == event_type]
        
        for occ in new_occurrences:
            best_track = None
            min_dist = float('inf')
            
            for track in type_tracks:
                latest = track.get_latest()
                
                # Cannot associate backwards in time
                if occ.lead_time <= latest.lead_time:
                    continue
                    
                dist = np.sqrt((occ.centroid_lat - latest.centroid_lat)**2 + 
                               (occ.centroid_lon - latest.centroid_lon)**2)
                               
                if dist < min_dist and dist < MAX_DIST:
                    min_dist = dist
                    best_track = track
                    
            if best_track:
                best_track.add_occurrence(occ)
            else:
                # Spawn new track
                new_track = EventTrack(self._generate_event_id(), event_type)
                new_track.add_occurrence(occ)
                self.active_tracks.append(new_track)
                type_tracks.append(new_track)
                
    def track_events(self, probability_tensor, base_time=None):
        """
        probability_tensor: HazardTensor instance (converts to [B, 4, L, Lat, Lon] internally)
        """
        import os
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        try:
            from ai.core.tensor_contract import HazardTensor
            if isinstance(probability_tensor, HazardTensor):
                prob_data = probability_tensor.to_grid()[0] # Take first batch element
            else:
                prob_data = probability_tensor # Fallback for old tests if needed
        except ImportError:
            prob_data = probability_tensor

        if base_time is None:
            base_time = datetime.datetime.now()
            
        self.active_tracks = [] # Reset for single forecast run
        self.event_counter = 1
        
        for class_idx in range(4):
            event_type = self.class_names[class_idx]
            
            # Process sequentially across lead times to build tracks
            for l_idx, lead in enumerate(self.lead_hours):
                if prob_data.shape[0] == 4 and len(prob_data.shape) == 4:
                    prob_map_2d = prob_data[class_idx, l_idx, :, :].cpu().numpy() if hasattr(prob_data, 'cpu') else prob_data[class_idx, l_idx, :, :]
                else:
                    # In case batch dimension wasn't stripped by caller but wasn't a HazardTensor
                    prob_map_2d = prob_data[0, class_idx, l_idx, :, :].cpu().numpy() if hasattr(prob_data, 'cpu') else prob_data[0, class_idx, l_idx, :, :]
                
                occurrences = self._detect_spatial_events(prob_map_2d, lead, class_idx, base_time)
                
                if occurrences:
                    self._associate_occurrences(occurrences, event_type)
                    
        return self.active_tracks
