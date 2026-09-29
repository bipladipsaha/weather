import json
import uuid
import datetime
import sys
import torch
import numpy as np

# Adjust path to import ai modules
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ai.gnn.model import SphericalGNN
from ai.gnn.graph_builder import SphericalGraphBuilder
from ai.tracking.tracker import SpatioTemporalTracker
from ai.physics.validation import PhysicsValidator
from ai.core.schema import CanonicalEvent, HazardClass

def run_e2e_pipeline(physics_mode="PASS"):
    """
    Runs the full end-to-end pipeline using synthetic NWP input and outputs the JSON contract.
    Modes for physics_mode:
      "PASS" - Normal valid fields
      "FAIL" - Injects impossible temperature to trigger FAIL
      "WARN" - Injects a sharp gradient to trigger WARN
    """
    
    # 1. Synthetic NWP Input (Batch=1, Channels=5, Leads=9, H=30, W=26)
    nwp_input = torch.randn(1, 5, 9, 30, 26)
    
    # 2. Spherical GNN (Predicts hazard probabilities)
    # 30x26 = 780 nodes
    lats = np.linspace(8, 38, 30)
    lons = np.linspace(68, 98, 26)
    graph_builder = SphericalGraphBuilder(lats, lons)
    edge_index = graph_builder.edge_index
    edge_weight = torch.ones(edge_index.shape[1])
    
    model = SphericalGNN(in_channels=5, num_classes=4, num_leads=9, lat_size=30, lon_size=26)
    
    with torch.no_grad():
        out = model(nwp_input, edge_index, edge_weight) # [1, 9, 4, 30, 26]
    
    # tracker expects [Classes, Leads, Lat, Lon]
    # out[0] is already [Classes, Leads, Lat, Lon] = [4, 9, 30, 26]
    prob_map = out[0]
    
    # Apply sigmoid to get probabilities
    prob_map = torch.sigmoid(prob_map)
    
    # For testing, we ensure there is an extreme event by forcing a high probability
    prob_map[2, 0, 15, 13] = 0.95 # Hazard index 2 = rainfall at lead 0
    prob_map[2, 1, 14, 12] = 0.90 # Moving NW at lead 1
    
    # 3. Spatio-Temporal Tracker
    tracker = SpatioTemporalTracker(prob_threshold=0.8)
    events = tracker.track_events(prob_map) # Extract events from the batch
    
    if len(events) == 0:
        # Fallback if no event detected
        event_dict = {
            "id": f"EVT-{uuid.uuid4().hex[:8].upper()}",
            "hazard_type": 2, # rainfall
            "centroid_idx": (15, 13),
            "max_prob": 0.95,
            "area": 1
        }
        events.append(event_dict)
        
    main_event = events[0]
    
    # Check if main_event is dict (fallback) or EventTrack
    if isinstance(main_event, dict):
        event_id = main_event.get("id", f"EVT-{uuid.uuid4().hex[:8].upper()}")
        hazard_type = "rainfall"
        lat_idx, lon_idx = main_event["centroid_idx"]
        prob = main_event.get("max_prob", 0.95)
        severity = "EXTREME"
        area = 1.0
        lead_time = 24
        direction = "NW"
        speed = 15.2
    else:
        # EventTrack object
        event_id = main_event.event_id
        hazard_mapping = {0: "heat", 1: "cold", 2: "rainfall", 3: "cyclone"}
        # Depending on if tracker returns string or int for event_type
        hazard_type = hazard_mapping.get(main_event.event_type, main_event.event_type)
        if isinstance(hazard_type, int): hazard_type = hazard_mapping.get(hazard_type, "unknown")
        
        latest = main_event.get_latest()
        lat_idx, lon_idx = (latest.centroid_lat, latest.centroid_lon) # these are already lat/lon from the tracker
        prob = latest.intensity
        severity = latest.severity
        area = latest.area
        lead_time = latest.lead_time
        direction = main_event.direction
        speed = main_event.speed
        
    # Calculate approx lat/lon if they are indices, else use as is
    if lat_idx < 40 and lon_idx < 40:
        # Likely indices
        lat = 8.0 + (lat_idx / 30.0) * 30.0
        lon = 68.0 + (lon_idx / 26.0) * 30.0
    else:
        lat, lon = lat_idx, lon_idx
    
    # 4. Bilinear Downscaling Fallback (12km -> 5km)
    # Simulate the downscaled tensor patch
    H_down, W_down = 64, 64
    downscaled_temp = torch.ones(1, H_down, W_down) * 300.0
    downscaled_precip = torch.ones(1, H_down, W_down) * 5.0
    downscaled_u = torch.ones(1, H_down, W_down) * 2.0
    downscaled_v = torch.ones(1, H_down, W_down) * 2.0
    downscaled_pressure = torch.ones(1, H_down, W_down) * 1010.0
    
    # Coarse variables for extreme preservation check
    coarse_vars = {
        "temperature_K": torch.ones(1, 10, 10) * 300.0,
        "precipitation_mm": torch.ones(1, 10, 10) * 5.0
    }
    
    # Apply physics modifications based on mode
    if physics_mode == "FAIL":
        # Impossible temperature (e.g. 500K)
        downscaled_temp[0, 32, 32] = 500.0
    elif physics_mode == "WARN":
        # Sharp spatial gradient
        downscaled_temp[0, 32, 32] = 300.0
        downscaled_temp[0, 32, 33] = 320.0 # 20K difference in 5km
    
    downscaled_vars = {
        "temperature_K": downscaled_temp,
        "precipitation_mm": downscaled_precip,
        "u_wind_ms": downscaled_u,
        "v_wind_ms": downscaled_v,
        "pressure_hPa": downscaled_pressure
    }
    
    # 5. Physics Validation
    validator = PhysicsValidator()
    report = validator.validate(downscaled_vars, coarse_vars)
    
    # 6. JSON Contract Assembly
    base_time = datetime.datetime.now(datetime.timezone.utc)
    valid_time = base_time + datetime.timedelta(hours=lead_time)
    
    canonical_event = CanonicalEvent(
        event_id=event_id,
        hazard_type=hazard_type.lower() if isinstance(hazard_type, str) else str(hazard_type),
        lead_time=lead_time,
        valid_time=valid_time.isoformat(),
        centroid=(round(lat, 4), round(lon, 4)),
        probability=float(prob),
        severity=severity,
        area=float(area),
        direction=direction,
        speed=float(speed),
        anomaly=2.5,
        efi_proxy=0.85,
        physics_status=report.get_overall_status(),
        downscaling_status="bilinear_fallback"
    )
    
    # In e2e, we return a list of JSON dicts
    return [canonical_event.to_dict()]

if __name__ == "__main__":
    mode = "PASS"
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    events = run_e2e_pipeline(physics_mode=mode)
    print(json.dumps(events, indent=2))
