import json
import uuid
import datetime
import sys

def get_pipeline_status():
    return {
        "status": "healthy",
        "nwp_input_available": False,
        "ensemble_available": False,
        "climatology_available": True,
        "anomaly_method": "proxy_efi",
        "gnn_status": "prototype_spherical_knn",
        "tracker_status": "active",
        "diffusion_training_status": "NOT_TRAINED",
        "physics_validation_status": "active",
        "impact_engine_status": "connected"
    }

def run_pipeline(demo_mode=False):
    """
    Mock integration script that simulates the AI pipeline generating an event.
    Outputs the structured JSON contract expected by the Node.js backend.
    """
    
    # 1. Simulate an event from the tracker
    event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
    base_time = datetime.datetime.now(datetime.timezone.utc)
    valid_time = base_time + datetime.timedelta(hours=24)
    
    # 2. Simulate physics validation output
    physics_status = "PASS"
    physics_checks = {
        "finite_values": "PASS",
        "spatial_gradients": "WARN", # Example of the diagnostic WARN
        "wind_consistency": "PASS"
    }
    
    event = {
        "event_id": event_id,
        "hazard_type": "rainfall", # mapped from EXTREME_RAIN
        "forecast": {
            "lead_time_hours": 24,
            "valid_time": valid_time.isoformat(),
            "latitude": 22.5,
            "longitude": 88.3
        },
        "severity": "EXTREME",
        "probability": 0.88,
        "anomaly": 2.5,
        "efi_proxy": 0.85,
        "tracking": {
            "direction": "NW",
            "speed_kmh": 15.2,
            "trajectory": []
        },
        "downscaling": {
            "source_resolution_km": 12,
            "target_resolution_km": 5,
            "method": "bilinear_fallback",
            "training_status": "NOT_TRAINED"
        },
        "physics": {
            "status": physics_status,
            "checks": physics_checks
        },
        "impact": {}
    }
    
    if demo_mode:
        event["demo_source"] = "demo/live deterministic source"
        
    return [event]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(json.dumps(get_pipeline_status(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        events = run_pipeline(demo_mode=True)
        print(json.dumps(events, indent=2))
    else:
        events = run_pipeline()
        print(json.dumps(events, indent=2))
