import numpy as np
import urllib.request
import json
from datetime import datetime, timedelta

def main():
    npy_path = r"C:\Users\bipla\Downloads\Weather\Weather\SIH26078_COMPLETE_BACKUP\working\sih26078_diagnostic\test_event_probabilities.npy"
    print(f"Loading {npy_path}...")
    data = np.load(npy_path)
    print(f"Data shape: {data.shape}")

    # Shape is (730, 4, 9, 30, 26) -> (days, hazards, lead_times, lat, lon)
    # Hazards: 0: heat, 1: cold, 2: rain, 3: cyclone
    hazard_idx = 2
    
    # The correct way to get forecast revisions for the SAME valid time:
    # If the valid time is at day index `target_valid_day_idx`, then the forecast initialized
    # at `target_valid_day_idx - k` will have a lead time of `k` days.
    # Assuming the 9 lead times are daily: T-0, T-24, T-48, ..., T-192.
    # We will pick a valid day that has a strong event at T-0.
    target_valid_day_idx = np.argmax(data[:, hazard_idx, 0, :, :].max(axis=(1, 2)))
    print(f"Selected valid day index {target_valid_day_idx} with max prob {data[target_valid_day_idx, hazard_idx, 0].max():.3f}")

    event_id = "EV-2023-HP-FLOOD"
    valid_time = datetime(2023, 7, 8)
    
    # Lead time indices: 0 is T-0, 1 is T-24, ..., 8 is T-192
    for lt_idx in range(9):
        lt_hours = lt_idx * 24
        init_day_idx = target_valid_day_idx - lt_idx
        
        # If the initialization day is out of bounds (before the dataset starts), skip
        if init_day_idx < 0:
            continue
            
        prob_grid = data[init_day_idx, hazard_idx, lt_idx, :, :]
        init_time = valid_time - timedelta(hours=lt_hours)

        payload = {
            "event_id": event_id,
            "initialization_time": init_time.isoformat() + "Z",
            "valid_time": valid_time.isoformat() + "Z",
            "hazard": "rain",
            "tensor_data": {
                "grid": prob_grid.tolist(),
                "bounds": {
                    "minLat": -4.5,
                    "maxLat": 39.0,
                    "minLon": 61.5,
                    "maxLon": 99.0
                }
            }
        }
        
        try:
            req = urllib.request.Request("http://localhost:3001/api/forecast/ingest", data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as res:
                if res.status == 201:
                    print(f"Ingested T-{lt_hours}h forecast successfully.")
                else:
                    print(f"Failed to ingest T-{lt_hours}h: {res.read().decode()}")
        except Exception as e:
            print(f"Error connecting to backend: {e}")

if __name__ == "__main__":
    main()
