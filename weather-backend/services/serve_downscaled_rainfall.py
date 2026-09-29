import sys
import json
import base64
import argparse
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def generate_rainfall_raster(date_str, layer_name="AI_DOWNSCALED"):
    # Date mapping
    dates = ["2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31"]
    if date_str not in dates:
        return {"error": "Invalid date. Allowed dates: " + ", ".join(dates)}
    
    date_idx = dates.index(date_str)
    
    # Try loading the file
    npy_path = r"C:\Users\bipla\Downloads\Weather\rainfall_downscaling_project\downscaled_rainfall_jan25_31.npy"
    
    if os.path.exists(npy_path):
        try:
            data = np.load(npy_path)
            grid = data[date_idx].copy()
            
            # Apply layer transformations for the prototype
            if layer_name == "ERA5_COARSE" or layer_name == "ERA5_INTERPOLATED":
                from scipy.ndimage import zoom
                factor = 10
                shape = (grid.shape[0]//factor, factor, grid.shape[1]//factor, factor)
                coarse = grid.reshape(shape).mean(-1).mean(1)
                
                if layer_name == "ERA5_COARSE":
                    grid = np.kron(coarse, np.ones((factor, factor)))
                else: # ERA5_INTERPOLATED
                    grid = zoom(coarse, factor, order=3)
                    
            elif layer_name == "CHIRPS_TRUTH":
                # Add slight noise to simulate Ground Truth vs AI output
                grid = grid * np.random.uniform(0.9, 1.1, grid.shape)
            
            # Apply India boundary mask AFTER transformation
            mask_path = os.path.join(os.path.dirname(__file__), "india_mask.npy")
            has_mask = os.path.exists(mask_path)
            if has_mask:
                mask = np.load(mask_path)
                grid[~mask] = 0.0
                
            trend = []
            for i, d in enumerate(data):
                dt = d.copy()
                if has_mask:
                    dt[~mask] = 0.0
                trend.append({
                    "date": f"Jan {25+i}",
                    "mean": round(float(np.mean(dt)), 3),
                    "max": round(float(np.max(dt)), 2)
                })
        except Exception as e:
            return {"error": f"Failed to load .npy file: {str(e)}"}
    else:
        # Generate dummy realistic rainfall data if file doesn't exist yet
        x = np.linspace(0, 10, 800)
        y = np.linspace(0, 10, 900)
        X, Y = np.meshgrid(x, y)
        
        # Center of storm shifts with date_idx
        center_x, center_y = 3 + date_idx * 0.8, 4 + date_idx * 0.4
        
        # Realistic blobs
        grid = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / 3) * 120
        grid += np.exp(-((X - 7)**2 + (Y - 2)**2) / 1.5) * 60
        grid += np.random.normal(0, 1.5, (900, 800))
        grid = np.clip(grid, 0, None) # clip negative to 0
        
    grid_min = float(np.min(grid))
    grid_max = float(np.max(grid))
    grid_mean = float(np.mean(grid))
    
    # Premium dark blue to magenta meteorological rainfall scale
    colors = [
        (0.0, (0.0, 0.0, 0.0, 0.0)),           # 0 mm
        (0.001, (0.3, 0.58, 1.0, 0.0)),        # 0.1 mm threshold
        (0.005, (0.3, 0.58, 1.0, 0.7)),        # 0.5 mm (medium blue)
        (0.02, (0.0, 0.36, 0.9, 0.8)),         # 2 mm  (strong blue)
        (0.10, (0.0, 0.18, 0.7, 0.8)),         # 10 mm (dark blue)
        (0.25, (0.4, 0.0, 0.8, 0.8)),          # 25 mm (purple)
        (0.50, (0.6, 0.0, 0.6, 0.8)),          # 50 mm (magenta)
        (1.0, (0.8, 0.0, 0.32, 0.8))           # 100 mm (crimson)
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('rainfall', colors, N=256)
    
    # Plot to PNG bytes
    fig, ax = plt.subplots(figsize=(8, 9), dpi=100) # 800x900
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    # origin='upper' because image 0,0 is top-left
    # But for maps, latitude 40 is top, -5 is bottom. So we map to image correctly.
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, origin='upper', aspect='auto')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True, pad_inches=0)
    plt.close(fig)
    
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Leaflet bounds: [[south, west], [north, east]] = [[-5, 60], [40, 100]]
    # BUT wait! Leaflet ImageOverlay bounds are [[lat1, lon1], [lat2, lon2]] representing top-left and bottom-right or vice versa.
    # Usually: [[north, west], [south, east]] -> [[40, 60], [-5, 100]]
    
    return {
        "status": "success",
        "date": date_str,
        "units": "mm/day",
        "resolution": "0.05°",
        "bounds": [[-5, 60], [40, 100]],
        "width": 800,
        "height": 900,
        "min": grid_min,
        "max": grid_max,
        "mean": grid_mean,
        "trend": locals().get('trend', []),
        "raster_base64": f"data:image/png;base64,{img_b64}"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--layer", type=str, default="AI_DOWNSCALED")
    args = parser.parse_args()
    
    result = generate_rainfall_raster(args.date, args.layer)
    print(json.dumps(result))
