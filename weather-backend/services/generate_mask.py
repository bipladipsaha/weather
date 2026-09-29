import json
import numpy as np
from matplotlib.path import Path
import os

# Bounds from serve_downscaled_rainfall.py
# bounds = [[-5, 60], [40, 100]] => lat_min=-5, lat_max=40, lon_min=60, lon_max=100
lat_min, lat_max = -5, 40
lon_min, lon_max = 60, 100
height, width = 900, 800

# Create grid points
lats = np.linspace(lat_max, lat_min, height) # Note: matplotlib imshow plots lat_max at top (row 0)
lons = np.linspace(lon_min, lon_max, width)
LONS, LATS = np.meshgrid(lons, lats)
points = np.vstack((LONS.flatten(), LATS.flatten())).T

# Load geojson
with open('india.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

mask = np.zeros(len(points), dtype=bool)

for feature in data['features']:
    geom = feature['geometry']
    coords = geom['coordinates']
    
    if geom['type'] == 'Polygon':
        polys = [coords]
    elif geom['type'] == 'MultiPolygon':
        polys = coords
    else:
        continue
        
    for poly in polys:
        # poly[0] is the outer boundary
        path = Path(poly[0])
        mask |= path.contains_points(points)

mask_2d = mask.reshape((height, width))
np.save('india_mask.npy', mask_2d)
print("Mask generated successfully.")
