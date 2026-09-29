import json
import numpy as np
from PIL import Image, ImageDraw
import os

lat_min, lat_max = -5, 40
lon_min, lon_max = 60, 100
height, width = 900, 800

def latlon_to_pixel(lat, lon):
    x = (lon - lon_min) / (lon_max - lon_min) * width
    # origin='upper' means row 0 is top (North, lat_max)
    y = (lat_max - lat) / (lat_max - lat_min) * height
    return (x, y)

with open('india.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

img = Image.new('1', (width, height), 0)
draw = ImageDraw.Draw(img)

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
        boundary = poly[0]
        pixel_coords = [latlon_to_pixel(p[1], p[0]) for p in boundary]
        draw.polygon(pixel_coords, fill=1)

mask_2d = np.array(img, dtype=bool)
np.save('india_mask.npy', mask_2d)
print("Mask generated successfully with PIL.")
