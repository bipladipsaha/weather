const turf = require('@turf/turf');

class TerrainAdapter {
  constructor() {
    this.openMeteoElevationUrl = 'https://api.open-meteo.com/v1/elevation';
  }

  /**
   * Split an array into chunks of a specific size
   */
  _chunkArray(array, size) {
    const result = [];
    for (let i = 0; i < array.length; i += size) {
      result.push(array.slice(i, i + size));
    }
    return result;
  }

  /**
   * Calculate terrain metrics based on a GeoJSON Polygon
   * @param {Object} geojsonPolygon 
   */
  async calculateTerrain(geojsonPolygon) {
    // 1. Calculate Area and Bounding Box
    const areaSqMeters = turf.area(geojsonPolygon);
    const areaKm2 = areaSqMeters / 1000000;
    const bbox = turf.bbox(geojsonPolygon);

    // 2. Generate a Point Grid over the Bounding Box
    const bboxPoly = turf.bboxPolygon(bbox);
    const bboxAreaKm2 = turf.area(bboxPoly) / 1000000;
    
    // Target ~200 points in the bounding box to keep API calls reasonable
    let cellSizeKm = Math.sqrt(bboxAreaKm2 / 200);
    if (cellSizeKm < 0.09) cellSizeKm = 0.09; // Open-Meteo Copernicus DEM is ~90m global
    if (cellSizeKm > 10) cellSizeKm = 10;     // Cap max cell size

    const grid = turf.pointGrid(bbox, cellSizeKm, { units: 'kilometers' });

    // 3. Filter points to only those inside the hazard polygon
    const pointsInside = grid.features.filter(pt => turf.booleanPointInPolygon(pt, geojsonPolygon));
    
    if (pointsInside.length === 0) {
      // If polygon is too small, just use its centroid
      pointsInside.push(turf.centroid(geojsonPolygon));
    }

    // 4. Fetch Elevations from Open-Meteo in batches of 100
    const pointsBatches = this._chunkArray(pointsInside, 100);
    const elevations = [];

    for (let bIdx = 0; bIdx < pointsBatches.length; bIdx++) {
      const batch = pointsBatches[bIdx];
      const lats = batch.map(pt => pt.geometry.coordinates[1].toFixed(5)).join(',');
      const lons = batch.map(pt => pt.geometry.coordinates[0].toFixed(5)).join(',');
      
      const url = `${this.openMeteoElevationUrl}?latitude=${lats}&longitude=${lons}`;
      
      // Retry with exponential backoff on 429
      let response;
      for (let attempt = 0; attempt < 3; attempt++) {
        response = await fetch(url);
        if (response.status === 429) {
          const waitMs = (attempt + 1) * 2000;
          console.log(`[TerrainAdapter] Rate limited, retrying in ${waitMs}ms...`);
          await new Promise(resolve => setTimeout(resolve, waitMs));
          continue;
        }
        break;
      }

      if (!response.ok) {
        throw new Error(`Open-Meteo Elevation API Error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      if (data.elevation && Array.isArray(data.elevation)) {
        elevations.push(...data.elevation);
      } else {
        throw new Error("Invalid response from Elevation API");
      }

      // Small delay between batches to respect rate limits
      if (bIdx < pointsBatches.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }

    // Assign elevations back to points for slope calculation
    pointsInside.forEach((pt, idx) => {
      pt.properties.elevation = elevations[idx];
    });

    // 5. Compute Basic Metrics
    const validElevations = elevations.filter(e => e !== null && !isNaN(e));
    if (validElevations.length === 0) {
      throw new Error("No valid elevation data returned for this area.");
    }

    const minElevation = Math.min(...validElevations);
    const maxElevation = Math.max(...validElevations);
    const meanElevation = validElevations.reduce((a, b) => a + b, 0) / validElevations.length;

    // 6. Compute Slopes
    // For each point, find its nearest neighbor in the grid to approximate local slope
    let slopesDeg = [];
    
    if (pointsInside.length > 1) {
      for (let i = 0; i < pointsInside.length; i++) {
        const pt1 = pointsInside[i];
        
        // Find nearest neighbor (brute force is fine for ~200 points)
        let minD = Infinity;
        let nearestPt = null;
        for (let j = 0; j < pointsInside.length; j++) {
          if (i === j) continue;
          const pt2 = pointsInside[j];
          const d = turf.distance(pt1, pt2, { units: 'kilometers' });
          if (d < minD) {
            minD = d;
            nearestPt = pt2;
          }
        }

        if (nearestPt && minD > 0) {
          const dzMeters = Math.abs(pt1.properties.elevation - nearestPt.properties.elevation);
          const dxMeters = minD * 1000;
          // slope in degrees = atan(dz/dx)
          const slopeRad = Math.atan(dzMeters / dxMeters);
          const slopeDeg = slopeRad * (180 / Math.PI);
          slopesDeg.push(slopeDeg);
        }
      }
    }

    const meanSlope = slopesDeg.length > 0 
      ? slopesDeg.reduce((a, b) => a + b, 0) / slopesDeg.length 
      : 0;
    const maxSlope = slopesDeg.length > 0 
      ? Math.max(...slopesDeg) 
      : 0;

    const samplingResolutionM = Math.round(cellSizeKm * 1000);
    const timestamp = new Date().toISOString();

    return {
      success: true,
      terrain: {
        mean_elevation_m: Math.round(meanElevation * 10) / 10,
        min_elevation_m: Math.round(minElevation * 10) / 10,
        max_elevation_m: Math.round(maxElevation * 10) / 10,
        elevation_range_m: Math.round((maxElevation - minElevation) * 10) / 10,
        mean_slope_deg: Math.round(meanSlope * 100) / 100,
        max_slope_deg: Math.round(maxSlope * 100) / 100,
      },
      slope_methodology: "Approximate: derived from elevation differences between nearest sampled grid points. Not equivalent to a native DEM slope product.",
      sample_points_used: validElevations.length,
      area_analysed_km2: Math.round(areaKm2 * 100) / 100,
      source: "Open-Meteo",
      dataset: "Copernicus DEM",
      native_resolution: "90m",
      sampling_resolution_m: samplingResolutionM,
      calculation_method: "dem_point_grid_sampling",
      computed_at: timestamp
    };
  }
}

module.exports = new TerrainAdapter();
