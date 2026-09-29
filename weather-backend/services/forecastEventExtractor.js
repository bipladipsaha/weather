const turf = require('@turf/turf');

class ForecastEventExtractor {
  /**
   * Extracts a GeoJSON footprint from a predicted probability tensor (2D grid).
   * Applies a threshold, runs connected components, and polygonizes the largest component.
   * 
   * @param {Object} tensorData - Object containing the 2D grid and spatial metadata
   * @param {Array<Array<number>>} tensorData.grid - 2D array of probabilities [latIdx][lonIdx]
   * @param {Object} tensorData.bounds - Spatial bounds { minLat, maxLat, minLon, maxLon }
   * @param {number} threshold - Probability threshold (e.g., 0.70)
   * @returns {Object} Extracted event feature with centroid, area, and footprint
   */
  extractEvent(tensorData, threshold) {
    if (!tensorData || !tensorData.grid || !tensorData.bounds) {
      return null;
    }

    const { grid, bounds } = tensorData;
    const numRows = grid.length;
    const numCols = grid[0].length;

    // 1. Thresholding: Create binary mask
    const mask = Array.from({ length: numRows }, () => new Array(numCols).fill(0));
    let hasPixels = false;
    for (let r = 0; r < numRows; r++) {
      for (let c = 0; c < numCols; c++) {
        if (grid[r][c] >= threshold) {
          mask[r][c] = 1;
          hasPixels = true;
        }
      }
    }

    if (!hasPixels) {
      return null;
    }

    // 2. Connected Components (BFS)
    const visited = Array.from({ length: numRows }, () => new Array(numCols).fill(false));
    let components = [];

    const getNeighbors = (r, c) => {
      const neighbors = [];
      // 8-way connectivity
      const dirs = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
      for (const [dr, dc] of dirs) {
        const nr = r + dr, nc = c + dc;
        if (nr >= 0 && nr < numRows && nc >= 0 && nc < numCols && mask[nr][nc] === 1 && !visited[nr][nc]) {
          neighbors.push([nr, nc]);
        }
      }
      return neighbors;
    };

    for (let r = 0; r < numRows; r++) {
      for (let c = 0; c < numCols; c++) {
        if (mask[r][c] === 1 && !visited[r][c]) {
          const component = [];
          const queue = [[r, c]];
          visited[r][c] = true;

          while (queue.length > 0) {
            const [currR, currC] = queue.shift();
            component.push([currR, currC]);
            
            const neighbors = getNeighbors(currR, currC);
            for (const [nr, nc] of neighbors) {
              visited[nr][nc] = true;
              queue.push([nr, nc]);
            }
          }
          components.push(component);
        }
      }
    }

    if (components.length === 0) {
      return null;
    }

    // 3. Largest Component
    components.sort((a, b) => b.length - a.length);
    const largestComponent = components[0];

    // 4. Polygonization
    // For each pixel in the largest component, create a square polygon, then union them.
    const latStep = (bounds.maxLat - bounds.minLat) / numRows;
    const lonStep = (bounds.maxLon - bounds.minLon) / numCols;

    let footprint = null;
    let maxProb = 0;
    let sumProb = 0;

    for (const [r, c] of largestComponent) {
      const prob = grid[r][c];
      maxProb = Math.max(maxProb, prob);
      sumProb += prob;

      // Calculate pixel bounds. Note: assuming grid[0][0] is maxLat, minLon (top-left)
      const pixelMaxLat = bounds.maxLat - (r * latStep);
      const pixelMinLat = bounds.maxLat - ((r + 1) * latStep);
      const pixelMinLon = bounds.minLon + (c * lonStep);
      const pixelMaxLon = bounds.minLon + ((c + 1) * lonStep);

      // Create polygon for this pixel
      const poly = turf.polygon([[
        [pixelMinLon, pixelMinLat],
        [pixelMaxLon, pixelMinLat],
        [pixelMaxLon, pixelMaxLat],
        [pixelMinLon, pixelMaxLat],
        [pixelMinLon, pixelMinLat]
      ]]);

      if (!footprint) {
        footprint = poly;
      } else {
        footprint = turf.union(turf.featureCollection([footprint, poly]));
      }
    }
    
    // Clean up the footprint if union resulted in a GeometryCollection or FeatureCollection
    // turf.union usually returns a Feature<Polygon|MultiPolygon>
    if (!footprint) {
      return null;
    }

    // 5. Centroid and Area
    const centroidFeature = turf.centroid(footprint);
    const [centroidLon, centroidLat] = centroidFeature.geometry.coordinates;
    const areaSqMeters = turf.area(footprint);
    const areaKm2 = areaSqMeters / 1000000;
    const meanProb = sumProb / largestComponent.length;

    return {
      type: "Feature",
      geometry: footprint.geometry,
      properties: {
        probability: maxProb,
        mean_probability: Math.round(meanProb * 1000) / 1000,
        centroid: { 
          lat: Math.round(centroidLat * 10000) / 10000, 
          lon: Math.round(centroidLon * 10000) / 10000 
        },
        area_km2: Math.round(areaKm2 * 10) / 10,
        component_pixels: largestComponent.length
      }
    };
  }
}

module.exports = new ForecastEventExtractor();
