const express = require('express');
const router = express.Router();
const populationAdapter = require('../services/populationAdapter');
const infrastructureAdapter = require('../services/infrastructureAdapter');
const terrainAdapter = require('../services/terrainAdapter');

// POST /api/exposure/population
// Expects a GeoJSON Feature (Polygon or MultiPolygon) in the body
router.post('/population', async (req, res) => {
  try {
    const geojsonPolygon = req.body;
    
    if (!geojsonPolygon || !geojsonPolygon.geometry) {
      return res.status(400).json({ success: false, error: "Invalid GeoJSON provided" });
    }

    const exposureResult = await populationAdapter.calculateExposure(geojsonPolygon);
    
    res.json({
      success: true,
      event_id: req.query.event_id || "unknown", // Optional tracking
      ...exposureResult
    });
  } catch (error) {
    res.json({
      success: false,
      error: error.message,
      population_exposed: null,
      source: "WorldPop",
      calculation_method: "polygon_raster_intersection"
    });
  }
});

// POST /api/exposure/infrastructure
// Expects a GeoJSON Feature (Polygon or MultiPolygon) in the body
router.post('/infrastructure', async (req, res) => {
  try {
    const geojsonPolygon = req.body;
    
    if (!geojsonPolygon || !geojsonPolygon.geometry) {
      return res.status(400).json({ success: false, error: "Invalid GeoJSON provided" });
    }

    const exposureResult = await infrastructureAdapter.calculateExposure(geojsonPolygon);
    
    // Merge event_id into the response
    res.json({
      ...exposureResult,
      event_id: req.query.event_id || req.body.event_id || "unknown"
    });
  } catch (error) {
    res.json({
      success: false,
      error: error.message,
      source: "OpenStreetMap",
      calculation_method: "spatial_intersection"
    });
  }
});

// POST /api/exposure/terrain
// Expects a GeoJSON Feature (Polygon or MultiPolygon) in the body
router.post('/terrain', async (req, res) => {
  try {
    const geojsonPolygon = req.body;
    
    if (!geojsonPolygon || !geojsonPolygon.geometry) {
      return res.status(400).json({ success: false, error: "Invalid GeoJSON provided" });
    }

    const exposureResult = await terrainAdapter.calculateTerrain(geojsonPolygon);
    
    res.json({
      ...exposureResult,
      event_id: req.query.event_id || req.body.event_id || "unknown"
    });
  } catch (error) {
    res.json({
      success: false,
      error: error.message,
      source: "Open-Meteo",
      dataset: "Copernicus DEM (90m)",
      calculation_method: "dem_point_grid_sampling"
    });
  }
});

// POST /api/exposure/hazard-impact
// Assembles all evidence layers for a hazard event
// Body: { hazard_type, event_id, polygon: GeoJSON Feature, event_metadata?: {...} }
router.post('/hazard-impact', async (req, res) => {
  try {
    const { hazard_type, event_id, polygon, event_metadata } = req.body;
    
    if (!hazard_type || !polygon || !polygon.geometry) {
      return res.status(400).json({ 
        success: false, 
        error: "Required: hazard_type (rain|heat|cold|cyclone) and polygon (GeoJSON Feature)" 
      });
    }

    const hazardImpactEngine = require('../services/hazardImpactEngine');
    const result = await hazardImpactEngine.assembleEvidence({
      hazard_type,
      event_id: event_id || "unknown",
      polygon,
      event_metadata: event_metadata || {}
    });
    
    res.json(result);
  } catch (error) {
    res.json({
      success: false,
      error: error.message,
      methodology: "Evidence-based hazard impact assessment"
    });
  }
});

module.exports = router;
