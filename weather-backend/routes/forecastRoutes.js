const express = require('express');
const router = express.Router();
const steanetAdapter = require('../services/steanetAdapter');
const forecastStore = require('../services/forecastStore');

// POST /api/forecast/ingest
// Ingests a raw STEA-Net prediction, extracts footprint, and stores it in the timeline
router.post('/ingest', (req, res) => {
  try {
    const rawOutput = req.body;
    const normalizedRecord = steanetAdapter.processModelOutput(rawOutput);
    forecastStore.ingestForecast(normalizedRecord);
    
    res.status(201).json({
      success: true,
      message: "Forecast successfully ingested and footprint extracted.",
      data: normalizedRecord
    });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

// GET /api/forecast/:event_id
// Retrieves all historical forecast revisions for a given event ID
router.get('/:event_id', (req, res) => {
  try {
    const { event_id } = req.params;
    const forecasts = forecastStore.getForecastsForEvent(event_id);
    
    res.json({
      success: true,
      count: forecasts.length,
      data: forecasts
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
