const express = require('express');
const router = express.Router();
const openMeteoAdapter = require('../services/openMeteoAdapter');

// GET /api/weather/historical
router.get('/historical', async (req, res) => {
  try {
    // Expected query params: lat, lon, start, end
    const params = {
      latitude: req.query.lat,
      longitude: req.query.lon,
      start_date: req.query.start,
      end_date: req.query.end,
      hourly: req.query.hourly || 'temperature_2m,precipitation,wind_speed_10m'
    };
    
    const data = await openMeteoAdapter.getHistoricalReanalysis(params);
    
    res.json({
      success: true,
      data: data
    });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
});

module.exports = router;
