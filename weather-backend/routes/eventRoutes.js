const express = require('express');
const router = express.Router();
const eventService = require('../services/eventService');
const EventModel = require('../models/eventModel');

// GET /api/events
router.get('/', (req, res) => {
  try {
    const events = eventService.getAllEvents();
    res.json({
      success: true,
      count: events.length,
      data: events
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/events (for ingesting Indian historical events)
router.post('/', (req, res) => {
  try {
    const event = new EventModel(req.body);
    const savedEvent = eventService.saveEvent(event);
    res.status(201).json({
      success: true,
      data: savedEvent
    });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
});



// POST /api/db/events/calculate-lead-time
// Calculates actionable lead time and tracks forecast revisions from internal stores
router.post('/calculate-lead-time', (req, res) => {
  try {
    const { event_id, thresholds } = req.body;
    
    if (!event_id || !thresholds) {
      return res.status(400).json({ 
        success: false, 
        error: "Missing required payload: event_id, thresholds" 
      });
    }

    // 1. Get the observation (ground truth event)
    const events = eventService.getAllEvents();
    const eventModel = events.find(e => e.event_id === event_id);
    
    if (!eventModel) {
      return res.status(404).json({ success: false, error: "Event not found" });
    }

    const turf = require('@turf/turf');
    const observationAdapter = require('../services/observationIngestAdapter');
    
    // Attempt to load authoritative ground truth GeoJSON
    let observedFootprint = observationAdapter.getIndependentFootprint(event_id);
    let footprintSource = "Independent Ground Truth GeoJSON";

    if (!observedFootprint) {
      console.warn(`[Validation] No independent footprint found for ${event_id}. Waiting for ground truth ingestion.`);
      footprintSource = "Awaiting Ground Truth (No Polygon)";
    }

    const observation = {
      onset_time: eventModel.start_time,
      peak_time: eventModel.end_time || eventModel.start_time,
      area_km2: observedFootprint ? turf.area(observedFootprint) / 1000000 : null,
      centroid: observedFootprint 
        ? { lat: turf.centroid(observedFootprint).geometry.coordinates[1], lon: turf.centroid(observedFootprint).geometry.coordinates[0] } 
        : { lat: eventModel.latitude, lon: eventModel.longitude },
      footprint: observedFootprint,
      source: observedFootprint ? observedFootprint.properties?.source || "Independent Validation Polygon" : footprintSource
    };

    // 2. Get the forecast timeline
    const forecastStore = require('../services/forecastStore');
    const forecasts = forecastStore.getForecastsForEvent(event_id);

    // If no forecasts exist yet for this event, we can't calculate lead time
    if (!forecasts || forecasts.length === 0) {
      return res.json({
        success: true,
        decision_lead_time_hours: null,
        first_detection_forecast: null,
        forecast_revisions: [],
        observation_truth: {
            onset_time: observation.onset_time,
            source: observation.source,
            data_type: "observation"
        },
        message: "No forecast timeline found for this event."
      });
    }

    // 3. Compute lead time
    const leadTimeEngine = require('../services/leadTimeEngine');
    const result = leadTimeEngine.calculateLeadTime({ observation, forecasts, thresholds });
    
    // --- LIVE FORECAST SIMULATION: Shift dates so the event peaks in the future ---
    const targetFutureOnset = new Date();
    targetFutureOnset.setHours(targetFutureOnset.getHours() + Math.floor(Math.random() * 48) + 48); // 48 to 96 hours in future
    
    if (result.observation_truth && result.observation_truth.onset_time) {
      const originalTime = new Date(result.observation_truth.onset_time).getTime();
      const timeShift = targetFutureOnset.getTime() - originalTime;
      
      result.observation_truth.onset_time = targetFutureOnset.toISOString();
      if (result.first_detection_forecast) {
         result.first_detection_forecast = new Date(new Date(result.first_detection_forecast).getTime() + timeShift).toISOString();
      }
      if (result.forecast_revisions) {
        result.forecast_revisions = result.forecast_revisions.map(r => {
           r.forecast_initialization = new Date(new Date(r.forecast_initialization).getTime() + timeShift).toISOString();
           return r;
        });
      }
    }

    res.json(result);
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;
