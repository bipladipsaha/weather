const express = require('express');
const router = express.Router();
const aiPipelineAdapter = require('../services/aiPipelineAdapter');
const hazardImpactEngine = require('../services/hazardImpactEngine');
const turf = require('@turf/turf');

// Instantiate the engine (assuming it's a class based on usage in view_file)
// Engine is exported as an instance

/**
 * Helper to generate a 5km polygon for impact engine ingestion based on event centroid.
 */
function create5kmPolygon(lat, lon) {
    const center = turf.point([lon, lat]);
    return turf.circle(center, 5, { steps: 64, units: 'kilometers' });
}

/**
 * @route GET /api/ai/health
 * @desc Expose AI Pipeline Status
 */
router.get('/health', async (req, res) => {
    try {
        const status = await aiPipelineAdapter.getHealth();
        res.json({ success: true, status });
    } catch (error) {
        res.status(500).json({ success: false, error: 'AI pipeline unavailable', details: error.message });
    }
});

/**
 * @route GET /api/ai/events
 * @desc Retrieve events from the AI pipeline
 */
router.get('/events', async (req, res) => {
    const isDemo = req.query.demo === 'true';
    try {
        const events = await aiPipelineAdapter.getEvents(isDemo);
        res.json({ success: true, events });
    } catch (error) {
        res.status(500).json({ success: false, error: 'Failed to fetch AI events', details: error.message });
    }
});

/**
 * @route GET /api/ai/events/:eventId
 * @desc Retrieve a specific event (mock behavior for now, just fetches all and filters)
 */
router.get('/events/:eventId', async (req, res) => {
    const isDemo = req.query.demo === 'true';
    try {
        const events = await aiPipelineAdapter.getEvents(isDemo);
        const event = events.find(e => e.event_id === req.params.eventId);
        if (event) {
            res.json({ success: true, event });
        } else {
            res.status(404).json({ success: false, error: 'Event not found' });
        }
    } catch (error) {
        res.status(500).json({ success: false, error: 'Failed to fetch AI event', details: error.message });
    }
});

/**
 * @route GET /api/ai/forecast
 * @desc Route specifically tailored to return the Python AI output representing a full forecast pipeline
 */
router.get('/forecast', async (req, res) => {
    const isDemo = req.query.demo === 'true';
    try {
        const events = await aiPipelineAdapter.getEvents(isDemo);
        res.json({ success: true, forecast_events: events });
    } catch (error) {
        res.status(500).json({ success: false, error: 'Failed to fetch AI forecast', details: error.message });
    }
});

/**
 * @route GET /api/ai/impacts
 * @desc Run AI Pipeline -> feed to Impact Engine -> return combined result
 */
router.get('/impacts', async (req, res) => {
    const isDemo = req.query.demo === 'true';
    try {
        // 1. Get Events from AI
        const events = await aiPipelineAdapter.getEvents(isDemo);
        
        // 2. Process each event with Hazard Impact Engine
        const impactResults = await Promise.all(events.map(async (event) => {
            // Check if physics failed to prevent unsafe downstream processing
            if (event.physics && event.physics.status === 'FAIL') {
                return {
                    event,
                    impact: { error: 'Impact processing aborted due to physics validation FAIL' }
                };
            }
            
            try {
                const polygon = create5kmPolygon(event.forecast.latitude, event.forecast.longitude);
                
                const impactEvidence = await hazardImpactEngine.assembleEvidence({
                    hazard_type: event.hazard_type,
                    event_id: event.event_id,
                    polygon: polygon,
                    event_metadata: event
                });
                
                event.impact = impactEvidence;
                return { event };
                
            } catch (engineError) {
                return { event, impact: { error: engineError.message } };
            }
        }));
        
        res.json({ success: true, results: impactResults });
    } catch (error) {
        res.status(500).json({ success: false, error: 'Failed to process AI impacts', details: error.message });
    }
});

module.exports = router;
