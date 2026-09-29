const leadTimeEngine = require('./services/leadTimeEngine');

const testPayload = {
  observation: {
    onset_time: "2023-07-09T12:00:00Z",
    peak_time: "2023-07-10T00:00:00Z",
    area_km2: 1500,
    centroid: { lat: 30.10, lon: 78.10 },
    source: "IMD Verified Ground Truth"
  },
  thresholds: {
    probability_min: 0.70,
    max_centroid_error_km: 150
  },
  forecasts: [
    {
      forecast_initialization: "2023-07-04T12:00:00Z", // T-120h
      valid_time: "2023-07-09T12:00:00Z",
      lead_time_hours: 120,
      event_probability: 0.45,
      area_km2: 2500,
      centroid: { lat: 31.00, lon: 77.00 } // far off
    },
    {
      forecast_initialization: "2023-07-05T12:00:00Z", // T-96h
      valid_time: "2023-07-09T12:00:00Z",
      lead_time_hours: 96,
      event_probability: 0.65,
      area_km2: 2000,
      centroid: { lat: 30.50, lon: 77.50 }
    },
    {
      forecast_initialization: "2023-07-06T12:00:00Z", // T-72h
      valid_time: "2023-07-09T12:00:00Z",
      lead_time_hours: 72,
      event_probability: 0.78, // Crosses threshold
      area_km2: 1800,
      centroid: { lat: 30.20, lon: 78.00 } // Good spatial match
    },
    {
      forecast_initialization: "2023-07-07T12:00:00Z", // T-48h
      valid_time: "2023-07-09T12:00:00Z",
      lead_time_hours: 48,
      event_probability: 0.88,
      area_km2: 1600,
      centroid: { lat: 30.15, lon: 78.05 }
    },
    {
      forecast_initialization: "2023-07-08T12:00:00Z", // T-24h
      valid_time: "2023-07-09T12:00:00Z",
      lead_time_hours: 24,
      event_probability: 0.92,
      area_km2: 1550,
      centroid: { lat: 30.12, lon: 78.08 }
    }
  ]
};

function testLeadTimeEngine() {
  console.log("=== Testing Phase 3: Decision Window & Lead-Time Engine ===\n");
  try {
    const result = leadTimeEngine.calculateLeadTime(testPayload);
    console.log(JSON.stringify(result, null, 2));
  } catch (err) {
    console.error("Test failed:", err.message);
  }
}

testLeadTimeEngine();
