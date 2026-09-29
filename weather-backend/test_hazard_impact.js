const hazardImpactEngine = require('./services/hazardImpactEngine');

const testFeature = {
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [78.17, 30.18],
        [78.19, 30.18],
        [78.19, 30.20],
        [78.17, 30.20],
        [78.17, 30.18]
      ]
    ]
  },
  "properties": {}
};

async function testHazardImpact() {
  console.log("=== Testing Hazard Impact Evidence Engine ===\n");
  try {
    const result = await hazardImpactEngine.assembleEvidence({
      hazard_type: 'rain',
      event_id: 'TEST-RAIN-001',
      polygon: testFeature,
      event_metadata: {
        date: new Date().toISOString().split('T')[0]
      }
    });

    console.log(JSON.stringify(result, null, 2));
  } catch (err) {
    console.error("Test failed:", err.message);
  }
}

testHazardImpact();
