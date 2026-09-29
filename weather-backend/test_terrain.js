const terrainAdapter = require('./services/terrainAdapter');

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

async function test() {
  try {
    const result = await terrainAdapter.calculateTerrain(testFeature);
    console.log(JSON.stringify(result, null, 2));
  } catch (err) {
    console.error("Test failed:", err);
  }
}

test();
