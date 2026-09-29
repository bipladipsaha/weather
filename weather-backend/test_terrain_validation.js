/**
 * Phase 2C-1 Terrain Validation
 * 
 * Tests the DEM adapter against 3 geographically distinct regions:
 * 
 * 1. Himalayan foothills (Dehradun) — expect high elevation, steep slopes
 * 2. Coastal lowland (Mumbai coast)  — expect low elevation, gentle slopes
 * 3. Gangetic plain (near Lucknow)   — expect moderate elevation, nearly flat
 * 
 * If the adapter is working correctly:
 *   Dehradun elevation >> Mumbai elevation
 *   Dehradun slope >> Lucknow slope
 *   Mumbai min elevation should be near sea level
 */

const terrainAdapter = require('./services/terrainAdapter');

const testCases = [
  {
    name: "Himalayan Foothills (Dehradun)",
    expect: "High elevation (~500-600m), noticeable slope",
    feature: {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[
          [78.17, 30.18],
          [78.19, 30.18],
          [78.19, 30.20],
          [78.17, 30.20],
          [78.17, 30.18]
        ]]
      },
      properties: {}
    }
  },
  {
    name: "Coastal Lowland (South Mumbai)",
    expect: "Very low elevation (near sea level), gentle slope",
    feature: {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[
          [72.82, 18.92],
          [72.84, 18.92],
          [72.84, 18.94],
          [72.82, 18.94],
          [72.82, 18.92]
        ]]
      },
      properties: {}
    }
  },
  {
    name: "Gangetic Plain (Lucknow outskirts)",
    expect: "Moderate elevation (~120m), nearly flat",
    feature: {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[
          [80.94, 26.84],
          [80.96, 26.84],
          [80.96, 26.86],
          [80.94, 26.86],
          [80.94, 26.84]
        ]]
      },
      properties: {}
    }
  }
];

async function runValidation() {
  console.log("=== PHASE 2C-1 TERRAIN VALIDATION ===\n");

  const results = [];

  for (const tc of testCases) {
    console.log(`Testing: ${tc.name}`);
    console.log(`  Expected: ${tc.expect}`);
    try {
      const result = await terrainAdapter.calculateTerrain(tc.feature);
      results.push({ name: tc.name, result });
      console.log(`  Mean Elevation: ${result.terrain.mean_elevation_m} m`);
      console.log(`  Min Elevation:  ${result.terrain.min_elevation_m} m`);
      console.log(`  Max Elevation:  ${result.terrain.max_elevation_m} m`);
      console.log(`  Elev Range:     ${result.terrain.elevation_range_m} m`);
      console.log(`  Mean Slope:     ${result.terrain.mean_slope_deg}°`);
      console.log(`  Max Slope:      ${result.terrain.max_slope_deg}°`);
      console.log(`  Sample Points:  ${result.sample_points_used}`);
      console.log(`  Sampling Res:   ${result.sampling_resolution_m}m`);
      console.log(`  Native Res:     ${result.native_resolution}`);
      console.log("");
    } catch (err) {
      console.error(`  FAILED: ${err.message}\n`);
      results.push({ name: tc.name, error: err.message });
    }
  }

  // Cross-validation checks
  console.log("=== CROSS-VALIDATION CHECKS ===\n");

  if (results.length === 3 && results.every(r => r.result)) {
    const [dehradun, mumbai, lucknow] = results.map(r => r.result.terrain);

    const checks = [
      {
        test: "Dehradun elevation > Mumbai elevation",
        pass: dehradun.mean_elevation_m > mumbai.mean_elevation_m,
        detail: `${dehradun.mean_elevation_m}m vs ${mumbai.mean_elevation_m}m`
      },
      {
        test: "Dehradun elevation > Lucknow elevation",
        pass: dehradun.mean_elevation_m > lucknow.mean_elevation_m,
        detail: `${dehradun.mean_elevation_m}m vs ${lucknow.mean_elevation_m}m`
      },
      {
        test: "Mumbai min elevation near sea level (< 50m)",
        pass: mumbai.min_elevation_m < 50,
        detail: `${mumbai.min_elevation_m}m`
      },
      {
        test: "Lucknow is flatter than Dehradun (lower mean slope)",
        pass: lucknow.mean_slope_deg < dehradun.mean_slope_deg,
        detail: `${lucknow.mean_slope_deg}° vs ${dehradun.mean_slope_deg}°`
      },
      {
        test: "Dehradun has larger elevation range than Lucknow",
        pass: dehradun.elevation_range_m > lucknow.elevation_range_m,
        detail: `${dehradun.elevation_range_m}m vs ${lucknow.elevation_range_m}m`
      }
    ];

    let allPass = true;
    for (const c of checks) {
      const icon = c.pass ? "✅" : "❌";
      console.log(`${icon} ${c.test}`);
      console.log(`   ${c.detail}`);
      if (!c.pass) allPass = false;
    }

    console.log("\n" + (allPass 
      ? "✅ ALL CHECKS PASSED — Terrain adapter is geographically consistent." 
      : "❌ SOME CHECKS FAILED — Investigate before building on top of this."));
  } else {
    console.log("❌ Could not run cross-validation — some tests failed.");
  }
}

runValidation().catch(console.error);
