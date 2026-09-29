const turf = require('@turf/turf');
const openMeteoAdapter = require('./openMeteoAdapter');
const infrastructureAdapter = require('./infrastructureAdapter');
const terrainAdapter = require('./terrainAdapter');

/**
 * Hazard Impact Evidence Engine
 * 
 * Assembles real-data evidence layers for a given hazard event and polygon.
 * Does NOT produce arbitrary risk scores or invented coefficients.
 * Instead, it collects independently verifiable measurements from:
 *   - Weather/Forecast data (Open-Meteo)
 *   - Infrastructure exposure (OSM/Overpass)
 *   - Terrain metrics (Copernicus DEM)
 * 
 * Each hazard type has its own evidence model because the relevant
 * physical quantities differ between rainfall, heat, cold, and cyclone events.
 */
class HazardImpactEngine {

  /**
   * Assemble evidence for a hazard event
   * @param {Object} params
   * @param {string} params.hazard_type - One of: rain, heat, cold, cyclone
   * @param {string} params.event_id - Event identifier
   * @param {Object} params.polygon - GeoJSON Feature (Polygon)
   * @param {Object} [params.event_metadata] - Optional event card data (severity, dates, etc.)
   */
  async assembleEvidence(params) {
    const { hazard_type, event_id, polygon, event_metadata = {} } = params;

    if (!polygon || !polygon.geometry) {
      throw new Error("Invalid GeoJSON polygon provided");
    }

    const centroid = turf.centroid(polygon);
    const [lon, lat] = centroid.geometry.coordinates;
    const areaKm2 = Math.round(turf.area(polygon) / 1000000 * 100) / 100;

    // Gather evidence layers in parallel where possible
    const evidenceLayers = {};
    const errors = {};
    const timestamp = new Date().toISOString();

    // --- Layer 1: Weather/Forecast evidence ---
    try {
      const weatherEvidence = await this._getWeatherEvidence(hazard_type, lat, lon, event_metadata);
      evidenceLayers.weather = weatherEvidence;
    } catch (err) {
      errors.weather = err.message;
      console.error(`[HazardImpactEngine] Weather evidence failed: ${err.message}`);
    }

    // --- Layer 2: Infrastructure exposure ---
    try {
      const infraResult = await infrastructureAdapter.calculateExposure(polygon);
      evidenceLayers.infrastructure = {
        hospitals_mapped: infraResult.infrastructure.hospitals.count,
        schools_mapped: infraResult.infrastructure.schools.count,
        roads_count: infraResult.infrastructure.roads.count,
        roads_length_km: infraResult.infrastructure.roads.length_km,
        railways_count: infraResult.infrastructure.railways.count,
        railways_length_km: infraResult.infrastructure.railways.length_km,
        bridges_mapped: infraResult.infrastructure.bridges.count,
        airports_mapped: infraResult.infrastructure.airports.count,
        source: infraResult.source,
        limitations: infraResult.limitations
      };
    } catch (err) {
      errors.infrastructure = err.message;
      console.error(`[HazardImpactEngine] Infrastructure evidence failed: ${err.message}`);
    }

    // --- Layer 3: Terrain metrics ---
    try {
      const terrainResult = await terrainAdapter.calculateTerrain(polygon);
      evidenceLayers.terrain = {
        ...terrainResult.terrain,
        slope_methodology: terrainResult.slope_methodology,
        source: terrainResult.source,
        dataset: terrainResult.dataset,
        native_resolution: terrainResult.native_resolution
      };
    } catch (err) {
      errors.terrain = err.message;
      console.error(`[HazardImpactEngine] Terrain evidence failed: ${err.message}`);
    }

    // --- Layer 4: Population exposure ---
    try {
      const populationAdapter = require('./populationAdapter');
      const popResult = await populationAdapter.calculateExposure(polygon);
      evidenceLayers.population = {
        population_exposed: popResult.population_exposed,
        status: popResult.status,
        source: popResult.source,
        dataset: popResult.dataset,
        year: popResult.year,
        resolution: popResult.resolution,
        country: popResult.country,
        calculation_method: popResult.calculation_method,
        aggregation: popResult.aggregation
      };
    } catch (err) {
      errors.population = err.message;
      console.error(`[HazardImpactEngine] Population evidence failed: ${err.message}`);
    }

    // --- Assemble hazard-specific evidence summary ---
    const hazardEvidence = this._buildHazardSummary(hazard_type, evidenceLayers, event_metadata);

    return {
      success: true,
      event_id,
      hazard_type,
      hazard_area_km2: areaKm2,
      centroid: { lat: Math.round(lat * 1000) / 1000, lon: Math.round(lon * 1000) / 1000 },
      evidence: evidenceLayers,
      hazard_summary: hazardEvidence,
      partial_failures: Object.keys(errors).length > 0 ? errors : undefined,
      methodology: "Evidence-based: each layer is independently sourced and verifiable. No arbitrary risk coefficients are applied.",
      sources: {
        weather: "Open-Meteo Archive API (ERA5 reanalysis)",
        infrastructure: "OpenStreetMap via Overpass API",
        terrain: "Copernicus DEM (90m) via Open-Meteo Elevation API"
      },
      computed_at: timestamp
    };
  }

  /**
   * Fetch weather evidence appropriate for the hazard type
   */
  async _getWeatherEvidence(hazardType, lat, lon, metadata) {
    // Determine date range: use event metadata if available, else last 7 days
    const endDate = metadata.date || new Date().toISOString().split('T')[0];
    const startDateObj = new Date(endDate);
    startDateObj.setDate(startDateObj.getDate() - 7);
    const startDate = startDateObj.toISOString().split('T')[0];

    // Select hourly variables based on hazard type
    let hourlyVars;
    switch (hazardType) {
      case 'rain':
        hourlyVars = 'precipitation,temperature_2m,relative_humidity_2m,surface_pressure';
        break;
      case 'heat':
        hourlyVars = 'temperature_2m,apparent_temperature,relative_humidity_2m';
        break;
      case 'cold':
        hourlyVars = 'temperature_2m,apparent_temperature,wind_speed_10m';
        break;
      case 'cyclone':
        hourlyVars = 'precipitation,temperature_2m,wind_speed_10m,wind_gusts_10m,surface_pressure';
        break;
      default:
        hourlyVars = 'temperature_2m,precipitation';
    }

    const result = await openMeteoAdapter.getHistoricalReanalysis({
      latitude: lat,
      longitude: lon,
      start_date: startDate,
      end_date: endDate,
      hourly: hourlyVars
    });

    // Extract summary statistics from the raw hourly data
    const hourly = result.value?.hourly || {};
    const summary = {};

    if (hazardType === 'rain' || hazardType === 'cyclone') {
      if (hourly.precipitation) {
        const precip = hourly.precipitation.filter(v => v !== null);
        summary.total_precipitation_mm = Math.round(precip.reduce((a, b) => a + b, 0) * 10) / 10;
        summary.max_hourly_precipitation_mm = Math.round(Math.max(...precip) * 10) / 10;
        summary.hours_with_rain = precip.filter(v => v > 0.1).length;
      }
    }

    if (hazardType === 'heat' || hazardType === 'cold') {
      if (hourly.temperature_2m) {
        const temps = hourly.temperature_2m.filter(v => v !== null);
        summary.mean_temperature_c = Math.round(temps.reduce((a, b) => a + b, 0) / temps.length * 10) / 10;
        summary.max_temperature_c = Math.round(Math.max(...temps) * 10) / 10;
        summary.min_temperature_c = Math.round(Math.min(...temps) * 10) / 10;
      }
      if (hourly.apparent_temperature) {
        const at = hourly.apparent_temperature.filter(v => v !== null);
        summary.max_apparent_temperature_c = Math.round(Math.max(...at) * 10) / 10;
        summary.min_apparent_temperature_c = Math.round(Math.min(...at) * 10) / 10;
      }
    }

    if (hazardType === 'cyclone') {
      if (hourly.wind_speed_10m) {
        const wind = hourly.wind_speed_10m.filter(v => v !== null);
        summary.max_wind_speed_kmh = Math.round(Math.max(...wind) * 10) / 10;
        summary.mean_wind_speed_kmh = Math.round(wind.reduce((a, b) => a + b, 0) / wind.length * 10) / 10;
      }
      if (hourly.wind_gusts_10m) {
        const gusts = hourly.wind_gusts_10m.filter(v => v !== null);
        summary.max_wind_gust_kmh = Math.round(Math.max(...gusts) * 10) / 10;
      }
      if (hourly.surface_pressure) {
        const press = hourly.surface_pressure.filter(v => v !== null);
        summary.min_surface_pressure_hpa = Math.round(Math.min(...press) * 10) / 10;
      }
    }

    if (hazardType === 'rain' && hourly.surface_pressure) {
      const press = hourly.surface_pressure.filter(v => v !== null);
      summary.min_surface_pressure_hpa = Math.round(Math.min(...press) * 10) / 10;
    }

    summary.observation_period = { start: startDate, end: endDate };
    summary.source = "Open-Meteo (ERA5 reanalysis)";

    return summary;
  }

  /**
   * Build a hazard-specific narrative summary from evidence layers.
   * This is a structured description of what was observed, NOT a risk score.
   */
  _buildHazardSummary(hazardType, evidence, metadata) {
    const summary = {
      hazard_type: hazardType,
      description_methodology: "Evidence-based summary. Each statement is derived from a specific data layer."
    };

    const statements = [];

    // Weather statements
    const w = evidence.weather;
    if (w) {
      switch (hazardType) {
        case 'rain':
          if (w.total_precipitation_mm !== undefined) {
            statements.push({
              observation: `${w.total_precipitation_mm} mm accumulated precipitation in the ERA5 reanalysis over the observation period`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          if (w.max_hourly_precipitation_mm !== undefined) {
            statements.push({
              observation: `Peak hourly rainfall intensity: ${w.max_hourly_precipitation_mm} mm/hr`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          break;

        case 'heat':
          if (w.max_temperature_c !== undefined) {
            statements.push({
              observation: `Maximum temperature reached ${w.max_temperature_c}°C`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          if (w.max_apparent_temperature_c !== undefined) {
            statements.push({
              observation: `Maximum apparent (feels-like) temperature: ${w.max_apparent_temperature_c}°C`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          break;

        case 'cold':
          if (w.min_temperature_c !== undefined) {
            statements.push({
              observation: `Minimum temperature dropped to ${w.min_temperature_c}°C`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          if (w.min_apparent_temperature_c !== undefined) {
            statements.push({
              observation: `Minimum apparent (feels-like) temperature: ${w.min_apparent_temperature_c}°C`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          break;

        case 'cyclone':
          if (w.max_wind_gust_kmh !== undefined) {
            statements.push({
              observation: `Maximum wind gust: ${w.max_wind_gust_kmh} km/h`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          if (w.total_precipitation_mm !== undefined) {
            statements.push({
              observation: `Associated rainfall: ${w.total_precipitation_mm} mm accumulated`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          if (w.min_surface_pressure_hpa !== undefined) {
            statements.push({
              observation: `Minimum surface pressure: ${w.min_surface_pressure_hpa} hPa`,
              source: "ERA5 reanalysis via Open-Meteo",
              data_type: "reanalysis"
            });
          }
          break;
      }
    }

    // Terrain statements
    const t = evidence.terrain;
    if (t) {
      if (hazardType === 'rain' || hazardType === 'cyclone') {
        statements.push({
          observation: `Affected area mean elevation: ${t.mean_elevation_m}m (range: ${t.min_elevation_m}m–${t.max_elevation_m}m)`,
          source: "Copernicus DEM via Open-Meteo"
        });
        if (t.min_elevation_m < 50) {
          statements.push({
            observation: `Low-lying terrain detected (min ${t.min_elevation_m}m) — potentially susceptible to water accumulation`,
            source: "Copernicus DEM via Open-Meteo",
            note: "Susceptibility observation, not a flood prediction"
          });
        }
      }
      if (hazardType === 'rain' && t.max_slope_deg > 3) {
        statements.push({
          observation: `Steep terrain detected (max slope ~${t.max_slope_deg}°) — potentially relevant to runoff and landslide susceptibility`,
          source: "Copernicus DEM via Open-Meteo (approximate slope)",
          note: "Slope is a derived approximation from sampled grid points"
        });
      }
    }

    // Infrastructure statements
    const inf = evidence.infrastructure;
    if (inf) {
      if (inf.hospitals_mapped > 0) {
        statements.push({
          observation: `${inf.hospitals_mapped} hospital(s) mapped in OpenStreetMap intersect the hazard footprint`,
          source: "OpenStreetMap via Overpass API",
          note: "OSM mapping coverage varies; this is not an authoritative hospital inventory"
        });
      }
      if (inf.schools_mapped > 0) {
        statements.push({
          observation: `${inf.schools_mapped} school(s) mapped in OpenStreetMap intersect the hazard footprint`,
          source: "OpenStreetMap via Overpass API"
        });
      }
      if (inf.roads_length_km > 0) {
        statements.push({
          observation: `${inf.roads_length_km} km of mapped road network intersects the hazard footprint`,
          source: "OpenStreetMap via Overpass API"
        });
      }
      if (inf.railways_length_km > 0) {
        statements.push({
          observation: `${inf.railways_length_km} km of mapped railway intersects the hazard footprint`,
          source: "OpenStreetMap via Overpass API"
        });
      }
      if (inf.airports_mapped > 0) {
        statements.push({
          observation: `${inf.airports_mapped} airport(s) mapped in OpenStreetMap intersect the hazard footprint`,
          source: "OpenStreetMap via Overpass API"
        });
      }
    }

    summary.evidence_statements = statements;
    return summary;
  }
}

module.exports = new HazardImpactEngine();
