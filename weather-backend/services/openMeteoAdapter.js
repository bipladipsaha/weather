const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const WEATHER_DATA_DIR = path.join(__dirname, '../data/weather');

class OpenMeteoAdapter {
  constructor() {
    if (!fs.existsSync(WEATHER_DATA_DIR)) {
      fs.mkdirSync(WEATHER_DATA_DIR, { recursive: true });
    }
  }

  /**
   * Generate a cache key based on query parameters
   */
  _generateCacheKey(params) {
    const str = JSON.stringify(params, Object.keys(params).sort());
    return crypto.createHash('md5').update(str).digest('hex');
  }

  /**
   * Fetch historical reanalysis data
   * @param {Object} params - { latitude, longitude, start_date, end_date, hourly }
   */
  async getHistoricalReanalysis(params) {
    const { latitude, longitude, start_date, end_date, hourly = 'temperature_2m,precipitation' } = params;
    
    // Request Validation
    if (!latitude || !longitude || !start_date || !end_date) {
      throw new Error('Missing required parameters: latitude, longitude, start_date, end_date');
    }

    const cacheKey = `reanalysis_${this._generateCacheKey(params)}.json`;
    const cachePath = path.join(WEATHER_DATA_DIR, cacheKey);

    // Cache check
    if (fs.existsSync(cachePath)) {
      console.log(`[Open-Meteo] Cache hit for ${cacheKey}`);
      return JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    }

    const url = `https://archive-api.open-meteo.com/v1/archive?latitude=${latitude}&longitude=${longitude}&start_date=${start_date}&end_date=${end_date}&hourly=${hourly}`;

    console.log(`[Open-Meteo] Fetching: ${url}`);
    
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000); // 10s timeout

    try {
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) {
        throw new Error(`Open-Meteo API Error: ${response.statusText}`);
      }
      
      const rawData = await response.json();
      
      // Determine dataset provenance dynamically.
      // Open-Meteo's archive API primarily uses ERA5 for historical data, 
      // but they list the exact models in `rawData.models` if explicitly requested or provided.
      let actualDataset = "era5_reanalysis_default";
      if (params.models) {
        actualDataset = params.models;
      } else if (rawData.timezone) {
        // Safe inference if no models array is present but request succeeded
        actualDataset = "era5_reanalysis";
      }
      
      // Normalize internal format while preserving provenance
      const normalizedData = {
        value: rawData,
        source: "open-meteo",
        dataset: actualDataset,
        timestamp: new Date().toISOString(),
        method: "API_FETCH",
        query_params: params
      };

      // Save to cache
      fs.writeFileSync(cachePath, JSON.stringify(normalizedData, null, 2), 'utf-8');
      
      return normalizedData;
    } catch (error) {
      console.error('[Open-Meteo] Fetch error:', error.message);
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }
}

module.exports = new OpenMeteoAdapter();
