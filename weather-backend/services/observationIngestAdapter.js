const fs = require('fs');
const path = require('path');
const turf = require('@turf/turf');

class ObservationIngestAdapter {
  constructor() {
    this.observationsPath = path.join(__dirname, '../data/observations');
  }

  /**
   * Load an authoritative ground truth GeoJSON for a specific event
   * @param {string} eventId 
   * @returns {Object|null} The GeoJSON Feature
   */
  getIndependentFootprint(eventId) {
    const files = fs.readdirSync(this.observationsPath);
    
    // Look for a file explicitly matching this event
    const targetFile = files.find(f => f.includes(eventId) && f.endsWith('.geojson'));
    
    if (!targetFile) {
      console.log(`[ObservationAdapter] No independent observation found for ${eventId}`);
      return null;
    }

    try {
      const filePath = path.join(this.observationsPath, targetFile);
      const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      
      // Ensure it's a valid GeoJSON FeatureCollection or Feature
      let feature = null;
      if (data.type === 'FeatureCollection' && data.features.length > 0) {
        // If there are multiple features (e.g. multi-polygon segments), union them
        if (data.features.length > 1) {
           let combined = data.features[0];
           for (let i = 1; i < data.features.length; i++) {
              combined = turf.union(turf.featureCollection([combined, data.features[i]]));
           }
           feature = combined;
        } else {
           feature = data.features[0];
        }
        
        // Preserve the properties of the first feature as authoritative metadata
        feature.properties = data.features[0].properties;
      } else if (data.type === 'Feature') {
        feature = data;
      } else {
        throw new Error("Invalid GeoJSON structure");
      }

      // Add our strict validation flag
      feature.properties.validation_source = "AUTHORITATIVE_INDEPENDENT";
      feature.properties.ingested_at = new Date().toISOString();

      return feature;
    } catch (error) {
      console.error(`[ObservationAdapter] Failed to parse ${targetFile}:`, error.message);
      return null;
    }
  }
}

module.exports = new ObservationIngestAdapter();
