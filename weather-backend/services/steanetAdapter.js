const forecastEventExtractor = require('./forecastEventExtractor');

class SteanetAdapter {
  /**
   * Translates a raw STEA-Net inference output into a normalized forecast record.
   * 
   * @param {Object} rawOutput - Raw model output
   * @param {string} rawOutput.event_id
   * @param {string} rawOutput.initialization_time
   * @param {string} rawOutput.valid_time
   * @param {string} rawOutput.hazard
   * @param {Object} rawOutput.probability_field - e.g., { lat, lon, maxProbability }
   */
  processModelOutput(rawOutput) {
    const { event_id, initialization_time, valid_time, hazard, tensor_data } = rawOutput;

    // Run the extraction (connected component / thresholding)
    // We pass the tensor_data and a low threshold to ensure we extract the feature for evaluation.
    // The lead-time engine will apply the strict threshold later.
    const extractedFeature = forecastEventExtractor.extractEvent(tensor_data, 0.1);

    // Compute lead time hours based on valid_time and initialization_time
    const initDate = new Date(initialization_time);
    const validDate = new Date(valid_time);
    const leadTimeHours = (validDate - initDate) / (1000 * 60 * 60);
    
    // Find max probability even if we didn't extract a coherent feature
    let maxProb = 0;
    if (tensor_data && tensor_data.grid) {
       for (const row of tensor_data.grid) {
          for (const val of row) {
             maxProb = Math.max(maxProb, val);
          }
       }
    }

    return {
      event_id,
      forecast_initialization: initialization_time,
      valid_time: valid_time,
      lead_time_hours: leadTimeHours,
      hazard,
      event_probability: extractedFeature ? extractedFeature.properties.probability : maxProb,
      centroid: extractedFeature ? extractedFeature.properties.centroid : { lat: 0, lon: 0 },
      area_km2: extractedFeature ? extractedFeature.properties.area_km2 : 0,
      footprint: extractedFeature ? {
        type: "Feature",
        geometry: extractedFeature.geometry,
        properties: {}
      } : null,
      source: "STEA-Net",
      data_type: "model_output"
    };
  }
}

module.exports = new SteanetAdapter();
