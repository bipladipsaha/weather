const turf = require('@turf/turf');

/**
 * Phase 3: Decision Window / Lead-Time Engine
 * 
 * Computes actionable lead time and tracks forecast evolution by comparing
 * successive STEA-Net forecasts (across lead times) against the actual 
 * observed event.
 */
class LeadTimeEngine {
  /**
   * Calculates the Decision Lead Time and forecast revisions/errors.
   * 
   * @param {Object} params
   * @param {Object} params.observation - Ground truth observed event
   * @param {string} params.observation.onset_time - ISO date string
   * @param {string} params.observation.peak_time - ISO date string
   * @param {number} params.observation.area_km2 
   * @param {Object} params.observation.centroid - { lat, lon }
   * @param {string} params.observation.source - e.g., "IMD Reports", "ERA5 reanalysis verification"
   * 
   * @param {Array<Object>} params.forecasts - Array of forecast revisions ordered by initialization time (oldest to newest)
   *   Each forecast should look like:
   *   {
   *     forecast_initialization: "2023-07-06T12:00:00Z",
   *     valid_time: "2023-07-09T12:00:00Z",
   *     lead_time_hours: 72,
   *     event_probability: 0.85,
   *     area_km2: 1200,
   *     centroid: { lat, lon }
   *   }
   * 
   * @param {Object} params.thresholds - Detection thresholds for the hazard
   * @param {number} params.thresholds.probability_min - e.g., 0.70
   * @param {number} params.thresholds.max_centroid_error_km - e.g., 200 (spatial overlap proxy)
   */
  calculateLeadTime(params) {
    const { observation, forecasts, thresholds } = params;

    if (!observation || !forecasts || forecasts.length === 0 || !thresholds) {
      throw new Error("Missing required parameters for Lead-Time calculation.");
    }

    const obsCentroidPt = turf.point([observation.centroid.lon, observation.centroid.lat]);
    const obsOnset = new Date(observation.onset_time);

    let firstValidDetection = null;
    const revisions = [];

    // Analyze each forecast revision
    for (const fc of forecasts) {
      const fcCentroidPt = turf.point([fc.centroid.lon, fc.centroid.lat]);
      const distanceErrorKm = turf.distance(obsCentroidPt, fcCentroidPt, { units: 'kilometers' });
      
      const areaErrorPct = observation.area_km2 > 0 
        ? ((fc.area_km2 - observation.area_km2) / observation.area_km2) * 100 
        : null;

      let iou = 0;
      if (observation.footprint && fc.footprint) {
        try {
           const intersection = turf.intersect(turf.featureCollection([observation.footprint, fc.footprint]));
           if (intersection) {
             const intersectionArea = turf.area(intersection);
             const unionArea = turf.area(observation.footprint) + turf.area(fc.footprint) - intersectionArea;
             iou = unionArea > 0 ? intersectionArea / unionArea : 0;
           }
        } catch (e) {
           console.error("IoU computation error", e.message);
        }
      }

      const fcInitTime = new Date(fc.forecast_initialization);
      const fcValidTime = new Date(fc.valid_time);
      const onsetErrorHours = (fcValidTime - obsOnset) / (1000 * 60 * 60);

      const isValidDetection = 
        fc.event_probability >= thresholds.probability_min &&
        distanceErrorKm <= thresholds.max_centroid_error_km;

      if (isValidDetection && !firstValidDetection) {
        firstValidDetection = fc;
      }

      revisions.push({
        forecast_initialization: fc.forecast_initialization,
        lead_time_hours: fc.lead_time_hours,
        probability: fc.event_probability,
        centroid_distance_error_km: Math.round(distanceErrorKm * 10) / 10,
        area_error_pct: areaErrorPct !== null ? Math.round(areaErrorPct * 10) / 10 : null,
        onset_time_error_hours: Math.round(onsetErrorHours * 10) / 10,
        iou: Math.round(iou * 1000) / 1000,
        meets_detection_criteria: isValidDetection,
        footprint: fc.footprint // passing footprint for rendering in the UI
      });
    }

    // Sort revisions by lead time descending (e.g. T-120 -> T-96 -> T-72)
    revisions.sort((a, b) => b.lead_time_hours - a.lead_time_hours);

    // Calculate Final Decision Lead Time based on timestamps
    let decisionLeadTimeHours = null;
    if (firstValidDetection) {
      const firstDetectionTime = new Date(firstValidDetection.forecast_initialization);
      decisionLeadTimeHours = (obsOnset - firstDetectionTime) / (1000 * 60 * 60);
    }

    return {
      success: true,
      observation_truth: {
        onset_time: observation.onset_time,
        source: observation.source,
        data_type: "observation"
      },
      detection_criteria: {
        probability_threshold: thresholds.probability_min,
        max_spatial_error_km: thresholds.max_centroid_error_km,
        rule: `Probability >= ${thresholds.probability_min} AND Centroid Error <= ${thresholds.max_centroid_error_km}km`
      },
      decision_lead_time_hours: decisionLeadTimeHours ? Math.round(decisionLeadTimeHours * 10) / 10 : null,
      first_detection_forecast: firstValidDetection ? firstValidDetection.forecast_initialization : null,
      forecast_revisions: revisions,
      methodology: "Lead time calculated objectively from forecast initialization timestamp to observed onset timestamp, based on strict probability and spatial matching thresholds."
    };
  }
}

module.exports = new LeadTimeEngine();
