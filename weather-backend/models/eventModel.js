class EventModel {
  /**
   * Creates a normalized Event object.
   * @param {Object} data 
   */
  constructor(data) {
    // Required fields
    if (!data.event_id || !data.hazard || !data.state || !data.latitude || !data.longitude) {
      throw new Error('Missing required fields: event_id, hazard, state, latitude, longitude');
    }

    this.event_id = data.event_id;
    this.hazard = data.hazard; // e.g., 'flood', 'cyclone', 'heatwave'
    this.state = data.state;
    this.district = data.district || 'Unknown';
    this.start_time = data.start_time || null;
    this.end_time = data.end_time || null;
    this.latitude = parseFloat(data.latitude);
    this.longitude = parseFloat(data.longitude);
    this.source = data.source || 'Unknown';
    
    // Optional fields
    this.observed_intensity = data.observed_intensity || null;
    this.damage_data = data.damage_data || null; // e.g., { lives_lost: 10, crop_damage_ha: 500 }
  }

  toJSON() {
    return {
      event_id: this.event_id,
      hazard: this.hazard,
      state: this.state,
      district: this.district,
      start_time: this.start_time,
      end_time: this.end_time,
      latitude: this.latitude,
      longitude: this.longitude,
      source: this.source,
      observed_intensity: this.observed_intensity,
      damage_data: this.damage_data
    };
  }
}

module.exports = EventModel;
