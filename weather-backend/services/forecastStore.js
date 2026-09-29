const fs = require('fs');
const path = require('path');

class ForecastStore {
  constructor() {
    this.storePath = path.join(__dirname, '../data/forecast_store.json');
    this.initializeStore();
  }

  initializeStore() {
    if (!fs.existsSync(path.dirname(this.storePath))) {
      fs.mkdirSync(path.dirname(this.storePath), { recursive: true });
    }
    if (!fs.existsSync(this.storePath)) {
      fs.writeFileSync(this.storePath, JSON.stringify({}, null, 2));
    }
  }

  _readData() {
    return JSON.parse(fs.readFileSync(this.storePath, 'utf8'));
  }

  _writeData(data) {
    fs.writeFileSync(this.storePath, JSON.stringify(data, null, 2));
  }

  /**
   * Store a forecast record
   * @param {Object} forecastRecord 
   */
  ingestForecast(forecastRecord) {
    if (!forecastRecord.event_id) throw new Error("Forecast record must have event_id");
    
    const data = this._readData();
    if (!data[forecastRecord.event_id]) {
      data[forecastRecord.event_id] = [];
    }

    // Check if we already have a forecast for this initialization time
    const existingIndex = data[forecastRecord.event_id].findIndex(
      f => f.forecast_initialization === forecastRecord.forecast_initialization
    );

    if (existingIndex >= 0) {
      data[forecastRecord.event_id][existingIndex] = forecastRecord;
    } else {
      data[forecastRecord.event_id].push(forecastRecord);
    }

    // Sort by initialization time
    data[forecastRecord.event_id].sort((a, b) => new Date(a.forecast_initialization) - new Date(b.forecast_initialization));
    
    this._writeData(data);
    return forecastRecord;
  }

  /**
   * Retrieve all forecast records for an event
   * @param {string} eventId 
   */
  getForecastsForEvent(eventId) {
    const data = this._readData();
    return data[eventId] || [];
  }
}

module.exports = new ForecastStore();
