const fs = require('fs');
const path = require('path');

const EVENTS_DIR = path.join(__dirname, '../data/events');

class EventService {
  /**
   * Ensure events directory exists
   */
  constructor() {
    if (!fs.existsSync(EVENTS_DIR)) {
      fs.mkdirSync(EVENTS_DIR, { recursive: true });
    }
  }

  /**
   * Get all structured Indian events
   */
  getAllEvents() {
    const events = [];
    try {
      const files = fs.readdirSync(EVENTS_DIR);
      for (const file of files) {
        if (file.endsWith('.json')) {
          const content = fs.readFileSync(path.join(EVENTS_DIR, file), 'utf-8');
          events.push(JSON.parse(content));
        }
      }
    } catch (error) {
      console.error('Error reading events:', error.message);
    }
    return events;
  }

  /**
   * Save a normalized event
   */
  saveEvent(eventModel) {
    const filePath = path.join(EVENTS_DIR, `${eventModel.event_id}.json`);
    fs.writeFileSync(filePath, JSON.stringify(eventModel.toJSON(), null, 2), 'utf-8');
    return eventModel.toJSON();
  }
}

module.exports = new EventService();
