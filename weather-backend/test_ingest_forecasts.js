

async function ingestForecasts() {
  const eventId = "EV-2023-HP-FLOOD";
  const eventLat = 31.9578;
  const eventLon = 77.1095;
  const validTime = "2023-07-08T00:00:00Z";

  // Simulate raw output fields from STEA-Net at different lead times
  const rawOutputs = [
    {
      event_id: eventId,
      initialization_time: "2023-07-03T00:00:00Z", // T-120
      valid_time: validTime,
      hazard: "flood",
      probability_field: { lat: eventLat + 0.5, lon: eventLon + 0.5, maxProbability: 0.45 }
    },
    {
      event_id: eventId,
      initialization_time: "2023-07-04T00:00:00Z", // T-96
      valid_time: validTime,
      hazard: "flood",
      probability_field: { lat: eventLat + 0.2, lon: eventLon + 0.2, maxProbability: 0.65 }
    },
    {
      event_id: eventId,
      initialization_time: "2023-07-05T00:00:00Z", // T-72
      valid_time: validTime,
      hazard: "flood",
      probability_field: { lat: eventLat + 0.1, lon: eventLon + 0.1, maxProbability: 0.78 }
    },
    {
      event_id: eventId,
      initialization_time: "2023-07-06T00:00:00Z", // T-48
      valid_time: validTime,
      hazard: "flood",
      probability_field: { lat: eventLat + 0.05, lon: eventLon + 0.05, maxProbability: 0.88 }
    }
  ];

  for (const raw of rawOutputs) {
    try {
      const res = await fetch('http://localhost:3001/api/forecast/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(raw)
      });
      const data = await res.json();
      if (data.success) {
        console.log(`Ingested forecast for T-${data.data.lead_time_hours}h: Prob=${data.data.event_probability}`);
      } else {
        console.error("Failed to ingest:", data.error);
      }
    } catch (err) {
      console.error("Request failed:", err.message);
    }
  }
}

ingestForecasts();
