const turf = require('@turf/turf');

class InfrastructureAdapter {
  constructor() {
    this.overpassUrl = 'https://overpass-api.de/api/interpreter';
  }

  /**
   * Converts a GeoJSON Polygon or MultiPolygon to an Overpass `poly` string
   * @param {Object} geojsonFeature - GeoJSON Feature
   * @returns {string} Overpass polygon string "lat1 lon1 lat2 lon2 ..."
   */
  _geoJsonToOverpassPoly(geojsonFeature) {
    let coordinates = [];

    const geomType = geojsonFeature.geometry.type;
    if (geomType === 'Polygon') {
      // Use the exterior ring (first array)
      coordinates = geojsonFeature.geometry.coordinates[0];
    } else if (geomType === 'MultiPolygon') {
      // Use the exterior ring of the first polygon as a simplification
      // Note: Full support for MultiPolygons in Overpass requires multiple queries or multiple poly filters
      // For this implementation, we take the largest or first one.
      coordinates = geojsonFeature.geometry.coordinates[0][0];
    } else {
      throw new Error('Unsupported geometry type for Overpass intersection');
    }

    // Overpass expects "lat1 lon1 lat2 lon2 ..."
    const polyString = coordinates.map(coord => `${coord[1]} ${coord[0]}`).join(' ');
    return polyString;
  }

  /**
   * Calculate distance of a way using its geometry nodes
   */
  _calculateWayLength(geometry) {
    if (!geometry || geometry.length < 2) return 0;
    
    let lengthKm = 0;
    for (let i = 0; i < geometry.length - 1; i++) {
      const pt1 = [geometry[i].lon, geometry[i].lat];
      const pt2 = [geometry[i+1].lon, geometry[i+1].lat];
      lengthKm += turf.distance(turf.point(pt1), turf.point(pt2), { units: 'kilometers' });
    }
    return lengthKm;
  }

  /**
   * Calculate infrastructure exposure
   * @param {Object} geojsonPolygon 
   */
  async calculateExposure(geojsonPolygon) {
    // 1. Calculate Hazard Area
    const areaSqMeters = turf.area(geojsonPolygon);
    const hazard_area_km2 = areaSqMeters / 1000000;

    // 2. Prepare Overpass Query
    const polyString = this._geoJsonToOverpassPoly(geojsonPolygon);

    // Categories:
    // Hospitals: amenity=hospital
    // Schools: amenity=school
    // Roads: highway=*
    // Railways: railway=*
    // Bridges: man_made=bridge OR bridge=yes (let's use both)
    // Airports: aeroway=aerodrome
    
    const query = `
      [out:json][timeout:60];
      (
        nwr["amenity"="hospital"](poly:"${polyString}");
        nwr["amenity"="school"](poly:"${polyString}");
        way["highway"](poly:"${polyString}");
        way["railway"](poly:"${polyString}");
        nwr["bridge"="yes"](poly:"${polyString}");
        nwr["man_made"="bridge"](poly:"${polyString}");
        nwr["aeroway"="aerodrome"](poly:"${polyString}");
      );
      out geom;
    `;

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 60000); // 60s timeout for Overpass
      
      const response = await fetch(this.overpassUrl, {
        method: 'POST',
        body: 'data=' + encodeURIComponent(query),
        headers: { 
          'Content-Type': 'application/x-www-form-urlencoded',
          'Accept': 'application/json',
          'User-Agent': 'WeatherTracker/1.0'
        },
        signal: controller.signal
      });
      clearTimeout(timeout);

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Overpass API Error: ${response.status} ${response.statusText}\n${errText}`);
      }

      const data = await response.json();
      
      // 3. Process Results
      const results = {
        hospitals: { count: 0, items: [] },
        schools: { count: 0, items: [] },
        roads: { count: 0, length_km: 0, items: [] },
        railways: { count: 0, length_km: 0, items: [] },
        bridges: { count: 0, items: [] },
        airports: { count: 0, items: [] }
      };

      // Set to track unique IDs to avoid double counting elements that might match multiple tags
      // (e.g., a bridge that is also a road)
      const processedIds = new Set();
      const elementMap = new Map();

      // First pass: index elements by ID so we don't process them twice 
      // if they matched multiple filters in the union
      for (const element of data.elements) {
        if (!elementMap.has(element.id)) {
          elementMap.set(element.id, element);
        }
      }

      for (const element of elementMap.values()) {
        const tags = element.tags || {};
        const id = element.id;

        // Helper to extract basic metadata for a point/polygon/line
        const extractMetadata = (type, el, additional = {}) => {
          const item = {
            type,
            osm_id: el.id,
            name: tags.name || 'Unknown'
          };
          
          if (el.type === 'way' && el.geometry && el.geometry.length > 0) {
            item.geometry = {
              type: 'LineString',
              coordinates: el.geometry.map(pt => [pt.lon, pt.lat])
            };
          } else if (el.lat && el.lon) {
            item.lat = el.lat;
            item.lon = el.lon;
          } else if (el.geometry && el.geometry.length > 0) {
            item.lat = el.geometry[0].lat;
            item.lon = el.geometry[0].lon;
          } else if (el.center) {
            item.lat = el.center.lat;
            item.lon = el.center.lon;
          }
          return { ...item, ...additional };
        };

        if (tags.amenity === 'hospital') {
          results.hospitals.count += 1;
          results.hospitals.items.push(extractMetadata('hospital', element));
        }
        if (tags.amenity === 'school') {
          results.schools.count += 1;
          results.schools.items.push(extractMetadata('school', element));
        }
        if (tags.highway && tags.highway !== 'no') {
          results.roads.count += 1;
          let rLen = 0;
          if (element.type === 'way' && element.geometry) {
            rLen = this._calculateWayLength(element.geometry);
            results.roads.length_km += rLen;
          }
          results.roads.items.push(extractMetadata('road', element, { length_km: Math.round(rLen * 100) / 100 }));
        }
        if (tags.railway && tags.railway !== 'no') {
          results.railways.count += 1;
          let rwLen = 0;
          if (element.type === 'way' && element.geometry) {
            rwLen = this._calculateWayLength(element.geometry);
            results.railways.length_km += rwLen;
          }
          results.railways.items.push(extractMetadata('railway', element, { length_km: Math.round(rwLen * 100) / 100 }));
        }
        if (tags.bridge === 'yes' || tags.man_made === 'bridge') {
          results.bridges.count += 1;
          results.bridges.items.push(extractMetadata('bridge', element));
        }
        if (tags.aeroway === 'aerodrome') {
          results.airports.count += 1;
          results.airports.items.push(extractMetadata('airport', element));
        }
      }

      // Round lengths
      results.roads.length_km = Math.round(results.roads.length_km * 100) / 100;
      results.railways.length_km = Math.round(results.railways.length_km * 100) / 100;

      const timestamp = new Date().toISOString();

      return {
        success: true,
        hazard_area_km2: Math.round(hazard_area_km2 * 100) / 100,
        limitations: "This represents infrastructure mapped in OpenStreetMap intersecting the supplied hazard geometry, not a complete authoritative inventory.",
        infrastructure: results,
        source: "OpenStreetMap",
        query_engine: "Overpass API",
        calculation_method: "spatial_intersection",
        computed_at: timestamp,
        provenance: {
          source: "OpenStreetMap",
          query_engine: "Overpass API",
          query_timestamp: timestamp,
          calculation_method: "polygon_intersection",
          hazard_geometry: "event polygon"
        }
      };
    } catch (error) {
      console.error("[InfrastructureAdapter] Exposure calculation failed:", error.message);
      throw error;
    }
  }
}

module.exports = new InfrastructureAdapter();
