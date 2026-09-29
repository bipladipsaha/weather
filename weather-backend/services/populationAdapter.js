const geoblaze = require('geoblaze');
const path = require('path');
const fs = require('fs');

const POPULATION_RASTER_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/IND/ind_ppp_2020_1km_Aggregated_UNadj.tif";
const LOCAL_RASTER_PATH = path.join(__dirname, '../data/exposure/worldpop_india_2020_1km.tif');

class PopulationAdapter {
  constructor() {
    this.rasterCache = null;
  }

  async calculateExposure(geojsonPolygon) {
    try {
      let rasterUrlOrPath = POPULATION_RASTER_URL;
      
      if (fs.existsSync(LOCAL_RASTER_PATH)) {
        rasterUrlOrPath = LOCAL_RASTER_PATH;
      }

      if (!this.rasterCache) {
        console.log(`[Population] Loading raster from: ${rasterUrlOrPath}`);
        // Read into memory buffer for geoblaze if local
        if (rasterUrlOrPath === LOCAL_RASTER_PATH) {
           const buffer = fs.readFileSync(LOCAL_RASTER_PATH);
           this.rasterCache = await geoblaze.parse(buffer);
        } else {
           this.rasterCache = await geoblaze.load(rasterUrlOrPath);
        }
      }

      const sumStats = await geoblaze.sum(this.rasterCache, geojsonPolygon);
      
      return {
        population_exposed: Math.round(sumStats[0]),
        status: "Success",
        source: "WorldPop",
        dataset: "ind_ppp_2020_1km_Aggregated_UNadj.tif",
        year: "2020",
        resolution: "1km",
        country: "India",
        calculation_method: "polygon_raster_intersection",
        aggregation: "sum"
      };
    } catch (error) {
      console.error('[Population] Calculation error:', error.message);
      return {
        population_exposed: null,
        status: "Data unavailable"
      };
    }
  }
}

module.exports = new PopulationAdapter();
