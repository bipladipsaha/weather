import json
import csv
import os

print("Extracting rich impact and forecast data from GIS engine...")

impact_data = {}

# 1. Load forecast stats from events_catalog.json
catalog_path = '../SIH26078_COMPLETE_BACKUP/working/sih26078_diagnostic/gis_impact_engine/event_intelligence/stea_net_frontend_package/frontend_package/events_catalog.json'
catalog_data = {}
if os.path.exists(catalog_path):
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            cat = json.load(f)
            for evt in cat.get('events', []):
                catalog_data[evt['persistent_event_id']] = {
                    'first_probability': evt.get('first_probability', 0) * 100,
                    'latest_probability': evt.get('latest_probability', 0) * 100,
                    'area_change_pct': evt.get('area_change_pct', 0),
                    'pop_change_pct': evt.get('population_change_pct', 0)
                }
        print(f"Loaded {len(catalog_data)} forecast revisions from catalog.")
    except Exception as e:
        print("Error parsing catalog JSON:", e)

# 2. Load GIS impacts from CSV
csv_path = '../SIH26078_COMPLETE_BACKUP/working/sih26078_diagnostic/gis_impact_engine/event_intelligence/event_intelligence_latest.csv'
if os.path.exists(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_id = row['event_id']
                
                # Parse metrics safely
                def get_float(key):
                    try: return float(row.get(key, 0) or 0)
                    except: return 0.0

                pop = get_float('population_exposed_2022')
                roads = get_float('exposed_road_km_total')
                hospitals = get_float('exposed_hospital_count')
                schools = get_float('exposed_school_count')
                power = get_float('exposed_power_generation_feature_count')
                airports = get_float('exposed_airport_count')
                
                crop_frac = get_float('environment_cropland_fraction')
                urban_frac = get_float('environment_built_up_fraction')
                forest_frac = get_float('environment_tree_cover_fraction')
                
                coast_dist = get_float('environment_coast_distance_km')
                elevation = get_float('environment_mean_land_elevation_m')
                
                impact_score = get_float('impact_context_score_0_100')
                compound = row.get('compound_hazard', '')
                
                # Format metrics intelligently
                metrics = []
                
                if pop > 0:
                    pop_str = f"{pop/1000000:.1f}M" if pop > 1000000 else f"{pop/1000:.1f}K" if pop > 1000 else str(int(pop))
                    metrics.append({"label": "Population", "value": pop_str, "color": "neo-blue"})
                
                if roads > 50:
                    metrics.append({"label": "Major Roads", "value": f"{int(roads)} km", "color": "neo-purple"})
                    
                if hospitals > 0:
                    metrics.append({"label": "Hospitals at Risk", "value": str(int(hospitals)), "color": "neo-pink"})
                    
                if power > 0:
                    metrics.append({"label": "Power Facilities", "value": str(int(power)), "color": "neo-pink"})
                    
                if crop_frac > 0.3:
                    metrics.append({"label": "Cropland Impact", "value": f"{int(crop_frac*100)}%", "color": "neo-green"})
                    
                if urban_frac > 0.3:
                    metrics.append({"label": "Urban Exposure", "value": f"{int(urban_frac*100)}%", "color": "neo-blue"})
                    
                if coast_dist < 20:
                    metrics.append({"label": "Coastal Threat", "value": f"{int(coast_dist)} km to coast", "color": "neo-blue"})
                    
                if compound and compound != "False" and compound != "":
                    metrics.append({"label": "Compound Hazard", "value": "Yes", "color": "neo-pink"})
                
                # If it's a completely remote event, show environmental stats
                if len(metrics) == 0:
                    if forest_frac > 0.4:
                        metrics.append({"label": "Forest Area", "value": f"{int(forest_frac*100)}%", "color": "neo-green"})
                    metrics.append({"label": "Elevation", "value": f"{int(elevation)} m", "color": "neo-purple"})
                    metrics.append({"label": "Remote Anomaly", "value": "Unpopulated", "color": "neo-yellow"})
                
                c_data = catalog_data.get(event_id, {})
                
                impact_data[event_id] = {
                    'score': round(impact_score, 1),
                    'metrics': metrics[:4], # take top 4 most relevant metrics
                    'all_metrics': metrics,
                    'forecast_revision': {
                        'prevProbability': round(c_data.get('first_probability') or 0),
                        'currProbability': round(c_data.get('latest_probability') or 0),
                        'areaChangePct': round(c_data.get('area_change_pct') or 0),
                        'popChangePct': round(c_data.get('pop_change_pct') or 0)
                    }
                }
                
        print(f"Extracted rich impact data for {len(impact_data)} events.")
    except Exception as e:
        print("Error parsing CSV:", e)
else:
    print("CSV not found at", csv_path)

with open('dashboard_impacts_rich.json', 'w') as f:
    json.dump(impact_data, f)
print("Saved to dashboard_impacts_rich.json.")
