import json
import csv
import os

print("Extracting impact data from GIS engine...")

impact_data = {}

csv_path = '../SIH26078_COMPLETE_BACKUP/working/sih26078_diagnostic/gis_impact_engine/event_intelligence/event_intelligence_latest.csv'
if os.path.exists(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_id = row['event_id']
                # Convert strings to floats/ints safely
                try:
                    pop = float(row.get('population_exposed_2022', 0))
                except:
                    pop = 0
                
                try:
                    roads = float(row.get('exposed_road_km_total', 0))
                except:
                    roads = 0
                
                # Format nicely
                if pop > 1000000:
                    pop_str = f"{pop/1000000:.1f}M"
                elif pop > 1000:
                    pop_str = f"{pop/1000:.1f}K"
                else:
                    pop_str = str(int(pop)) if pop > 0 else "0"
                
                roads_str = f"{int(roads)} km" if roads > 0 else "0"
                
                impact_data[event_id] = {
                    'population': pop_str,
                    'roads': roads_str
                }
        print(f"Extracted impact data for {len(impact_data)} events.")
    except Exception as e:
        print("Error parsing CSV:", e)
else:
    print("CSV not found at", csv_path)

with open('dashboard_impacts.json', 'w') as f:
    json.dump(impact_data, f)
print("Saved to dashboard_impacts.json.")
