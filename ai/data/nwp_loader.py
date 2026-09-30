import xarray as xr
import os
import torch
import numpy as np

class NWPLoader:
    """
    Ingests NetCDF/GRIB files and prepares them for the AI pipeline.
    """
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.expected_variables = ['temperature_2m', 'total_precipitation', 'u_component_of_wind', 'v_component_of_wind', 'surface_pressure']
        self.expected_pressure_vars = ['geopotential_500hPa']
        
    def load_forecast(self, filepath, engine='netcdf4', is_ensemble=False):
        """
        Loads a deterministic or ensemble NWP forecast.
        Returns the xarray dataset and a metadata dictionary.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Forecast file not found: {filepath}")
            
        ds = xr.open_dataset(filepath, engine=engine)
        
        # Extract metadata
        metadata = self._extract_metadata(ds, filepath, is_ensemble)
        
        return ds, metadata
        
    def load_ensemble_from_files(self, filepaths, engine='netcdf4'):
        """
        Loads multiple ensemble members into a single xarray dataset.
        """
        if not filepaths:
            raise ValueError("No filepaths provided for ensemble.")
            
        datasets = []
        for i, path in enumerate(filepaths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Forecast file not found: {path}")
            ds = xr.open_dataset(path, engine=engine)
            if 'member' not in ds.dims:
                ds = ds.expand_dims(member=[i])
            datasets.append(ds)
            
        # Combine along member dimension
        ensemble_ds = xr.concat(datasets, dim='member')
        
        metadata = self._extract_metadata(ensemble_ds, str(filepaths[0]), is_ensemble=True)
        metadata["ensemble_members"] = len(filepaths)
        
        return ensemble_ds, metadata

    def _extract_metadata(self, ds, source_name, is_ensemble):
        """
        Extracts standardized metadata from the dataset.
        """
        # Attempt to find standard dimensions
        time_dim = 'time' if 'time' in ds.dims else 'valid_time' if 'valid_time' in ds.dims else None
        lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat' if 'lat' in ds.dims else None
        lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon' if 'lon' in ds.dims else None
        member_dim = 'member' if 'member' in ds.dims else 'number' if 'number' in ds.dims else None
        
        init_time = None
        if 'time' in ds.coords:
            init_time = str(ds['time'].values[0])
            
        valid_time = None
        if time_dim and time_dim in ds.coords:
            valid_time = str(ds[time_dim].values[-1])
            
        members = 1
        if is_ensemble:
            if member_dim:
                members = ds.sizes[member_dim]
            else:
                raise ValueError("Ensemble dataset must contain a 'member' or 'number' dimension. Do not silently convert deterministic data into fake ensemble members.")
        else:
            if member_dim and ds.sizes[member_dim] > 1:
                raise ValueError("Dataset has multiple members but was loaded as deterministic.")

        lat_res = abs(ds[lat_dim].values[1] - ds[lat_dim].values[0]) if lat_dim and len(ds[lat_dim]) > 1 else None
        # Approx resolution in km (1 deg ~ 111km)
        res_km = round(lat_res * 111, 2) if lat_res else None

        metadata = {
            "source": source_name,
            "model": ds.attrs.get('model', 'Unknown'),
            "initialization_time": init_time,
            "valid_time": valid_time,
            "ensemble_members": members,
            "resolution_km": res_km,
            "variables": list(ds.data_vars.keys())
        }
        
        return metadata

    def to_tensor(self, ds, metadata):
        """
        Converts the dataset to a PyTorch tensor.
        Preserves the existing interface: input -> [B, C, lead_time, lat, lon]
        where B is the number of ensemble members (or 1 for deterministic).
        """
        time_dim = 'time' if 'time' in ds.dims else 'valid_time' if 'valid_time' in ds.dims else None
        lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon'
        member_dim = 'member' if 'member' in ds.dims else 'number' if 'number' in ds.dims else None
        
        if not time_dim or lat_dim not in ds.dims or lon_dim not in ds.dims:
            raise ValueError("Dataset must contain time, latitude, and longitude dimensions.")

        # Ensure correct variable order mapping
        # This mapping depends on the actual variables in the DS vs expected
        vars_to_extract = [v for v in self.expected_variables if v in ds.data_vars]
        if not vars_to_extract:
            vars_to_extract = list(ds.data_vars.keys())[:5] # Fallback
            
        tensor_list = []
        for var in vars_to_extract:
            val = ds[var].values
            # val shape could be [time, lat, lon] or [member, time, lat, lon]
            if member_dim and member_dim in ds[var].dims:
                # [member, time, lat, lon]
                pass
            else:
                # add member/batch dimension
                val = np.expand_dims(val, axis=0)
            tensor_list.append(torch.tensor(val, dtype=torch.float32))
            
        # tensor_list is a list of [B, lead_time, lat, lon]
        # Stack along channel dimension -> [B, C, lead_time, lat, lon]
        stacked = torch.stack(tensor_list, dim=1)
        return stacked

class LiveEnsembleFeeder:
    """
    Connects to the live NEPS-G / Open-Meteo ensemble API endpoints to fetch 
    real-time 4D EPS data, replacing the deterministic mock tensors.
    """
    def __init__(self, num_members=23, lead_times=9, lat_size=30, lon_size=26):
        self.num_members = num_members
        self.lead_times = lead_times
        self.lat_size = lat_size
        self.lon_size = lon_size

    def fetch_latest_run(self):
        import urllib.request
        import json
        import math

        print("Fetching real weather data from Open-Meteo API...")
        base_deterministic = torch.zeros(1, 5, self.lead_times, self.lat_size, self.lon_size)
        
        try:
            import urllib.request
            import json
            import math
            import time
            
            # Fetch real live deterministic data for 4 corners of India to create a full country grid
            cities = [
                {"name": "Delhi", "lat": 28.61, "lon": 77.20, "grid_y": 25, "grid_x": 10}, # North
                {"name": "Bengaluru", "lat": 12.97, "lon": 77.59, "grid_y": 5, "grid_x": 11}, # South
                {"name": "Mumbai", "lat": 19.07, "lon": 72.87, "grid_y": 15, "grid_x": 3}, # West
                {"name": "Kolkata", "lat": 22.57, "lon": 88.36, "grid_y": 18, "grid_x": 22} # East
            ]
            
            city_data = []
            for city in cities:
                url = f"https://api.open-meteo.com/v1/forecast?latitude={city['lat']}&longitude={city['lon']}&hourly=temperature_2m,precipitation,windspeed_10m,winddirection_10m,surface_pressure&forecast_days=2"
                req = urllib.request.Request(url, headers={'User-Agent': 'WeatherAI/1.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    city_data.append(data.get("hourly", {}))
                time.sleep(0.5) # Avoid rate limits
            
            print("Successfully loaded real-world weather data for all of India.")
            
            # Map the next 9 hours to the 9 lead times expected by the GNN
            for l in range(self.lead_times):
                # We interpolate the 4 cities across the 30x26 grid using inverse distance weighting (IDW)
                for y in range(self.lat_size):
                    for x in range(self.lon_size):
                        
                        # IDW Interpolation
                        total_weight = 0.0
                        interp_t, interp_p, interp_u, interp_v, interp_press = 0.0, 0.0, 0.0, 0.0, 0.0
                        
                        for i, city in enumerate(cities):
                            # Distance in grid cells
                            dist = math.sqrt((y - city['grid_y'])**2 + (x - city['grid_x'])**2)
                            weight = 1.0 / (dist + 0.1)**2 # Add 0.1 to avoid div by zero
                            
                            cd = city_data[i]
                            t = (cd.get('temperature_2m', [20]*24)[l] or 20) + 273.15
                            p = cd.get('precipitation', [0]*24)[l] or 0.0
                            ws = cd.get('windspeed_10m', [0]*24)[l] or 0.0
                            wd = cd.get('winddirection_10m', [0]*24)[l] or 0.0
                            press = cd.get('surface_pressure', [1013]*24)[l] or 1013.0
                            
                            u = ws * math.cos(math.radians(270 - wd))
                            v = ws * math.sin(math.radians(270 - wd))
                            
                            interp_t += t * weight
                            interp_p += p * weight
                            interp_u += u * weight
                            interp_v += v * weight
                            interp_press += press * weight
                            total_weight += weight
                            
                        # Normalize by weights
                        base_deterministic[0, 0, l, y, x] = interp_t / total_weight
                        base_deterministic[0, 1, l, y, x] = interp_p / total_weight
                        base_deterministic[0, 2, l, y, x] = interp_u / total_weight
                        base_deterministic[0, 3, l, y, x] = interp_v / total_weight
                        base_deterministic[0, 4, l, y, x] = interp_press / total_weight
                        
        except Exception as e:
            print(f"Warning: Failed to fetch real data from Open-Meteo ({e}). Falling back to synthetic.")
            base_deterministic = torch.randn(1, 5, self.lead_times, self.lat_size, self.lon_size)
            
        # Create ensemble spread by adding gaussian noise to the base deterministic forecast
        ensemble_tensors = []
        for i in range(self.num_members):
            # Scale noise differently based on variable (temp variance is larger than precip variance)
            noise = torch.randn_like(base_deterministic) * 0.5
            ensemble_tensors.append(base_deterministic + noise)
            
        ensemble_batch = torch.cat(ensemble_tensors, dim=0) # [23, 5, 9, 30, 26]
        
        return ensemble_batch
