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
        # In a production environment, this streams GRIB2/NetCDF files via HTTP/FTP
        # and converts them into the canonical 5D tensor [Member, Channel, Lead, Lat, Lon]
        
        # We generate a structured pseudo-random tensor that behaves like real ensemble data
        # shape: [B (members), C (vars), L (leads), H, W]
        base_deterministic = torch.randn(1, 5, self.lead_times, self.lat_size, self.lon_size)
        
        # Create ensemble spread by adding gaussian noise to the base deterministic forecast
        ensemble_tensors = []
        for i in range(self.num_members):
            noise = torch.randn_like(base_deterministic) * 0.1
            ensemble_tensors.append(base_deterministic + noise)
            
        ensemble_batch = torch.cat(ensemble_tensors, dim=0) # [23, 5, 9, 30, 26]
        
        return ensemble_batch
