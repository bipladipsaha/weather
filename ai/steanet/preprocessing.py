import xarray as xr
import numpy as np

# Exact 30 channels expected by the model
CHANNEL_NAMES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "geopotential_1000hPa",
    "geopotential_850hPa",
    "geopotential_700hPa",
    "geopotential_500hPa",
    "geopotential_300hPa",
    "temperature_1000hPa",
    "temperature_850hPa",
    "temperature_700hPa",
    "temperature_500hPa",
    "temperature_300hPa",
    "u_component_of_wind_1000hPa",
    "u_component_of_wind_850hPa",
    "u_component_of_wind_700hPa",
    "u_component_of_wind_500hPa",
    "u_component_of_wind_300hPa",
    "v_component_of_wind_1000hPa",
    "v_component_of_wind_850hPa",
    "v_component_of_wind_700hPa",
    "v_component_of_wind_500hPa",
    "v_component_of_wind_300hPa",
    "specific_humidity_1000hPa",
    "specific_humidity_850hPa",
    "specific_humidity_700hPa",
    "specific_humidity_500hPa",
    "specific_humidity_300hPa"
]

LEAD_HOURS = [0, 12, 24, 36, 48, 60, 72, 96, 120]

def preprocess_nwp_dataset(ds: xr.Dataset, climatology_ds=None, std_ds=None):
    """
    Transforms raw NWP xarray dataset into the [30, 9, 30, 26] tensor.
    
    Expected raw dataset dimensions:
    - time (initialization time)
    - step / lead_time (forecast hours)
    - latitude
    - longitude
    - level (for upper air)
    
    If climatology_ds and std_ds are not provided, this function will mock the standardization.
    """
    
    # 1. Spatial alignment (Lat: -5 to 40, Lon: 60 to 100)
    # Ensure correct ordering (North to South or South to North depending on model, usually N->S in ERA5)
    # STEA-Net was trained on WeatherBench2 which is South to North (-90 to 90)
    # Let's slice accordingly
    ds_cropped = ds.sel(latitude=slice(-5, 40), longitude=slice(60, 100))
    
    # 2. Temporal alignment
    if 'step' in ds_cropped.dims:
        ds_cropped = ds_cropped.sel(step=np.timedelta64(LEAD_HOURS, 'h'), method='nearest')
    
    # 3. Channel selection and renaming
    tensors = []
    
    for channel in CHANNEL_NAMES:
        # Determine variable and level
        if "_1000hPa" in channel:
            var = channel.replace("_1000hPa", "")
            da = ds_cropped[var].sel(level=1000)
        elif "_850hPa" in channel:
            var = channel.replace("_850hPa", "")
            da = ds_cropped[var].sel(level=850)
        elif "_700hPa" in channel:
            var = channel.replace("_700hPa", "")
            da = ds_cropped[var].sel(level=700)
        elif "_500hPa" in channel:
            var = channel.replace("_500hPa", "")
            da = ds_cropped[var].sel(level=500)
        elif "_300hPa" in channel:
            var = channel.replace("_300hPa", "")
            da = ds_cropped[var].sel(level=300)
        else:
            # Surface variables
            da = ds_cropped[channel]
            
        # Standardize (mocking for now if climatology/std are absent)
        if climatology_ds is not None and std_ds is not None:
            # Calculate standard anomaly
            # anomaly = (da - clim) / std
            pass
        else:
            # Mock normalization (just a placeholder so it runs)
            # In production, use the actual climatology files.
            da = (da - da.mean()) / (da.std() + 1e-6)
            
        tensors.append(da.values)
        
    # Stack into [Channels, Leads, Lat, Lon]
    # Handle single batch dimension if 'time' exists
    stacked = np.stack(tensors, axis=0)
    
    # Ensure shape is [1, 30, 9, 30, 26] if it was a single forecast
    if len(stacked.shape) == 4:
        stacked = np.expand_dims(stacked, axis=0)
    
    return stacked.astype(np.float32)
