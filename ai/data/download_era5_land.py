import cdsapi
import os

def download_era5_land(years, output_dir):
    """
    Downloads ERA5-Land data for the specified years.
    Variables are matched to the coarse ERA5 dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = "reanalysis-era5-land"
    
    client = cdsapi.Client()
    
    for year in years:
        output_file = os.path.join(output_dir, f"era5_land_{year}_01.nc")
        print(f"Queueing download for {year} -> {output_file}")
        
        request = {
            "variable": [
                "2m_dewpoint_temperature",
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "total_precipitation"
            ],
            "year": str(year),
            "month": "01",
            "day": [
                "01", "02", "03", "04", "05", "06", 
                "07", "08", "09", "10", "11", "12", 
                "13", "14", "15", "16", "17", "18", 
                "19", "20", "21", "22", "23", "24", 
                "25", "26", "27", "28", "29", "30", "31"
            ],
            "time": [
                "00:00", "06:00", "12:00", "18:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [40, 60, -5, 100]
        }
        
        # Download the file
        client.retrieve(dataset, request, output_file)
        print(f"Successfully downloaded {year}!")

if __name__ == "__main__":
    # We need 2018-2026 to match your ERA5 coarse data
    years_to_download = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    
    # Save them to the ai/data/raw directory
    out_dir = os.path.join(os.path.dirname(__file__), "raw_era5_land")
    
    print("Starting ERA5-Land bulk download...")
    download_era5_land(years_to_download, out_dir)
    print("All downloads complete!")
