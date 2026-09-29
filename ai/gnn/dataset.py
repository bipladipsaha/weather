import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticWeatherDataset(Dataset):
    """
    A placeholder dataset for prototype validation.
    Generates synthetic [B, 30, 9, 30, 26] tensors mimicking the 
    STEA-Net standard preprocessing output.
    """
    def __init__(self, num_samples=100, lat_size=30, lon_size=26, channels=30, leads=9):
        self.num_samples = num_samples
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.channels = channels
        self.leads = leads
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Synthetic NWP features + Anomaly/EFI Features
        x = np.random.randn(self.channels, self.leads, self.lat_size, self.lon_size).astype(np.float32)
        
        # Synthetic target probabilities [4 classes, 9 leads, 30 lat, 26 lon]
        y = np.random.rand(4, self.leads, self.lat_size, self.lon_size).astype(np.float32)
        
        return torch.tensor(x), torch.tensor(y)

def get_dataloader(batch_size=4):
    ds = SyntheticWeatherDataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
