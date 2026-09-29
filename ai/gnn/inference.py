import torch
import numpy as np
from model import SphericalGNN
from graph_builder import SphericalGraphBuilder

class GNNInference:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Grid parameters exactly matching STEA-Net baseline
        self.lats = np.linspace(-5, 40, 30)
        self.lons = np.linspace(60, 100, 26)
        
        # Build Graph once for inference
        builder = SphericalGraphBuilder(self.lats, self.lons)
        self.edge_index, self.edge_weight = builder.get_graph_tensors()
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        
        self.model = SphericalGNN().to(self.device)
        
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            
        self.model.eval()
        
    def predict(self, input_tensor):
        """
        Expects numpy array [B, 30, 9, 30, 26]
        """
        # Handle single unbatched inputs [30, 9, 30, 26]
        if len(input_tensor.shape) == 4:
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
        x = torch.tensor(input_tensor, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(x, self.edge_index, self.edge_weight)
            
        return preds.cpu().numpy()
