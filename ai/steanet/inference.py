import torch
import numpy as np
from model import STEANet

class STEAInference:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = STEANet(in_channels=30, num_classes=4).to(self.device)
        
        if checkpoint_path:
            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # Sometimes models are saved with an outer dictionary like {"model_state": ...}
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict)
            
        self.model.eval()
        
    def predict(self, input_tensor):
        """
        Expects a normalized float32 tensor or numpy array of shape [B, 30, 9, 30, 26]
        Channels must be strictly aligned with the training metadata (2m_temperature, 10m_wind, etc.)
        """
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).float()
            
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            # STEANet uses BCEWithLogitsLoss, so we apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits)
            
        return probabilities.cpu().numpy()
