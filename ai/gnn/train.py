import torch
import torch.nn as nn
import torch.optim as optim
import logging
from model import SphericalGNN
from graph_builder import SphericalGraphBuilder
from dataset import get_dataloader
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_prototype():
    logger.info("Initializing PROTOTYPE Spherical GNN training pipeline...")
    logger.warning("WARNING: Training on SYNTHETIC data. This is NOT an operational model.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Graph Construction
    lats = np.linspace(-5, 40, 30)
    lons = np.linspace(60, 100, 26)
    builder = SphericalGraphBuilder(lats, lons)
    edge_index, edge_weight = builder.get_graph_tensors()
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # 2. Model Initialization
    model = SphericalGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy for probability output
    
    # 3. Dataloader
    dataloader = get_dataloader(batch_size=4)
    
    # 4. Training Loop (Mock)
    epochs = 2
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            preds = model(x, edge_index, edge_weight)
            
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
    logger.info("Prototype training completed.")
    
if __name__ == "__main__":
    train_prototype()
