import torch
import numpy as np
import os
from inference import STEAInference

def run_test():
    print("=" * 60)
    print("TESTING STEA-NET BASELINE INFERENCE")
    print("=" * 60)
    
    # Checkpoint path
    checkpoint_path = "../../final_model/model/stea_net_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running test with randomly initialized weights.")
        checkpoint_path = None
    else:
        print(f"Loading checkpoint from: {checkpoint_path}")

    # Initialize inference
    try:
        inference_engine = STEAInference(checkpoint_path=checkpoint_path)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    # Generate dummy input matching [B, Channels, Leads, Lat, Lon]
    # Channels: 30
    # Leads: 9
    # Lat: 30
    # Lon: 26
    print("\nGenerating dummy NWP input tensor [1, 30, 9, 30, 26]...")
    dummy_input = np.random.randn(1, 30, 9, 30, 26).astype(np.float32)
    
    # Run prediction
    print("Running forward pass...")
    try:
        probabilities = inference_engine.predict(dummy_input)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return
        
    print("\nInference successful!")
    print(f"Output shape: {probabilities.shape}")
    print("Expected shape: (1, 4, 9, 30, 26)")
    
    assert probabilities.shape == (1, 4, 9, 30, 26), "Output shape mismatch!"
    
    print("\nOutput summary:")
    print(f"Min probability: {probabilities.min():.4f}")
    print(f"Max probability: {probabilities.max():.4f}")
    print(f"Mean probability: {probabilities.mean():.4f}")
    
    print("\nChannel mapping documentation:")
    print("0 = Heat probability map")
    print("1 = Cold probability map")
    print("2 = Rain probability map")
    print("3 = Cyclone probability map")
    
    print("\nGrid documentation:")
    print("Latitude: -5 to 40 (30 grid points)")
    print("Longitude: 60 to 100 (26 grid points)")
    print("Leads (hours): 0, 12, 24, 36, 48, 60, 72, 96, 120 (9 steps)")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
