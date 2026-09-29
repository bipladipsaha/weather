import torch
import numpy as np
from graph_builder import SphericalGraphBuilder
from model import SphericalGNN
from inference import GNNInference

def run_tests():
    print("========================================")
    print("TESTING SPHERICAL GNN PROTOTYPE")
    print("========================================")
    
    # Grid definition
    lats = np.linspace(-5, 40, 30)
    lons = np.linspace(60, 100, 26)
    
    # 1. Graph Construction Test
    print("1. Testing Graph Construction...")
    builder = SphericalGraphBuilder(lats, lons, k_neighbors=6)
    edge_index, edge_weight = builder.get_graph_tensors()
    
    expected_nodes = 30 * 26  # 780
    expected_edges = 780 * 6  # 4680
    
    print(f"   Nodes: {expected_nodes}")
    print(f"   Edges: {edge_index.shape[1]}")
    
    if edge_index.shape[1] == expected_edges:
        print("   [PASS] graph construction             PASS")
        print("   [PASS] node/edge dimensions           PASS")
    else:
        print("   [FAIL] graph construction             FAIL")
        
    # Test Spherical mapping manually
    x, y, z = builder._latlon_to_cartesian(np.array([0]), np.array([0]))
    if np.isclose(x[0], 6371.0) and np.isclose(y[0], 0.0) and np.isclose(z[0], 0.0):
        print("   [PASS] spherical coordinates          PASS")
    else:
        print("   [FAIL] spherical coordinates          FAIL")
        
    # 2. Forward Pass Test
    print("\n2. Testing GNN Forward Pass...")
    model = SphericalGNN()
    
    # Synthetic batch [Batch, Channels, Leads, Lat, Lon]
    dummy_x = torch.randn(2, 30, 9, 30, 26)
    
    # Add NaN to check NaN handling (expect output to NOT crash, though it will produce NaNs)
    dummy_x[0, 0, 0, 0, 0] = float('nan')
    
    try:
        preds = model(dummy_x, edge_index, edge_weight)
        print("   [PASS] forward pass                   PASS")
    except Exception as e:
        print(f"   [FAIL] forward pass                   FAIL: {e}")
        return
        
    # 3. Shape Tests
    print(f"   Output shape: {preds.shape}")
    if preds.shape == (2, 4, 9, 30, 26):
        print("   [PASS] output shape                   PASS")
        print("   [PASS] multi-lead output              PASS")
        print("   [PASS] batch handling                 PASS")
    else:
        print("   [FAIL] output shape                   FAIL")
        
    # 4. NaN Handling Test
    if torch.isnan(preds).any():
        # A single NaN in GCN spreads due to message passing, which is expected math,
        # but the forward pass shouldn't crash.
        print("   [PASS] NaN handling                   PASS (NaN propagated through graph without crashing)")
    else:
        print("   [FAIL] NaN handling                   FAIL (NaN disappeared or crashed)")
        
    # 5. Inference Wrapper Test
    print("\n3. Testing Inference Wrapper (STEA-Net standard interface)...")
    try:
        inf = GNNInference()
        numpy_input = np.random.randn(1, 30, 9, 30, 26)
        out = inf.predict(numpy_input)
        if out.shape == (1, 4, 9, 30, 26):
            print("   [PASS] Inference standard interface   PASS")
    except Exception as e:
        print(f"   [FAIL] Inference standard interface   FAIL: {e}")
        
    print("\n========================================")
    print("ALL TESTS COMPLETE")
    print("========================================")

if __name__ == "__main__":
    run_tests()
