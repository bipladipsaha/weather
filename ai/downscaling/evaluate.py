import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time

from diffusion import DummyUNet, ConditionalDiffusionModel
from conditioning import ConditionEncoder
from dataset import DownscalingDataset
from inference import DownscalingInference

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Load Model
    unet = DummyUNet(c_in=4, c_cond=32, c_out=4)
    cond_encoder = ConditionEncoder(c_in=4, cond_in=2, hidden=32)
    model = ConditionalDiffusionModel(unet, cond_encoder).to(device)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded trained model checkpoint successfully.")
    else:
        print("WARNING: Checkpoint not found. Evaluating untrained model.")

    model.eval()
    
    # 38-sample unseen test set
    test_ds = DownscalingDataset(num_samples=38)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    infer = DownscalingInference()
    
    bilinear_mse = 0.0
    diffusion_mse = 0.0
    
    start_time = time.time()
    
    with torch.no_grad():
        for coarse, cond, target in test_loader:
            coarse = coarse.to(device)
            cond = cond.to(device)
            target = target.to(device)
            B, C, T, H12, W12 = coarse.shape
            
            # 1. Bilinear Baseline
            bilinear_out = infer.bilinear_baseline(coarse, target_h=72, target_w=62)
            bilinear_mse += F.mse_loss(bilinear_out, target).item()
            
            # 2. Diffusion Inference (Mock 1-step based on our training setup)
            # Starting point: bilinear upsampled + synthetic noise scale 0.1
            noisy_start = bilinear_out + torch.randn_like(target) * 0.1
            time_steps = torch.zeros(B, dtype=torch.long, device=device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                noise_pred = model(noisy_start, time_steps, coarse, cond)
            
            # Recover clean prediction
            diffusion_out = noisy_start - noise_pred * 0.1
            
            # Clamp to valid range (0-1) as in dataset
            diffusion_out = torch.clamp(diffusion_out, 0.0, 1.0)
            
            diffusion_mse += F.mse_loss(diffusion_out, target).item()
            
    bilinear_mse /= len(test_loader)
    diffusion_mse /= len(test_loader)
    
    print("\n" + "="*40)
    print("      EVALUATION RESULTS (38 Samples)      ")
    print("="*40)
    print(f"Bilinear Baseline MSE : {bilinear_mse:.6f}")
    print(f"Trained Diffusion MSE : {diffusion_mse:.6f}")
    print(f"Improvement over Base : {(bilinear_mse - diffusion_mse) / bilinear_mse * 100:.2f}%")
    print(f"Total evaluation time : {time.time() - start_time:.2f}s")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()
