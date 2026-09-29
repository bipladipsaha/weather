import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
import argparse
import logging
import json

from diffusion import SmallUNet, ConditionalDiffusionModel
from conditioning import ConditionEncoder
from train import ExtremeAwareLoss
from normalization import DataNormalizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.data.paired_dataset import PairedERA5Dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def log_memory_stats(prefix=""):
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    logger.info(f"{prefix} GPU Mem - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Peak: {peak:.2f} MB")

class DDPMNoiseScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        B = original_samples.shape[0]
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(B, 1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(B, 1, 1, 1, 1)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

def run_single_batch_test(model, criterion, optimizer, scaler, dataloader, device, noise_scheduler):
    logger.info("=== Starting 1-Batch Test (28km -> 11km real-data diffusion prototype) ===")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.reset_peak_memory_stats()
    log_memory_stats("Before Batch:")
    
    batch = next(iter(dataloader))
    coarse = batch['coarse'].to(device, non_blocking=True)
    fine = batch['fine'].to(device, non_blocking=True)
    cond = batch['conditioning'].to(device, non_blocking=True)
    target_mask = batch['target_mask'].to(device, non_blocking=True)
    
    B = coarse.shape[0]
    mask_pct = (target_mask.float().mean().item()) * 100
    
    logger.info(f"Input coarse shape: {coarse.shape}")
    logger.info(f"Target fine shape: {fine.shape}")
    logger.info(f"Mask percentage (valid): {mask_pct:.2f}%")
    
    # NaN masking before network forward pass
    fine = torch.where(target_mask, fine, torch.zeros_like(fine))
    
    time_steps = torch.randint(0, noise_scheduler.num_train_timesteps, (B,), device=device)
    logger.info(f"Sampled timestep(s): {time_steps.tolist()}")
    
    noise = torch.randn_like(fine)
    noise = torch.where(target_mask, noise, torch.zeros_like(noise))
    
    noisy_target = noise_scheduler.add_noise(fine, noise, time_steps)
    
    torch.cuda.synchronize()
    start_fw = time.perf_counter()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        noise_pred = model(noisy_target, time_steps, coarse, cond)
        # We pass target_mask to ExtremeAwareLoss so it ignores NaNs/invalid ocean pixels
        loss = criterion(noise_pred, noise, target_mask=target_mask)
        
    torch.cuda.synchronize()
    end_fw = time.perf_counter()
    
    start_bw = time.perf_counter()
    scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    grads_finite = True
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            grads_finite = False
            break
            
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    end_bw = time.perf_counter()
    
    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Forward time: {end_fw - start_fw:.4f} s")
    logger.info(f"Backward time: {end_bw - start_bw:.4f} s")
    logger.info(f"Gradients finite: {grads_finite}")
    
    log_memory_stats("After Batch:")
    
    del coarse, fine, cond, target_mask, noisy_target, noise, noise_pred, loss
    torch.cuda.empty_cache()
    
    logger.info("=== Single Batch Test Complete ===")
    return True

def train(epochs, batch_size=1, accumulation_steps=4, lr=1e-4, checkpoint_dir="checkpoints/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    unet = SmallUNet(c_in=4, c_cond=32, c_out=4)
    cond_encoder = ConditionEncoder(c_in=4, cond_in=2, hidden=32)
    model = ConditionalDiffusionModel(unet, cond_encoder).to(device)
    
    criterion = ExtremeAwareLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    noise_scheduler = DDPMNoiseScheduler(device=device)
    
    era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
    era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
    variables = ['tp', 'u10', 'd2m', 't2m']
    
    # Real Dataset with Normalization
    abs_stats_path = os.path.join(os.path.dirname(__file__), "normalization_stats.json")
    res_stats_path = os.path.join(os.path.dirname(__file__), "residual_normalization_stats.json")
    
    coarse_normalizer = DataNormalizer(variables)
    if os.path.exists(abs_stats_path):
        coarse_normalizer.load(abs_stats_path)
        
    residual_normalizer = DataNormalizer(variables)
    
    # We need to build the train dataset first without normalizer to fit the residual normalizer if needed
    train_ds_raw = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="train", target_mode="residual")
    
    if os.path.exists(res_stats_path):
        residual_normalizer.load(res_stats_path)
    else:
        logger.info("Computing residual normalization stats...")
        residual_normalizer.fit(train_ds_raw, list(range(len(train_ds_raw))))
        residual_normalizer.save(res_stats_path)
        
    train_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="train", target_mode="residual", normalizer=residual_normalizer, coarse_normalizer=coarse_normalizer)
    val_ds = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="val", target_mode="residual", normalizer=residual_normalizer, coarse_normalizer=coarse_normalizer)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    best_val_loss = float('inf')
    early_stop_patience = 10
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        logger.info(f"--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            coarse = batch['coarse'].to(device, non_blocking=True)
            fine = batch['fine'].to(device, non_blocking=True)
            cond = batch['conditioning'].to(device, non_blocking=True)
            target_mask = batch['target_mask'].to(device, non_blocking=True)
            
            fine = torch.where(target_mask, fine, torch.zeros_like(fine))
            B = coarse.shape[0]
            time_steps = torch.randint(0, noise_scheduler.num_train_timesteps, (B,), device=device)
            noise = torch.randn_like(fine)
            noise = torch.where(target_mask, noise, torch.zeros_like(noise))
            noisy_target = noise_scheduler.add_noise(fine, noise, time_steps)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                noise_pred = model(noisy_target, time_steps, coarse, cond)
                loss = criterion(noise_pred, noise, target_mask=target_mask)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
            del coarse, fine, cond, target_mask, noisy_target, noise, noise_pred, loss
            
        train_loss /= len(train_loader)
        torch.cuda.empty_cache()
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                coarse = batch['coarse'].to(device, non_blocking=True)
                fine = batch['fine'].to(device, non_blocking=True)
                cond = batch['conditioning'].to(device, non_blocking=True)
                target_mask = batch['target_mask'].to(device, non_blocking=True)
                
                fine = torch.where(target_mask, fine, torch.zeros_like(fine))
                B = coarse.shape[0]
                time_steps = torch.randint(0, noise_scheduler.num_train_timesteps, (B,), device=device)
                noise = torch.randn_like(fine)
                noise = torch.where(target_mask, noise, torch.zeros_like(noise))
                noisy_target = noise_scheduler.add_noise(fine, noise, time_steps)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    noise_pred = model(noisy_target, time_steps, coarse, cond)
                    v_loss = criterion(noise_pred, noise, target_mask=target_mask)
                    
                val_loss += v_loss.item()
                del coarse, fine, cond, target_mask, noisy_target, noise, noise_pred, v_loss
                
        val_loss /= len(val_loader)
        torch.cuda.empty_cache()
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved new best model to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info("Early stopping triggered!")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["batch_test", "epoch_test", "full_train"], required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "batch_test":
        unet = SmallUNet(c_in=4, c_cond=32, c_out=4)
        cond_encoder = ConditionEncoder(c_in=4, cond_in=2, hidden=32)
        model = ConditionalDiffusionModel(unet, cond_encoder).to(device)
        criterion = ExtremeAwareLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        noise_scheduler = DDPMNoiseScheduler(device=device)
        
        era5_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5'))
        era5_land_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_era5_land'))
        variables = ['tp', 'u10', 'd2m', 't2m']
        
        abs_stats_path = os.path.join(os.path.dirname(__file__), "normalization_stats.json")
        res_stats_path = os.path.join(os.path.dirname(__file__), "residual_normalization_stats.json")
        
        coarse_normalizer = DataNormalizer(variables)
        if os.path.exists(abs_stats_path):
            coarse_normalizer.load(abs_stats_path)
            
        residual_normalizer = DataNormalizer(variables)
        if os.path.exists(res_stats_path):
            residual_normalizer.load(res_stats_path)
        
        dataset = PairedERA5Dataset(era5_dir, era5_land_dir, variables, sequence_length=9, split="train", target_mode="residual", normalizer=residual_normalizer, coarse_normalizer=coarse_normalizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
        
        run_single_batch_test(model, criterion, optimizer, scaler, dataloader, device, noise_scheduler)
        
    elif args.mode == "epoch_test":
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "residual_diffusion")
        train(epochs=1, batch_size=1, accumulation_steps=4, checkpoint_dir=ckpt_dir)
        
    elif args.mode == "full_train":
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "residual_diffusion")
        train(epochs=100, batch_size=1, accumulation_steps=4, checkpoint_dir=ckpt_dir)
