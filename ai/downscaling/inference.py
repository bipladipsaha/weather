import torch
import torch.nn.functional as F
import datetime

class DownscalingInference:
    def __init__(self, model, noise_scheduler, normalizer=None, device="cuda"):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = normalizer
        self.device = device
        
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)
            
    def bilinear_baseline(self, coarse_forecast, target_h=72, target_w=62):
        """
        Provides the baseline interpolation.
        coarse_forecast: [B, C, T, H12, W12]
        """
        B, C, T, H, W = coarse_forecast.shape
        # MUST permute B, T, C before reshaping to B*T to prevent scrambling channels and time
        coarse_2d = coarse_forecast.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        up_2d = F.interpolate(coarse_2d, size=(target_h, target_w), mode='bilinear', align_corners=False)
        up = up_2d.view(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)
        
        return up
        
    @torch.no_grad()
    def downscale(self, raw_coarse_forecast, conditioning, target_shape=(451, 401), target_mode="absolute", coarse_normalizer=None):
        """
        Genuine DDPM reverse inference loop.
        raw_coarse_forecast must be unnormalized!
        """
        if self.model is None:
            raise RuntimeError("Model not loaded for inference.")
            
        B, C, T, H_c, W_c = raw_coarse_forecast.shape
        H_f, W_f = target_shape
        
        # 1. Compute baseline if in residual mode
        if target_mode == "residual":
            baseline = self.bilinear_baseline(raw_coarse_forecast, H_f, W_f)
            
        # 2. Normalize coarse forecast for conditioning
        if coarse_normalizer is not None:
            coarse_input = coarse_normalizer.transform(raw_coarse_forecast)
        elif self.normalizer is not None and target_mode == "absolute":
            coarse_input = self.normalizer.transform(raw_coarse_forecast)
        else:
            coarse_input = raw_coarse_forecast
        
        # Start from pure noise
        x = torch.randn(B, C, T, H_f, W_f, device=self.device)
        
        # Reverse process
        alphas = self.noise_scheduler.alphas
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        
        # Accelerated 20-step inference
        num_inference_steps = 20
        step_ratio = self.noise_scheduler.num_train_timesteps // num_inference_steps
        
        for i in reversed(range(num_inference_steps)):
            t = i * step_ratio
            time_steps = torch.full((B,), t, device=self.device, dtype=torch.long)
            
            with torch.autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu', dtype=torch.float16):
                noise_pred = self.model(x, time_steps, coarse_input, conditioning)
                
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]
            
            if i > 0:
                noise = torch.randn_like(x)
                prev_t = (i - 1) * step_ratio
                alpha_bar_prev = alphas_cumprod[prev_t]
                sigma_t = torch.sqrt((1 - alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t))
            else:
                noise = torch.zeros_like(x)
                sigma_t = 0
                
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + sigma_t * noise
            
        # Denormalize
        if self.normalizer is not None:
            x = self.normalizer.inverse_transform(x)
            
        # Reconstruct and safeguard
        if target_mode == "residual":
            x = baseline + x
            # Physical safeguard: TP cannot be negative (assuming TP is channel 0 based on variables list ['tp', 'u10', 'd2m', 't2m'])
            x[:, 0] = torch.clamp(x[:, 0], min=0.0)
            
        uncertainty = torch.zeros_like(x) # Placeholder for ensemble spread later
        
        metadata = {
            "input_resolution": f"{H_c}x{W_c}",
            "output_resolution": f"{H_f}x{W_f}",
            "model_version": "28 km -> 11 km ERA5/ERA5-Land prototype",
            "conditioning_used": ["coarse_meteorological", "terrain", "event_mask"],
            "training_status": "TRAINED",
            "timestamp": str(datetime.datetime.now())
        }
        
        return x, uncertainty, metadata
