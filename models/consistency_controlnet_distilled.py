import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_base import Unet
from models.unet_base import get_time_embedding
from models.controlnet import ControlNet
from copy import deepcopy


class ConsistencyControlNet(nn.Module):
    """
    Consistency Model for ControlNet
    """
    def __init__(self, model_config):
        super().__init__()
        
        # Main UNet for consistency prediction
        self.unet = Unet(model_config)
        
        # Hint processing block (similar to ControlNet)
        self.hint_block = nn.Sequential(
            nn.Conv2d(model_config['hint_channels'], 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, self.unet.down_channels[0], kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.unet.down_channels[0], 
                      self.unet.down_channels[0], 
                      kernel_size=1, padding=0)
        )
        
        # Time embedding for consistency model
        self.t_emb_dim = model_config['time_emb_dim']
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        # Consistency model parameters
        self.sigma_min = model_config.get('sigma_min', 0.002)
        self.sigma_max = model_config.get('sigma_max', 80.0)
        self.sigma_data = model_config.get('sigma_data', 0.5)
        
    def c_skip(self, sigma):
        """Skip connection scaling function"""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        
        sigma_data_tensor = torch.tensor(self.sigma_data, dtype=sigma.dtype, device=sigma.device)
        return sigma_data_tensor ** 2 / (sigma ** 2 + sigma_data_tensor ** 2)
    
    def c_out(self, sigma):
        """Output scaling function"""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        
        sigma_data_tensor = torch.tensor(self.sigma_data, dtype=sigma.dtype, device=sigma.device)
        return sigma * sigma_data_tensor / torch.sqrt(sigma ** 2 + sigma_data_tensor ** 2)
    
    def c_in(self, sigma):
        """Input scaling function"""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        
        sigma_data_tensor = torch.tensor(self.sigma_data, dtype=sigma.dtype, device=sigma.device)
        return 1.0 / torch.sqrt(sigma ** 2 + sigma_data_tensor ** 2)
    
    def c_noise(self, sigma):
        """Noise conditioning function"""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        
        return 0.25 * torch.log(sigma.clamp(min=1e-8))  # Prevent log(0)
        
    def forward(self, x_t, sigma, hint):
        """
        Forward pass for consistency model
        """
        # Handle boundary condition: f(x, σ_min) = x
        if torch.all(sigma <= self.sigma_min):
            return x_t
            
        # Ensure sigma is properly shaped
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
            
        # Apply input scaling
        c_in_val = self.c_in(sigma)
        x_scaled = c_in_val * x_t
        
        # Time embedding based on noise level
        c_noise_val = self.c_noise(sigma.squeeze())
        t_emb = get_time_embedding(
            (c_noise_val * 1000).long().clamp(0, 999), 
            self.t_emb_dim
        )
        t_emb = self.t_proj(t_emb)
        
        # Process hint
        hint_out = self.hint_block(hint)
        
        # Combine input with hint
        x_combined = self.unet.conv_in(x_scaled)
        x_combined += hint_out
        
        # Process through UNet
        down_outs = []
        for down in self.unet.downs:
            down_outs.append(x_combined)
            x_combined = down(x_combined, t_emb)
            
        for mid in self.unet.mids:
            x_combined = mid(x_combined, t_emb)
            
        for up in self.unet.ups:
            down_out = down_outs.pop()
            x_combined = up(x_combined, down_out, t_emb)
            
        # Output processing
        x_combined = self.unet.norm_out(x_combined)
        x_combined = nn.SiLU()(x_combined)
        F_theta = self.unet.conv_out(x_combined)
        
        # Apply consistency model parameterization
        c_skip_val = self.c_skip(sigma)
        c_out_val = self.c_out(sigma)
        
        # Final prediction: skip connection + scaled network output
        x_0_pred = c_skip_val * x_t + c_out_val * F_theta
        
        return x_0_pred


class ConsistencyControlNetDistilled(nn.Module):
    """
    Consistency Model distilled from pre-trained DDPM ControlNet
    """
    def __init__(self, model_config, teacher_ckpt_path=None, device=None):
        super().__init__()
        
        # Student model (consistency model)
        self.student = ConsistencyControlNet(model_config)
        
        # EMA teacher model (copy of student for consistency training)
        self.ema_teacher = deepcopy(self.student)
        self.ema_teacher.eval()
        
        # Optional: DDPM teacher for distillation
        self.ddpm_teacher = None
        if teacher_ckpt_path:
            # Use ControlNet's built-in smart loading (handles full ControlNet checkpoints correctly)
            self.ddpm_teacher = ControlNet(model_config, model_locked=True, 
                                         model_ckpt=teacher_ckpt_path, device=device)
            self.ddpm_teacher.eval()
            
            # Noise scheduler for DDPM teacher
            from scheduler.linear_noise_scheduler import LinearNoiseScheduler
            self.teacher_scheduler = LinearNoiseScheduler(
                num_timesteps=1000, beta_start=0.0001, beta_end=0.02
            )
        
        # Consistency model parameters
        self.sigma_min = model_config.get('sigma_min', 0.002)
        self.sigma_max = model_config.get('sigma_max', 80.0)
        self.num_timesteps = 1000
        self.ema_decay = 0.995
        
    def update_ema_teacher(self):
        """Update EMA teacher parameters"""
        with torch.no_grad():
            for ema_param, student_param in zip(self.ema_teacher.parameters(), 
                                               self.student.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(student_param.data, 
                                                        alpha=1 - self.ema_decay)
    
    def get_noise_schedule(self, num_steps, device=None):
        """Get noise schedule for consistency training"""
        if device is None:
            device = next(self.parameters()).device
            
        # Karras et al. schedule
        rho = 7.0
        steps = torch.arange(num_steps, dtype=torch.float32, device=device)
        
        sigma_min_tensor = torch.tensor(self.sigma_min, device=device)
        sigma_max_tensor = torch.tensor(self.sigma_max, device=device)
        
        sigmas = sigma_min_tensor ** (1/rho) + steps / (num_steps - 1) * (
            sigma_max_tensor ** (1/rho) - sigma_min_tensor ** (1/rho)
        )
        sigmas = sigmas ** rho
        
        return sigmas
        
    def forward(self, x_t, sigma, hint):
        return self.student(x_t, sigma, hint)
    
    def get_ddpm_teacher_prediction(self, x_t, sigma, hint):
        """Get DDPM teacher's x_0 prediction"""
        if self.ddpm_teacher is None:
            raise ValueError("DDPM teacher not initialized")
            
        with torch.no_grad():
            # Convert sigma to timestep for DDPM teacher
            t = self.sigma_to_timestep(sigma.squeeze())
            
            # Teacher predicts noise
            noise_pred = self.ddpm_teacher(x_t, t, hint)
            
            # Convert noise prediction to x_0 prediction
            batch_size = x_t.shape[0]
            
            if t.dim() == 0:
                t = t.unsqueeze(0)
            
            sqrt_one_minus_alpha = self.teacher_scheduler.sqrt_one_minus_alpha_cum_prod.to(x_t.device)[t]
            sqrt_alpha = self.teacher_scheduler.sqrt_alpha_cum_prod.to(x_t.device)[t]
            
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(batch_size, 1, 1, 1)
            sqrt_alpha = sqrt_alpha.reshape(batch_size, 1, 1, 1)
            
            x_0_pred = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
            x_0_pred = torch.clamp(x_0_pred, -1., 1.)
            
        return x_0_pred
    
    def sigma_to_timestep(self, sigma):
        """Convert continuous noise level to discrete timestep"""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        
        # Ensure sigma is on the same device as the model
        device = next(self.parameters()).device
        sigma = sigma.to(device)
        
        # Use the EXACT same noise schedule as the teacher
        if hasattr(self, 'teacher_scheduler'):
            # Use teacher's pre-computed schedule for better alignment
            alphas_cumprod = self.teacher_scheduler.alpha_cum_prod.to(device)
            sigma_schedule = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        else:
            # Fallback to manual computation (should match teacher exactly)
            betas = torch.linspace(0.0001, 0.02, self.num_timesteps, device=device)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sigma_schedule = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        
        # Find closest match - ensure both tensors are on same device
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        
        distances = torch.abs(sigma_schedule.unsqueeze(0) - sigma.unsqueeze(-1))
        t = torch.argmin(distances, dim=-1)
        
        return t.long().clamp(0, self.num_timesteps - 1)
    
    def consistency_training_loss(self, x_0, hint, sigma_1, sigma_2):
        """
        Proper consistency training loss
        Args:
            x_0: Clean image
            hint: Control signal  
            sigma_1: Smaller noise level
            sigma_2: Larger noise level
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Ensure sigma_1 < sigma_2
        if torch.any(sigma_1 >= sigma_2):
            sigma_1, sigma_2 = torch.minimum(sigma_1, sigma_2), torch.maximum(sigma_1, sigma_2)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Create noisy versions
        x_sigma1 = x_0 + sigma_1.view(-1, 1, 1, 1) * noise
        x_sigma2 = x_0 + sigma_2.view(-1, 1, 1, 1) * noise
        
        # EMA teacher prediction at sigma_1 (target)
        with torch.no_grad():
            x_0_target = self.ema_teacher(x_sigma1, sigma_1, hint)
            
        # Student prediction at sigma_2
        x_0_pred = self.student(x_sigma2, sigma_2, hint)
        
        # Consistency loss
        loss = F.mse_loss(x_0_pred, x_0_target)
        
        return loss
    
    def distillation_loss(self, x_0, hint, sigma, alpha=0.5, epoch=None, total_epochs=None):
        """
        Combined consistency + distillation loss with progressive weighting
        """
        if self.ddpm_teacher is None:
            raise ValueError("DDPM teacher required for distillation")
            
        batch_size = x_0.shape[0]
        noise = torch.randn_like(x_0)
        
        # Create noisy image
        x_t = x_0 + sigma.view(-1, 1, 1, 1) * noise
        
        # Student prediction
        x_0_student = self.student(x_t, sigma, hint)
        
        # DDPM teacher prediction
        x_0_teacher = self.get_ddpm_teacher_prediction(x_t, sigma, hint)
        
        # Reconstruction loss (consistency with ground truth)
        recon_loss = F.mse_loss(x_0_student, x_0)
        
        # Distillation loss (consistency with teacher)
        distill_loss = F.mse_loss(x_0_student, x_0_teacher)
        
        # Progressive training: start with more teacher guidance, gradually emphasize ground truth
        if epoch is not None and total_epochs is not None:
            # Start with high distillation weight, gradually decrease
            progress = epoch / total_epochs
            dynamic_alpha = alpha * (1 - progress) + 0.1 * progress  # Don't go to 0
            alpha = max(dynamic_alpha, 0.1)  # Minimum 10% distillation
        
        # Combined loss
        total_loss = alpha * recon_loss + (1 - alpha) * distill_loss
        
        return total_loss, recon_loss, distill_loss
    
    def sample_sigmas(self, batch_size, device):
        """Sample noise levels for training"""
        # Sample from log-normal distribution
        log_sigma_min = torch.log(torch.tensor(self.sigma_min, device=device))
        log_sigma_max = torch.log(torch.tensor(self.sigma_max, device=device))
        
        u = torch.rand(batch_size, device=device)
        log_sigma = log_sigma_min + u * (log_sigma_max - log_sigma_min)
        
        return torch.exp(log_sigma)
    
    def training_step(self, x_0, hint, use_ddpm_teacher=False):
        """
        Complete training step
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        if use_ddpm_teacher and self.ddpm_teacher is not None:
            # Distillation training
            sigma = self.sample_sigmas(batch_size, device)
            loss, recon_loss, distill_loss = self.distillation_loss(x_0, hint, sigma)
            
            # Update EMA teacher
            self.update_ema_teacher()
            
            return {
                'total_loss': loss,
                'recon_loss': recon_loss,
                'distill_loss': distill_loss
            }
        else:
            # Pure consistency training
            sigma_1 = self.sample_sigmas(batch_size, device)
            sigma_2 = self.sample_sigmas(batch_size, device)
            
            loss = self.consistency_training_loss(x_0, hint, sigma_1, sigma_2)
            
            # Update EMA teacher
            self.update_ema_teacher()
            
            return {'consistency_loss': loss}
    
    def generate(self, hint, shape, num_steps=1, guidance_scale=1.0):
        """
        Generate samples using the consistency model
        """
        device = next(self.parameters()).device
        
        if num_steps == 1:
            # Single-step generation
            x_T = torch.randn(shape, device=device)
            sigma = torch.full((shape[0],), self.sigma_max, device=device)
            
            with torch.no_grad():
                x_0 = self.student(x_T, sigma, hint)
                
            return x_0
        else:
            # Multi-step generation
            sigmas = self.get_noise_schedule(num_steps + 1, device)
            x = torch.randn(shape, device=device)
            
            with torch.no_grad():
                for i in range(num_steps):
                    sigma = sigmas[i]
                    sigma_next = sigmas[i + 1]
                    
                    # Consistency model prediction
                    x_0 = self.student(x, sigma, hint)
                    
                    if i < num_steps - 1:
                        # Add noise for next step
                        noise = torch.randn_like(x)
                        x = x_0 + sigma_next * noise
                    else:
                        x = x_0
                        
            return x