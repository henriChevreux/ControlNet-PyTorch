import torch
import torch.nn as nn
from models.unet_base import Unet
from models.unet_base import get_time_embedding
from models.controlnet import ControlNet


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
        
    def forward(self, x_t, t, hint):
        """
        Forward pass for consistency model
        Args:
            x_t: Noisy image at timestep t
            t: Timestep (scalar or tensor)
            hint: Control signal (Canny edges)
        Returns:
            x_0_pred: Predicted clean image
        """
        # Time embedding
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        # Process hint
        hint_out = self.hint_block(hint)
        
        # Combine input with hint
        x_combined = self.unet.conv_in(x_t)
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
        x_0_pred = self.unet.conv_out(x_combined)
        
        return x_0_pred


class ConsistencyControlNetDistilled(nn.Module):
    """
    Consistency Model distilled from pre-trained DDPM ControlNet
    """
    def __init__(self, model_config, teacher_ckpt_path, device=None):
        super().__init__()
        
        # Student model (consistency model)
        self.student = ConsistencyControlNet(model_config)
        
        # Teacher model (pre-trained DDPM ControlNet)
        self.teacher = ControlNet(model_config, model_locked=True, 
                                 model_ckpt=teacher_ckpt_path, device=device)
        self.teacher.eval()  # Freeze teacher
        
        # Noise scheduler for teacher
        from scheduler.linear_noise_scheduler import LinearNoiseScheduler
        self.teacher_scheduler = LinearNoiseScheduler(
            num_timesteps=1000, beta_start=0.0001, beta_end=0.02
        )
        
    def forward(self, x_t, t, hint):
        return self.student(x_t, t, hint)
    
    def get_teacher_prediction(self, x_t, t, hint):
        """Get teacher's noise prediction and convert to x_0"""
        with torch.no_grad():
            # Teacher predicts noise
            noise_pred = self.teacher(x_t, t, hint)
            
            # Convert noise prediction to x_0 prediction
            # Fix the tensor reshaping issue
            batch_size = x_t.shape[0]
            
            # Reshape t to match the expected format
            if t.dim() == 0:
                t = t.unsqueeze(0)
            
            # Get the scheduler values and reshape properly
            sqrt_one_minus_alpha = self.teacher_scheduler.sqrt_one_minus_alpha_cum_prod.to(x_t.device)[t]
            sqrt_alpha = self.teacher_scheduler.sqrt_alpha_cum_prod.to(x_t.device)[t]
            
            # Reshape for broadcasting: (B,) -> (B,1,1,1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(batch_size, 1, 1, 1)
            sqrt_alpha = sqrt_alpha.reshape(batch_size, 1, 1, 1)
            
            # Calculate x_0 from noise prediction
            x_0_pred = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
            x_0_pred = torch.clamp(x_0_pred, -1., 1.)
            
        return x_0_pred
    
    def consistency_loss(self, x_0_pred, x_0_target):
        """Simple consistency loss"""
        return torch.nn.functional.mse_loss(x_0_pred, x_0_target)
    
    def distillation_loss(self, x_t, t, hint, x_0_target, alpha=0.5):
        """
        Combined loss: consistency loss + distillation loss
        """
        # Student prediction
        x_0_student = self.student(x_t, t, hint)
        
        # Teacher prediction
        x_0_teacher = self.get_teacher_prediction(x_t, t, hint)
        
        # Consistency loss
        consistency_loss = self.consistency_loss(x_0_student, x_0_target)
        
        # Distillation loss (student should match teacher)
        distillation_loss = torch.nn.functional.mse_loss(x_0_student, x_0_teacher)
        
        # Combined loss
        total_loss = alpha * consistency_loss + (1 - alpha) * distillation_loss
        
        return total_loss, consistency_loss, distillation_loss