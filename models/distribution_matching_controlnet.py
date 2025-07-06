import torch
import torch.nn as nn
from models.unet_base import Unet
from models.unet_base import get_time_embedding
from models.controlnet import ControlNet


def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class DistributionMatchingControlNet(nn.Module):
    """
    Distribution Matching Model for ControlNet
    This model learns to match the distribution of generated samples to the target distribution
    """
    def __init__(self, model_config):
        super().__init__()
        
        # Main UNet for distribution matching prediction
        self.unet = Unet(model_config)
        
        # Hint processing block (similar to ControlNet)
        self.hint_block = nn.Sequential(
            nn.Conv2d(model_config['hint_channels'], 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, self.unet.down_channels[0], kernel_size=3, padding=1),
            nn.SiLU(),
            make_zero_module(nn.Conv2d(self.unet.down_channels[0], 
                                      self.unet.down_channels[0], 
                                      kernel_size=1, padding=0))
        )
        
        # Time embedding for distribution matching model
        self.t_emb_dim = model_config['time_emb_dim']
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
    def forward(self, x_t, t, hint):
        """
        Forward pass for distribution matching model
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


class DistributionMatchingControlNetDistilled(nn.Module):
    """
    Distribution Matching Model distilled from pre-trained DDPM ControlNet
    """
    def __init__(self, model_config, teacher_ckpt_path, device=None):
        super().__init__()
        
        # Student model (distribution matching model)
        self.student = DistributionMatchingControlNet(model_config)
        
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
    
    def distribution_matching_loss(self, x_0_pred, x_0_target):
        """Distribution matching loss using more stable metrics"""
        # Clamp predictions to prevent extreme values
        x_0_pred = torch.clamp(x_0_pred, -1., 1.)
        x_0_target = torch.clamp(x_0_target, -1., 1.)
        
        # MSE loss for pixel-level matching (primary loss)
        mse_loss = nn.functional.mse_loss(x_0_pred, x_0_target)
        
        # L1 loss for additional distribution matching
        l1_loss = nn.functional.l1_loss(x_0_pred, x_0_target)
        
        # Cosine similarity loss for distribution matching
        # Flatten and normalize
        pred_flat = x_0_pred.view(x_0_pred.size(0), -1)
        target_flat = x_0_target.view(x_0_target.size(0), -1)
        
        # Normalize to unit vectors
        pred_norm = pred_flat / (torch.norm(pred_flat, dim=1, keepdim=True) + 1e-8)
        target_norm = target_flat / (torch.norm(target_flat, dim=1, keepdim=True) + 1e-8)
        
        # Cosine similarity (1 - cosine_sim for loss)
        cosine_loss = 1 - torch.mean(torch.sum(pred_norm * target_norm, dim=1))
        
        # Combine losses with weights
        total_loss = mse_loss + 0.1 * l1_loss + 0.1 * cosine_loss
        
        return total_loss
    
    def distillation_loss(self, x_t, t, hint, x_0_target, alpha=0.5):
        """
        Combined loss: distribution matching loss + distillation loss
        """
        # Student prediction
        x_0_student = self.student(x_t, t, hint)
        
        # Teacher prediction
        x_0_teacher = self.get_teacher_prediction(x_t, t, hint)
        
        # Distribution matching loss
        dist_matching_loss = self.distribution_matching_loss(x_0_student, x_0_target)
        
        # Distillation loss (student should match teacher)
        distillation_loss = torch.nn.functional.mse_loss(x_0_student, x_0_teacher)
        
        # Combined loss with more weight on distillation for stability
        total_loss = 0.3 * dist_matching_loss + 0.7 * distillation_loss
        
        return total_loss, dist_matching_loss, distillation_loss 