import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_base import Unet
from models.unet_base import get_time_embedding
from models.controlnet import ControlNet
import torchvision.models as models


def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeatureExtractor(nn.Module):
    """Feature extractor optimized for both grayscale (MNIST) and RGB (CIFAR) images"""
    def __init__(self, in_channels=1, trainable=False):
        super().__init__()
        
        # Scale feature channels based on input channels for better capacity
        base_channels = 32 if in_channels == 1 else 64  # More channels for RGB
        
        # Multi-scale feature extraction for both MNIST and CIFAR
        self.features = nn.ModuleList([
            # Low-level features (edges, basic shapes)
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU()
            ),
            # Mid-level features (digit parts / object parts)
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU()
            ),
            # High-level features (digit shapes / object shapes)
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(),
                nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU()
            ),
            # Very high-level features (global structure)
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(),
                nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU()
            )
        ])
        
        if not trainable:
            # Initialize with reasonable weights then freeze
            self._initialize_weights()
            for param in self.parameters():
                param.requires_grad = False
        
    def _initialize_weights(self):
        """Initialize with Xavier/Kaiming normal for better feature extraction"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = []
        current_x = x
        
        for layer in self.features:
            current_x = layer(current_x)
            features.append(current_x)
            
        return features


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
        
        # Feature extractor for perceptual/distribution matching
        # Use correct input channels for the dataset (3 for CIFAR, 1 for MNIST)
        im_channels = model_config.get('im_channels', 1)
        self.feature_extractor = FeatureExtractor(in_channels=im_channels)
        
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
    
    def feature_distribution_matching_loss(self, pred_features, target_features):
        """
        Match feature distributions across the batch
        This is the core of true distribution matching
        """
        total_loss = 0.0
        
        for pred_feat, target_feat in zip(pred_features, target_features):
            # Flatten spatial dimensions: (B, C, H, W) -> (B, C*H*W)
            pred_flat = pred_feat.view(pred_feat.size(0), -1)
            target_flat = target_feat.view(target_feat.size(0), -1)
            
            # Match first and second moments across the batch
            # First moment (mean)
            pred_mean = pred_flat.mean(dim=0)  # Mean across batch
            target_mean = target_flat.mean(dim=0)
            mean_loss = F.mse_loss(pred_mean, target_mean)
            
            # Second moment (variance)
            pred_var = pred_flat.var(dim=0, unbiased=False)
            target_var = target_flat.var(dim=0, unbiased=False)
            var_loss = F.mse_loss(pred_var, target_var)
            
            # Optional: Higher order moments for better distribution matching
            pred_centered = pred_flat - pred_mean.unsqueeze(0)
            target_centered = target_flat - target_mean.unsqueeze(0)
            
            # Third moment (skewness approximation)
            pred_skew = (pred_centered ** 3).mean(dim=0)
            target_skew = (target_centered ** 3).mean(dim=0)
            skew_loss = F.mse_loss(pred_skew, target_skew)
            
            # Combine feature-level losses
            feature_loss = mean_loss + var_loss + 0.1 * skew_loss
            total_loss += feature_loss
            
        return total_loss / len(pred_features)
    
    def wasserstein_distance_loss(self, pred_batch, target_batch):
        """
        Approximate Wasserstein distance for distribution matching
        """
        # Flatten images: (B, C, H, W) -> (B, C*H*W)
        pred_flat = pred_batch.view(pred_batch.size(0), -1)
        target_flat = target_batch.view(target_batch.size(0), -1)
        
        # Sort along feature dimension for Wasserstein-1 approximation
        pred_sorted, _ = torch.sort(pred_flat, dim=1)
        target_sorted, _ = torch.sort(target_flat, dim=1)
        
        # L1 distance between sorted distributions
        wasserstein_loss = F.l1_loss(pred_sorted, target_sorted)
        
        return wasserstein_loss
    
    def gram_matrix_loss(self, pred_features, target_features):
        """
        Match Gram matrices for style/distribution matching
        """
        total_loss = 0.0
        
        for pred_feat, target_feat in zip(pred_features, target_features):
            B, C, H, W = pred_feat.size()
            
            # Reshape to (B, C, H*W)
            pred_reshaped = pred_feat.view(B, C, H * W)
            target_reshaped = target_feat.view(B, C, H * W)
            
            # Compute Gram matrices: (B, C, C)
            pred_gram = torch.bmm(pred_reshaped, pred_reshaped.transpose(1, 2))
            target_gram = torch.bmm(target_reshaped, target_reshaped.transpose(1, 2))
            
            # Normalize by number of elements
            pred_gram = pred_gram / (C * H * W)
            target_gram = target_gram / (C * H * W)
            
            # Match Gram matrices
            gram_loss = F.mse_loss(pred_gram, target_gram)
            total_loss += gram_loss
            
        return total_loss / len(pred_features)
    
    def true_distribution_matching_loss(self, x_0_pred, x_0_target):
        """
        True distribution matching loss combining multiple distributional metrics
        """
        # Clamp predictions
        x_0_pred = torch.clamp(x_0_pred, -1., 1.)
        x_0_target = torch.clamp(x_0_target, -1., 1.)
        
        # Extract features for both predictions and targets
        pred_features = self.feature_extractor(x_0_pred)
        target_features = self.feature_extractor(x_0_target)
        
        # 1. Feature distribution matching (core DMD component)
        feature_dist_loss = self.feature_distribution_matching_loss(pred_features, target_features)
        
        # 2. Wasserstein distance approximation
        wasserstein_loss = self.wasserstein_distance_loss(x_0_pred, x_0_target)
        
        # 3. Gram matrix matching for style/texture distribution
        gram_loss = self.gram_matrix_loss(pred_features, target_features)
        
        # 4. Small amount of pixel-wise loss for stability (much less weight)
        pixel_loss = F.mse_loss(x_0_pred, x_0_target)
        
        # Combine with appropriate weights
        total_loss = (
            1.0 * feature_dist_loss +      # Primary distribution matching
            0.5 * wasserstein_loss +       # Distribution distance
            0.3 * gram_loss +              # Style/texture distribution  
            0.1 * pixel_loss               # Stability (minimal weight)
        )
        
        return total_loss, {
            'feature_dist': feature_dist_loss,
            'wasserstein': wasserstein_loss, 
            'gram': gram_loss,
            'pixel': pixel_loss
        }
    
    def distillation_loss(self, x_t, t, hint, x_0_target, alpha=0.3):
        """
        Combined loss: true distribution matching + teacher distillation
        """
        # Student prediction
        x_0_student = self.student(x_t, t, hint)
        
        # Teacher prediction  
        x_0_teacher = self.get_teacher_prediction(x_t, t, hint)
        
        # True distribution matching loss (student should match target distribution)
        dist_matching_loss, dist_components = self.true_distribution_matching_loss(x_0_student, x_0_target)
        
        # Teacher distillation loss (student should also learn from teacher)
        teacher_distill_loss = F.mse_loss(x_0_student, x_0_teacher)
        
        # Combined loss: emphasize distribution matching over teacher copying
        total_loss = alpha * teacher_distill_loss + (1 - alpha) * dist_matching_loss
        
        return total_loss, dist_matching_loss, teacher_distill_loss, dist_components