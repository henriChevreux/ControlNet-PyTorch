import torch
import torch.nn as nn
from models.unet_base import Unet
from models.unet_base import get_time_embedding


def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ConsistencyControlNet(nn.Module):
    """
    Consistency Model version of ControlNet for MNIST
    This model learns to predict x_0 directly from x_t and hint
    """
    def __init__(self, model_config, device=None):
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
            make_zero_module(nn.Conv2d(self.unet.down_channels[0], 
                                      self.unet.down_channels[0], 
                                      kernel_size=1, padding=0))
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