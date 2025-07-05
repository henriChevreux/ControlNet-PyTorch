import torch


class ConsistencyScheduler:
    """
    Scheduler for Consistency Model training
    Implements the consistency loss and sampling strategy
    """
    def __init__(self, num_timesteps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
        # Create noise schedule
        self.sigmas = self._create_noise_schedule()
        
    def _create_noise_schedule(self):
        """Create noise schedule following EDM paper"""
        ramp = torch.linspace(0, 1, self.num_timesteps)
        sigmas = self.sigma_min ** (1 - ramp) * self.sigma_max ** ramp
        return sigmas
    
    def add_noise(self, x_0, t):
        """Add noise to clean image according to timestep"""
        # Move sigmas to the same device as x_0
        sigmas = self.sigmas.to(x_0.device)
        sigma = sigmas[t].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = x_0 + sigma * noise
        return x_t, noise
    
    def consistency_loss(self, model, x_t, t, hint, x_0_target):
        """
        Compute consistency loss
        L = ||f(x_t, t, hint) - f(x_s, s, hint)||^2
        where s is a nearby timestep
        """
        # Get current prediction
        x_0_pred_t = model(x_t, t, hint)
        
        # Get nearby timestep (s = t - 1)
        s = torch.clamp(t - 1, 0, self.num_timesteps - 1)
        
        # Add noise to target at timestep s
        x_s, _ = self.add_noise(x_0_target, s)
        
        # Get prediction at timestep s
        x_0_pred_s = model(x_s, s, hint)
        
        # Consistency loss
        loss = torch.nn.functional.mse_loss(x_0_pred_t, x_0_pred_s)
        
        return loss
    
    def sample(self, model, x_t, t, hint):
        """Single-step sampling"""
        with torch.no_grad():
            x_0_pred = model(x_t, t, hint)
        return x_0_pred