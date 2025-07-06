import torch
import yaml
import argparse
import os
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.distribution_matching_controlnet import DistributionMatchingControlNetDistilled
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import torchvision.utils as vutils
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    
    # Create model
    teacher_ckpt_path = os.path.join(train_config['task_name'], 
                                    train_config['controlnet_ckpt_name'])
    
    model = DistributionMatchingControlNetDistilled(
        model_config, 
        teacher_ckpt_path, 
        device=device
    ).to(device)
    
    # Load student checkpoint
    student_ckpt_path = os.path.join(train_config['task_name'], 
                                    'distribution_matching_controlnet_distilled_ckpt.pth')
    
    if not os.path.exists(student_ckpt_path):
        raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt_path}")
    
    model.student.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    model.eval()
    
    # Create output directory
    output_dir = os.path.join(train_config['task_name'], 'distribution_matching_samples')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate samples based on mode
    if args.mode == 'test':
        generate_from_test_data(model, scheduler, dataset_config, output_dir, args.num_samples)
    elif args.mode == 'random':
        generate_from_random_noise(model, scheduler, output_dir, args.num_samples, model_config)
    elif args.mode == 'custom':
        generate_from_custom_hints(model, scheduler, output_dir, args.num_samples, model_config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def generate_from_test_data(model, scheduler, dataset_config, output_dir, num_samples):
    """Generate samples from test data"""
    print(f"Generating {num_samples} samples from test data...")
    
    # Create test dataset
    test_dataset = MnistDataset('test',
                               im_path=dataset_config['im_test_path'],
                               return_hints=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    samples_generated = 0
    with torch.no_grad():
        for im, hint in test_loader:
            if samples_generated >= num_samples:
                break
                
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Start from pure noise
            x_t = torch.randn_like(im).to(device)
            t = torch.full((im.shape[0],), scheduler.num_timesteps - 1, device=device)
            
            # Single-step sampling
            x_0_pred = model.student(x_t, t, hint)
            
            # Save comparison
            comparison = torch.cat([hint, x_0_pred, im], dim=0)
            vutils.save_image(comparison, 
                            os.path.join(output_dir, f'test_comparison_{samples_generated:03d}.png'), 
                            nrow=3)
            
            samples_generated += 1
    
    print(f"Generated {samples_generated} samples from test data")


def generate_from_random_noise(model, scheduler, output_dir, num_samples, model_config):
    """Generate samples from random noise with random hints"""
    print(f"Generating {num_samples} samples from random noise...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random hint (Canny-like edges)
            hint = torch.randn(1, model_config['hint_channels'], 
                              model_config['im_size'], model_config['im_size']).to(device)
            
            # Start from pure noise
            x_t = torch.randn(1, model_config['im_channels'], 
                             model_config['im_size'], model_config['im_size']).to(device)
            t = torch.full((1,), scheduler.num_timesteps - 1, device=device)
            
            # Single-step sampling
            x_0_pred = model.student(x_t, t, hint)
            
            # Save sample
            comparison = torch.cat([hint, x_0_pred], dim=0)
            vutils.save_image(comparison, 
                            os.path.join(output_dir, f'random_sample_{i:03d}.png'), 
                            nrow=2)
    
    print(f"Generated {num_samples} samples from random noise")


def generate_from_custom_hints(model, scheduler, output_dir, num_samples, model_config):
    """Generate samples from custom geometric hints"""
    print(f"Generating {num_samples} samples from custom hints...")
    
    # Create custom geometric hints
    custom_hints = create_geometric_hints(num_samples, model_config)
    
    with torch.no_grad():
        for i, hint in enumerate(custom_hints):
            hint = hint.unsqueeze(0).to(device)  # Add batch dimension
            
            # Start from pure noise
            x_t = torch.randn(1, model_config['im_channels'], 
                             model_config['im_size'], model_config['im_size']).to(device)
            t = torch.full((1,), scheduler.num_timesteps - 1, device=device)
            
            # Single-step sampling
            x_0_pred = model.student(x_t, t, hint)
            
            # Save sample
            comparison = torch.cat([hint, x_0_pred], dim=0)
            vutils.save_image(comparison, 
                            os.path.join(output_dir, f'custom_sample_{i:03d}.png'), 
                            nrow=2)
    
    print(f"Generated {num_samples} samples from custom hints")


def create_geometric_hints(num_samples, model_config):
    """Create custom geometric hints for testing"""
    hints = []
    im_size = model_config['im_size']
    
    for i in range(num_samples):
        # Create different geometric patterns
        if i % 4 == 0:
            # Horizontal lines
            hint = torch.zeros(model_config['hint_channels'], im_size, im_size)
            for j in range(0, im_size, 4):
                hint[:, j:j+1, :] = 1.0
        elif i % 4 == 1:
            # Vertical lines
            hint = torch.zeros(model_config['hint_channels'], im_size, im_size)
            for j in range(0, im_size, 4):
                hint[:, :, j:j+1] = 1.0
        elif i % 4 == 2:
            # Diagonal lines
            hint = torch.zeros(model_config['hint_channels'], im_size, im_size)
            for j in range(im_size):
                hint[:, j, j] = 1.0
        else:
            # Circle
            hint = torch.zeros(model_config['hint_channels'], im_size, im_size)
            center = im_size // 2
            radius = im_size // 4
            for y in range(im_size):
                for x in range(im_size):
                    if (x - center) ** 2 + (y - center) ** 2 <= radius ** 2:
                        hint[:, y, x] = 1.0
        
        hints.append(hint)
    
    return hints


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from Distilled Distribution Matching ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    parser.add_argument('--mode', choices=['test', 'random', 'custom'],
                        default='test', type=str,
                        help='Sampling mode: test (from test data), random (from random noise), custom (from geometric hints)')
    parser.add_argument('--num_samples', default=10, type=int,
                        help='Number of samples to generate')
    args = parser.parse_args()
    sample(args) 