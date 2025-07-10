import torch
import yaml
import argparse
import os
from dataset.mnist_dataset import MnistDataset
from dataset.cifar_dataset import CifarDataset
from torch.utils.data import DataLoader
from models.consistency_controlnet_distilled import ConsistencyControlNetDistilled
from scheduler.consistency_scheduler import ConsistencyScheduler
import torchvision.utils as vutils

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
    scheduler = ConsistencyScheduler(
        num_timesteps=diffusion_config['num_timesteps']
    )
    
    # Create model
    teacher_ckpt_path = os.path.join(train_config['task_name'], 
                                    train_config['ddpm_ckpt_name'])
    
    model = ConsistencyControlNetDistilled(
        model_config, 
        teacher_ckpt_path, 
        device=device
    ).to(device)
    
    # Load student checkpoint
    student_ckpt_path = os.path.join(train_config['task_name'], 
                                    'consistency_controlnet_distilled.pth')
    
    if not os.path.exists(student_ckpt_path):
        raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt_path}")
    
    checkpoint = torch.load(student_ckpt_path, map_location=device, weights_only=False)
    model.student.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create output directory
    output_dir = os.path.join(train_config['task_name'], 'consistency_samples')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Option 1: Generate from random noise with control hints
    if args.mode == 'random':
        generate_from_random_noise(model, scheduler, model_config, output_dir, args.num_samples)
    
    # Option 2: Generate from test dataset with their hints
    elif args.mode == 'test':
        generate_from_test_data(model, scheduler, dataset_config, output_dir, args.num_samples)
    
    # Option 3: Generate from custom hints
    elif args.mode == 'custom':
        generate_from_custom_hints(model, scheduler, model_config, output_dir, args.num_samples)
    
    print('Inference completed!')


def generate_from_random_noise(model, scheduler, model_config, output_dir, num_samples):
    """Generate samples from random noise with random control hints"""
    print(f"Generating {num_samples} samples from random noise...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random noise
            x_t = torch.randn(1, model_config['im_channels'], 
                             model_config['im_size'], model_config['im_size']).to(device)
            
            # Generate random control hint (simulate Canny edges)
            hint = torch.randn(1, model_config['hint_channels'], 
                              model_config['im_size'], model_config['im_size']).to(device)
            
            # Use sigma_max for single-step generation (not timestep)
            sigma = torch.full((1,), model.student.sigma_max, device=device)
            
            # Generate sample
            x_0_pred = model.student(x_t, sigma, hint)
            
            # Save sample
            sample = torch.clamp(x_0_pred, -1, 1)
            sample = (sample + 1) / 2  # Convert to [0, 1]
            
            vutils.save_image(sample, os.path.join(output_dir, f'random_sample_{i:03d}.png'))
            
            # Save hint for reference - adapt channels for visualization
            hint_vis = torch.clamp(hint, -1, 1)
            hint_vis = (hint_vis + 1) / 2
            
            # Adapt hint channels to match image channels for consistent visualization
            im_channels = model_config['im_channels']
            hint_channels = model_config['hint_channels']
            
            if hint_channels == 1 and im_channels == 3:
                # Convert grayscale hint to RGB
                hint_vis = hint_vis.repeat(1, 3, 1, 1)
            elif hint_channels == 3 and im_channels == 1:
                # Convert RGB hint to grayscale
                hint_vis = hint_vis.mean(dim=1, keepdim=True)
            elif hint_channels != im_channels:
                # Take first channel and adapt
                hint_vis = hint_vis[:, 0:1, :, :]
                if im_channels > 1:
                    hint_vis = hint_vis.repeat(1, im_channels, 1, 1)
            
            vutils.save_image(hint_vis, os.path.join(output_dir, f'random_hint_{i:03d}.png'))


def generate_from_test_data(model, scheduler, dataset_config, output_dir, num_samples):
    """Generate samples from test dataset using their control hints"""
    print(f"Generating {num_samples} samples from test data...")
    
    # Create test dataset
    if dataset_config['task_name'] == 'mnist':
        test_dataset = MnistDataset('test',
                                   im_path=dataset_config['im_test_path'],
                                   return_hints=True)
    elif dataset_config['task_name'] == 'cifar10':
        test_dataset = CifarDataset('test',
                                   im_path=dataset_config['im_test_path'],
                                   return_hints=True)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_config['task_name']}")
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for i, (im, hint) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Start from pure noise
            x_t = torch.randn_like(im)
            
            # Use sigma_max for single-step generation (not timestep)
            sigma = torch.full((im.shape[0],), model.student.sigma_max, device=device)
            
            # Generate sample - consistency model expects sigma, not timestep
            x_0_pred = model.student(x_t, sigma, hint)
            
            # Save results
            sample = torch.clamp(x_0_pred, -1, 1)
            sample = (sample + 1) / 2
            
            # Original image for comparison
            original = (im + 1) / 2
            
            # Get the number of channels in the images
            im_channels = original.shape[1]
            
            # Prepare hint for visualization - adapt to match image channels
            if hint.shape[1] == 1:
                # Single channel hint (e.g., Canny edges)
                hint_vis = hint[:, 0:1, :, :]  # Take first channel
                if im_channels == 3:
                    # Convert to RGB by repeating
                    hint_vis = hint_vis.repeat(1, 3, 1, 1)
                # For grayscale images (im_channels == 1), keep as is
            elif hint.shape[1] == 3:
                # Multi-channel hint
                if im_channels == 1:
                    # Convert to grayscale
                    hint_vis = hint.mean(dim=1, keepdim=True)
                else:
                    # Keep as is for RGB
                    hint_vis = hint
            else:
                # Handle other hint channel configurations
                # Take first channel and adapt to image channels
                hint_vis = hint[:, 0:1, :, :]
                if im_channels > 1:
                    hint_vis = hint_vis.repeat(1, im_channels, 1, 1)
            
            # Normalize hint for display
            hint_vis = torch.clamp(hint_vis, -1, 1)
            hint_vis = (hint_vis + 1) / 2
            
            # Create comparison grid (all have same number of channels now)
            comparison = torch.cat([hint_vis, original, sample], dim=0)
            vutils.save_image(comparison, os.path.join(output_dir, f'test_comparison_{i:03d}.png'), nrow=3)


def generate_from_custom_hints(model, scheduler, model_config, output_dir, num_samples):
    """Generate samples from custom control hints"""
    print(f"Generating {num_samples} samples from custom hints...")
    
    # Create some custom control hints
    custom_hints = create_custom_hints(model_config, num_samples)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random noise
            x_t = torch.randn(1, model_config['im_channels'], 
                             model_config['im_size'], model_config['im_size']).to(device)
            
            # Use custom hint
            hint = custom_hints[i:i+1].to(device)
            
            # Use sigma_max for single-step generation (not timestep)
            sigma = torch.full((1,), model.student.sigma_max, device=device)
            
            # Generate sample
            x_0_pred = model.student(x_t, sigma, hint)
            
            # Save sample
            sample = torch.clamp(x_0_pred, -1, 1)
            sample = (sample + 1) / 2
            
            vutils.save_image(sample, os.path.join(output_dir, f'custom_sample_{i:03d}.png'))
            
            # Save hint with proper channel adaptation
            hint_vis = hint.clone()
            im_channels = model_config['im_channels']
            hint_channels = model_config['hint_channels']
            
            if hint_channels == 1 and im_channels == 3:
                hint_vis = hint_vis.repeat(1, 3, 1, 1)
            elif hint_channels == 3 and im_channels == 1:
                hint_vis = hint_vis.mean(dim=1, keepdim=True)
            elif hint_channels != im_channels:
                hint_vis = hint_vis[:, 0:1, :, :]
                if im_channels > 1:
                    hint_vis = hint_vis.repeat(1, im_channels, 1, 1)
            
            vutils.save_image(hint_vis, os.path.join(output_dir, f'custom_hint_{i:03d}.png'))


def create_custom_hints(model_config, num_samples):
    """Create custom control hints for different digit shapes"""
    hints = []
    
    for i in range(num_samples):
        # Create a simple geometric shape as hint
        hint = torch.zeros(1, model_config['hint_channels'], 
                          model_config['im_size'], model_config['im_size'])
        
        # Create different patterns based on sample index
        if i % 5 == 0:  # Horizontal lines
            hint[:, :, 8:12, :] = 1.0
        elif i % 5 == 1:  # Vertical lines
            hint[:, :, :, 8:12] = 1.0
        elif i % 5 == 2:  # Cross
            hint[:, :, 8:12, :] = 1.0
            hint[:, :, :, 8:12] = 1.0
        elif i % 5 == 3:  # Circle-like
            center = model_config['im_size'] // 2
            for y in range(model_config['im_size']):
                for x in range(model_config['im_size']):
                    dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
                    if 8 <= dist <= 12:
                        hint[:, :, y, x] = 1.0
        else:  # Random pattern
            hint = torch.rand_like(hint)
        
        hints.append(hint)
    
    return torch.cat(hints, dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from Distilled Consistency ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    parser.add_argument('--mode', choices=['random', 'test', 'custom'],
                        default='test', help='Inference mode')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate')
    args = parser.parse_args()
    sample(args) 