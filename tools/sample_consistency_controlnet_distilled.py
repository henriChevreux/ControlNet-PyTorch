import torch
import yaml
import argparse
import os
from dataset.mnist_dataset import MnistDataset
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
                                    'consistency_controlnet_distilled_ckpt.pth')
    
    if not os.path.exists(student_ckpt_path):
        raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt_path}")
    
    model.student.load_state_dict(torch.load(student_ckpt_path, map_location=device))
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
            
            # Use final timestep for single-step generation
            t = torch.full((1,), scheduler.num_timesteps - 1, device=device)
            
            # Generate sample
            x_0_pred = model.student(x_t, t, hint)
            
            # Save sample
            sample = torch.clamp(x_0_pred, -1, 1)
            sample = (sample + 1) / 2  # Convert to [0, 1]
            
            vutils.save_image(sample, os.path.join(output_dir, f'random_sample_{i:03d}.png'))
            
            # Save hint for reference
            hint_vis = torch.clamp(hint, -1, 1)
            hint_vis = (hint_vis + 1) / 2
            vutils.save_image(hint_vis, os.path.join(output_dir, f'random_hint_{i:03d}.png'))


def generate_from_test_data(model, scheduler, dataset_config, output_dir, num_samples):
    """Generate samples from test dataset using their control hints"""
    print(f"Generating {num_samples} samples from test data...")
    
    # Create test dataset
    test_dataset = MnistDataset('test',
                               im_path=dataset_config['im_test_path'],
                               return_hints=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for i, (im, hint) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Start from pure noise
            x_t = torch.randn_like(im)
            t = torch.full((im.shape[0],), scheduler.num_timesteps - 1, device=device)
            
            # Generate sample
            x_0_pred = model.student(x_t, t, hint)
            
            # Save results
            sample = torch.clamp(x_0_pred, -1, 1)
            sample = (sample + 1) / 2
            
            # Original image for comparison
            original = (im + 1) / 2
            
            # Convert hint to grayscale for comparison (take first channel)
            hint_gray = hint[:, 0:1, :, :]  # Take only first channel
            
            # Save individual images
            vutils.save_image(sample, os.path.join(output_dir, f'test_sample_{i:03d}.png'))
            vutils.save_image(original, os.path.join(output_dir, f'test_original_{i:03d}.png'))
            vutils.save_image(hint_gray, os.path.join(output_dir, f'test_hint_{i:03d}.png'))
            
            # Save full hint (3-channel)
            vutils.save_image(hint, os.path.join(output_dir, f'test_hint_full_{i:03d}.png'))
            
            # Create comparison grid (all grayscale)
            comparison = torch.cat([hint_gray, original, sample], dim=0)
            vutils.save_image(comparison, os.path.join(output_dir, f'test_comparison_{i:03d}.png'), nrow=3)


def generate_from_custom_hints(model, scheduler, model_config, output_dir, num_samples):
    """Generate samples from custom control hints"""
    print(f"Generating {num_samples} samples from custom hints...")
    
    # Create some custom control hints (e.g., different digit shapes)
    custom_hints = create_custom_hints(model_config, num_samples)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random noise
            x_t = torch.randn(1, model_config['im_channels'], 
                             model_config['im_size'], model_config['im_size']).to(device)
            
            # Use custom hint
            hint = custom_hints[i:i+1].to(device)
            
            # Use final timestep for single-step generation
            t = torch.full((1,), scheduler.num_timesteps - 1, device=device)
            
            # Generate sample
            x_0_pred = model.student(x_t, t, hint)
            
            # Save sample
            sample = torch.clamp(x_0_pred, -1, 1)
            sample = (sample + 1) / 2
            
            vutils.save_image(sample, os.path.join(output_dir, f'custom_sample_{i:03d}.png'))
            vutils.save_image(hint, os.path.join(output_dir, f'custom_hint_{i:03d}.png'))


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