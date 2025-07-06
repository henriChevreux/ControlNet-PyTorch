import torch
import yaml
import argparse
import os
import time
import numpy as np
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.controlnet import ControlNet
from models.consistency_controlnet_distilled import ConsistencyControlNetDistilled
from models.distribution_matching_controlnet import DistributionMatchingControlNetDistilled
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.consistency_scheduler import ConsistencyScheduler
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compare_models(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create schedulers
    ddpm_scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    consistency_scheduler = ConsistencyScheduler(
        num_timesteps=diffusion_config['num_timesteps']
    )
    
    # Load DDPM ControlNet (teacher)
    print("Loading DDPM ControlNet...")
    ddpm_controlnet = ControlNet(
        model_config,
        model_locked=True,
        model_ckpt=os.path.join(train_config['task_name'], train_config['ddpm_ckpt_name']),
        device=device
    ).to(device)
    
    # Load DDPM ControlNet weights
    ddpm_controlnet_ckpt = os.path.join(train_config['task_name'], 
                                       train_config['controlnet_ckpt_name'])
    if os.path.exists(ddpm_controlnet_ckpt):
        ddpm_controlnet.load_state_dict(torch.load(ddpm_controlnet_ckpt, map_location=device))
        print("Loaded DDPM ControlNet checkpoint")
    else:
        print("Warning: DDPM ControlNet checkpoint not found. Using base DDPM only.")
    
    ddpm_controlnet.eval()
    
    # Load Consistency ControlNet (student)
    print("Loading Consistency ControlNet...")
    consistency_controlnet = ConsistencyControlNetDistilled(
        model_config,
        os.path.join(train_config['task_name'], train_config['ddpm_ckpt_name']),
        device=device
    ).to(device)
    
    # Load consistency student checkpoint
    consistency_ckpt = os.path.join(train_config['task_name'], 
                                   'consistency_controlnet_distilled_ckpt.pth')
    if os.path.exists(consistency_ckpt):
        consistency_controlnet.student.load_state_dict(torch.load(consistency_ckpt, map_location=device))
        print("Loaded Consistency ControlNet checkpoint")
    else:
        print("Warning: Consistency ControlNet checkpoint not found.")
    
    consistency_controlnet.eval()
    
    # Load Distribution Matching ControlNet (student)
    print("Loading Distribution Matching ControlNet...")
    dmd_controlnet = DistributionMatchingControlNetDistilled(
        model_config,
        os.path.join(train_config['task_name'], train_config['ddpm_ckpt_name']),
        device=device
    ).to(device)
    
    # Load distribution matching student checkpoint
    dmd_ckpt = os.path.join(train_config['task_name'], 
                                     'distribution_matching_controlnet_distilled_ckpt.pth')
    if os.path.exists(dmd_ckpt):
        dmd_controlnet.student.load_state_dict(torch.load(dmd_ckpt, map_location=device))
        print("Loaded Distribution Matching ControlNet checkpoint")
    else:
        print("Warning: Distribution Matching ControlNet checkpoint not found.")
    
    dmd_controlnet.eval()
    
    # Create test dataset
    test_dataset = MnistDataset('test',
                               im_path=dataset_config['im_test_path'],
                               return_hints=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Create output directory
    output_dir = os.path.join(train_config['task_name'], 'all_models_comparison')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Comparison metrics
    ddpm_times = []
    consistency_times = []
    dmd_times = []
    ddpm_samples = []
    consistency_samples = []
    dmd_samples = []
    hints = []
    originals = []
    
    print(f"Generating {args.num_samples} comparison samples...")
    
    sample_count = 0
    with torch.no_grad():
        for im, hint in test_loader:
            if sample_count >= args.num_samples:
                break
                
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # DDPM ControlNet sampling
            start_time = time.time()
            ddpm_sample, ddpm_time = generate_ddpm_sample(ddpm_controlnet, ddpm_scheduler, im, hint, args.ddpm_steps)
            ddpm_times.append(ddpm_time)
            ddpm_samples.append(ddpm_sample.cpu())
            
            # Consistency ControlNet sampling
            start_time = time.time()
            consistency_sample, consistency_time = generate_consistency_sample(consistency_controlnet, consistency_scheduler, im, hint)
            consistency_times.append(consistency_time)
            consistency_samples.append(consistency_sample.cpu())
            
            # Distribution Matching ControlNet sampling
            start_time = time.time()
            dmd_sample, dmd_time = generate_dmd_sample(dmd_controlnet, consistency_scheduler, im, hint)
            dmd_times.append(dmd_time)
            dmd_samples.append(dmd_sample.cpu())
            
            hints.append(hint.cpu())
            originals.append(im.cpu())
            sample_count += 1
    
    # Save comparison grid
    save_comparison_grid(ddpm_samples, consistency_samples, dmd_samples, 
                        hints, originals, output_dir, args.num_samples)
    
    # Print statistics
    print("\n" + "="*60)
    print("MODEL COMPARISON STATISTICS")
    print("="*60)
    print(f"DDPM ControlNet:")
    print(f"  Average sampling time: {np.mean(ddpm_times):.4f}s ± {np.std(ddpm_times):.4f}s")
    print(f"  Total sampling time: {np.sum(ddpm_times):.4f}s")
    print(f"  Steps: {args.ddpm_steps}")
    
    print(f"\nConsistency ControlNet:")
    print(f"  Average sampling time: {np.mean(consistency_times):.4f}s ± {np.std(consistency_times):.4f}s")
    print(f"  Total sampling time: {np.sum(consistency_times):.4f}s")
    print(f"  Steps: 1 (single-step)")
    print(f"  Speedup: {np.mean(ddpm_times)/np.mean(consistency_times):.1f}x")
    
    print(f"\nDistribution Matching ControlNet:")
    print(f"  Average sampling time: {np.mean(dmd_times):.4f}s ± {np.std(dmd_times):.4f}s")
    print(f"  Total sampling time: {np.sum(dmd_times):.4f}s")
    print(f"  Steps: 1 (single-step)")
    print(f"  Speedup: {np.mean(ddpm_times)/np.mean(dmd_times):.1f}x")
    
    print(f"\nConsistency vs Distribution Matching:")
    print(f"  Consistency is {np.mean(dmd_times)/np.mean(consistency_times):.2f}x slower")
    
    print("\nResults saved to:", output_dir)


def generate_ddpm_sample(model, scheduler, im, hint, num_steps):
    start_time = time.time()
    x_t = torch.randn_like(im)
    for t in range(num_steps - 1, -1, -1):
        t_tensor = torch.full((im.shape[0],), t, device=im.device)
        noise_pred = model(x_t, t_tensor, hint)
        if t > 0:
            x_t, _ = scheduler.sample_prev_timestep(x_t, noise_pred, t_tensor)
        else:
            _, x_t = scheduler.sample_prev_timestep(x_t, noise_pred, t_tensor)
    end_time = time.time()
    return x_t, end_time - start_time


def generate_consistency_sample(model, scheduler, im, hint):
    start_time = time.time()
    x_t = torch.randn_like(im)
    t = torch.full((im.shape[0],), scheduler.num_timesteps - 1, device=im.device)
    x_0_pred = model.student(x_t, t, hint)
    end_time = time.time()
    return x_0_pred, end_time - start_time


def generate_dmd_sample(model, scheduler, im, hint):
    start_time = time.time()
    x_t = torch.randn_like(im)
    t = torch.full((im.shape[0],), scheduler.num_timesteps - 1, device=im.device)
    x_0_pred = model.student(x_t, t, hint)
    end_time = time.time()
    return x_0_pred, end_time - start_time


def to_three_channels(img):
    # Handle both 3D [C, H, W] and 4D [B, C, H, W] tensors
    if img.dim() == 4:  # [B, C, H, W]
        if img.shape[1] == 1:
            return img.repeat(1, 3, 1, 1)
        elif img.shape[1] == 3:
            return img
        else:
            raise ValueError(f"Unexpected channel count: {img.shape[1]}")
    elif img.dim() == 3:  # [C, H, W]
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1)
        elif img.shape[0] == 3:
            return img
        else:
            raise ValueError(f"Unexpected channel count: {img.shape[0]}")
    else:
        raise ValueError(f"Unexpected tensor dimensions: {img.shape}")


def save_comparison_grid(ddpm_samples, consistency_samples, dmd_samples, 
                        hints, originals, output_dir, num_samples):
    """Save comparison grid of all models"""
    
    for i in range(min(num_samples, len(ddpm_samples))):
        # Always take the first channel for all images
        hint_gray = hints[i][:1]
        original_gray = originals[i][:1]
        ddpm_gray = ddpm_samples[i][:1]
        consistency_gray = consistency_samples[i][:1]
        dmd_gray = dmd_samples[i][:1]

        # Convert to 3 channels for visualization
        hint_rgb = to_three_channels(hint_gray)
        original_rgb = to_three_channels(original_gray)
        ddpm_rgb = to_three_channels(ddpm_gray)
        consistency_rgb = to_three_channels(consistency_gray)
        dmd_rgb = to_three_channels(dmd_gray)

        comparison = torch.cat([
            hint_rgb,
            original_rgb,
            ddpm_rgb,
            consistency_rgb,
            dmd_rgb
        ], dim=0)
        
        vutils.save_image(comparison, 
                         os.path.join(output_dir, f'comparison_{i:03d}.png'), 
                         nrow=5)
    
    # Create summary grid
    if len(ddpm_samples) > 0:
        n_summary = min(8, len(ddpm_samples))
        hint_gray = [to_three_channels(hints[i][:1]) for i in range(n_summary)]
        original_gray = [to_three_channels(originals[i][:1]) for i in range(n_summary)]
        ddpm_gray = [to_three_channels(ddpm_samples[i][:1]) for i in range(n_summary)]
        consistency_gray = [to_three_channels(consistency_samples[i][:1]) for i in range(n_summary)]
        dmd_gray = [to_three_channels(dmd_samples[i][:1]) for i in range(n_summary)]
        
        ddpm_summary = torch.cat(ddpm_gray, dim=0)
        consistency_summary = torch.cat(consistency_gray, dim=0)
        dmd_summary = torch.cat(dmd_gray, dim=0)
        hints_summary = torch.cat(hint_gray, dim=0)
        originals_summary = torch.cat(original_gray, dim=0)
        
        vutils.save_image(ddpm_summary, 
                         os.path.join(output_dir, 'ddpm_summary.png'), 
                         nrow=4)
        vutils.save_image(consistency_summary, 
                         os.path.join(output_dir, 'consistency_summary.png'), 
                         nrow=4)
        vutils.save_image(dmd_summary, 
                         os.path.join(output_dir, 'distribution_matching_summary.png'), 
                         nrow=4)
        vutils.save_image(hints_summary, 
                         os.path.join(output_dir, 'hints_summary.png'), 
                         nrow=4)
        vutils.save_image(originals_summary, 
                         os.path.join(output_dir, 'originals_summary.png'), 
                         nrow=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare all ControlNet models')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    parser.add_argument('--num_samples', default=10, type=int,
                        help='Number of samples to generate for comparison')
    parser.add_argument('--ddpm_steps', default=50, type=int,
                        help='Number of DDPM sampling steps')
    args = parser.parse_args()
    compare_models(args) 