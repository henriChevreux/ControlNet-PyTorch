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
    
    # Load student checkpoint
    student_ckpt = os.path.join(train_config['task_name'], 
                               'consistency_controlnet_distilled_ckpt.pth')
    if os.path.exists(student_ckpt):
        consistency_controlnet.student.load_state_dict(torch.load(student_ckpt, map_location=device))
        print("Loaded Consistency ControlNet checkpoint")
    else:
        print("Warning: Consistency ControlNet checkpoint not found.")
    
    consistency_controlnet.eval()
    
    # Create test dataset
    test_dataset = MnistDataset('test',
                               im_path=dataset_config['im_test_path'],
                               return_hints=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Create output directory
    output_dir = os.path.join(train_config['task_name'], 'model_comparison')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Comparison metrics
    ddpm_times = []
    consistency_times = []
    ddpm_samples = []
    consistency_samples = []
    hints = []
    originals = []
    
    print(f"Generating {args.num_samples} comparison samples...")
    
    with torch.no_grad():
        for i, (im, hint) in enumerate(test_loader):
            if i >= args.num_samples:
                break
                
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Store original and hint
            originals.append(im.cpu())
            hints.append(hint.cpu())
            
            # Generate with DDPM ControlNet
            print(f"Sample {i+1}: Generating with DDPM ControlNet...")
            ddpm_sample, ddpm_time = generate_ddpm_sample(
                ddpm_controlnet, ddpm_scheduler, im, hint, args.ddpm_steps
            )
            ddpm_samples.append(ddpm_sample.cpu())
            ddpm_times.append(ddpm_time)
            
            # Generate with Consistency ControlNet
            print(f"Sample {i+1}: Generating with Consistency ControlNet...")
            consistency_sample, consistency_time = generate_consistency_sample(
                consistency_controlnet, consistency_scheduler, im, hint
            )
            consistency_samples.append(consistency_sample.cpu())
            consistency_times.append(consistency_time)
    
    # Save comparison images
    save_comparison_images(output_dir, hints, originals, ddpm_samples, consistency_samples)
    
    # Print performance metrics
    print_performance_metrics(ddpm_times, consistency_times, args.ddpm_steps)
    
    # Save detailed metrics
    save_metrics(output_dir, ddpm_times, consistency_times, args.ddpm_steps)
    
    print(f"Comparison completed! Results saved in {output_dir}")


def generate_ddpm_sample(model, scheduler, im, hint, num_steps):
    """Generate sample using DDPM ControlNet with iterative denoising"""
    start_time = time.time()
    
    # Start from pure noise
    x_t = torch.randn_like(im)
    
    # Iterative denoising
    for t in range(num_steps - 1, -1, -1):
        t_tensor = torch.full((im.shape[0],), t, device=im.device)
        
        # Predict noise
        noise_pred = model(x_t, t_tensor, hint)
        
        # Denoise step
        if t > 0:
            x_t, _ = scheduler.sample_prev_timestep(x_t, noise_pred, t_tensor)
        else:
            _, x_t = scheduler.sample_prev_timestep(x_t, noise_pred, t_tensor)
    
    end_time = time.time()
    return x_t, end_time - start_time


def generate_consistency_sample(model, scheduler, im, hint):
    """Generate sample using Consistency ControlNet with single step"""
    start_time = time.time()
    
    # Start from pure noise
    x_t = torch.randn_like(im)
    
    # Single-step generation
    t = torch.full((im.shape[0],), scheduler.num_timesteps - 1, device=im.device)
    x_0_pred = model.student(x_t, t, hint)
    
    end_time = time.time()
    return x_0_pred, end_time - start_time


def save_comparison_images(output_dir, hints, originals, ddpm_samples, consistency_samples):
    """Save comparison images in a grid format"""
    print("Saving comparison images...")
    
    for i in range(len(hints)):
        # Convert hint to grayscale for display
        hint_gray = hints[i][:, 0:1, :, :]  # Take first channel
        
        # Create comparison grid: [Hint, Original, DDPM, Consistency]
        comparison = torch.cat([
            hint_gray,           # Hint
            originals[i],        # Original
            ddpm_samples[i],     # DDPM ControlNet
            consistency_samples[i]  # Consistency ControlNet
        ], dim=0)
        
        # Normalize to [0, 1] range
        comparison = torch.clamp(comparison, -1, 1)
        comparison = (comparison + 1) / 2
        
        # Save individual comparison
        vutils.save_image(comparison, 
                         os.path.join(output_dir, f'comparison_{i:03d}.png'), 
                         nrow=4)
        
        # Save full hint (3-channel) separately
        hint_full = torch.clamp(hints[i], -1, 1)
        hint_full = (hint_full + 1) / 2
        vutils.save_image(hint_full, 
                         os.path.join(output_dir, f'hint_full_{i:03d}.png'))
    
    # Create summary grid
    create_summary_grid(output_dir, hints, originals, ddpm_samples, consistency_samples)


def create_summary_grid(output_dir, hints, originals, ddpm_samples, consistency_samples):
    """Create a summary grid showing multiple samples"""
    print("Creating summary grid...")
    
    # Select first 8 samples for summary
    num_summary = min(8, len(hints))
    
    # Create rows for each model type
    hint_row = torch.cat([hints[i][:, 0:1, :, :] for i in range(num_summary)], dim=0)
    original_row = torch.cat([originals[i] for i in range(num_summary)], dim=0)
    ddpm_row = torch.cat([ddpm_samples[i] for i in range(num_summary)], dim=0)
    consistency_row = torch.cat([consistency_samples[i] for i in range(num_summary)], dim=0)
    
    # Combine all rows
    summary = torch.cat([hint_row, original_row, ddpm_row, consistency_row], dim=0)
    
    # Normalize
    summary = torch.clamp(summary, -1, 1)
    summary = (summary + 1) / 2
    
    # Save summary grid
    vutils.save_image(summary, 
                     os.path.join(output_dir, 'summary_grid.png'), 
                     nrow=num_summary)


def print_performance_metrics(ddpm_times, consistency_times, ddpm_steps):
    """Print performance comparison metrics"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    ddpm_mean = np.mean(ddpm_times)
    ddpm_std = np.std(ddpm_times)
    consistency_mean = np.mean(consistency_times)
    consistency_std = np.std(consistency_times)
    
    speedup = ddpm_mean / consistency_mean
    
    print(f"DDPM ControlNet ({ddpm_steps} steps):")
    print(f"  Mean time: {ddpm_mean:.4f} ± {ddpm_std:.4f} seconds")
    print(f"  Total time: {np.sum(ddpm_times):.2f} seconds")
    
    print(f"\nConsistency ControlNet (1 step):")
    print(f"  Mean time: {consistency_mean:.4f} ± {consistency_std:.4f} seconds")
    print(f"  Total time: {np.sum(consistency_times):.2f} seconds")
    
    print(f"\nSpeedup: {speedup:.2f}x faster")
    print(f"Time reduction: {((ddpm_mean - consistency_mean) / ddpm_mean * 100):.1f}%")


def save_metrics(output_dir, ddpm_times, consistency_times, ddpm_steps):
    """Save detailed metrics to file"""
    metrics_file = os.path.join(output_dir, 'performance_metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write("MODEL COMPARISON METRICS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"DDPM ControlNet ({ddpm_steps} steps):\n")
        f.write(f"  Mean time: {np.mean(ddpm_times):.4f} seconds\n")
        f.write(f"  Std time: {np.std(ddpm_times):.4f} seconds\n")
        f.write(f"  Min time: {np.min(ddpm_times):.4f} seconds\n")
        f.write(f"  Max time: {np.max(ddpm_times):.4f} seconds\n")
        f.write(f"  Total time: {np.sum(ddpm_times):.2f} seconds\n\n")
        
        f.write(f"Consistency ControlNet (1 step):\n")
        f.write(f"  Mean time: {np.mean(consistency_times):.4f} seconds\n")
        f.write(f"  Std time: {np.std(consistency_times):.4f} seconds\n")
        f.write(f"  Min time: {np.min(consistency_times):.4f} seconds\n")
        f.write(f"  Max time: {np.max(consistency_times):.4f} seconds\n")
        f.write(f"  Total time: {np.sum(consistency_times):.2f} seconds\n\n")
        
        speedup = np.mean(ddpm_times) / np.mean(consistency_times)
        f.write(f"Speedup: {speedup:.2f}x faster\n")
        f.write(f"Time reduction: {((np.mean(ddpm_times) - np.mean(consistency_times)) / np.mean(ddpm_times) * 100):.1f}%\n")
    
    # Save timing data for plotting
    timing_data = {
        'ddpm_times': ddpm_times,
        'consistency_times': consistency_times,
        'speedup': speedup
    }
    
    np.save(os.path.join(output_dir, 'timing_data.npy'), timing_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare DDPM ControlNet vs Consistency ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to compare')
    parser.add_argument('--ddpm_steps', type=int, default=50,
                        help='Number of denoising steps for DDPM (default: 50 for speed)')
    args = parser.parse_args()
    
    compare_models(args) 