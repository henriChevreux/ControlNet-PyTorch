import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.distribution_matching_controlnet import DistributionMatchingControlNetDistilled
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create noise scheduler for adding noise during training
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    
    # Create dataset
    mnist = MnistDataset('train',
                         im_path=dataset_config['im_path'],
                         return_hints=True)
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True)
    
    # Create distilled model - use ControlNet checkpoint
    teacher_ckpt_path = os.path.join(train_config['task_name'], 
                                    train_config['controlnet_ckpt_name'])
    
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}. "
                               "Please train ControlNet first.")
    
    model = DistributionMatchingControlNetDistilled(
        model_config, 
        teacher_ckpt_path, 
    ).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Training parameters
    num_epochs = train_config.get('distribution_matching_epochs', 20)
    optimizer = Adam(model.student.parameters(), lr=train_config.get('distribution_matching_lr', 0.0001))
    
    # Training loop
    for epoch_idx in range(num_epochs):
        dist_matching_losses = []
        distillation_losses = []
        total_losses = []
        
        for im, hint in tqdm(mnist_loader):
            optimizer.zero_grad()
            
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Sample random timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images
            noise = torch.randn_like(im)
            x_t = scheduler.add_noise(im, noise, t)
            
            # Compute distillation loss
            total_loss, dist_matching_loss, dist_loss = model.distillation_loss(x_t, t, hint, im)
            
            # Check for NaN values
            if torch.isnan(total_loss) or torch.isnan(dist_matching_loss) or torch.isnan(dist_loss):
                print(f"Warning: NaN detected in loss computation!")
                print(f"Total loss: {total_loss}, Dist matching: {dist_matching_loss}, Distillation: {dist_loss}")
                continue
            
            total_losses.append(total_loss.item())
            dist_matching_losses.append(dist_matching_loss.item())
            distillation_losses.append(dist_loss.item())
            
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        print(f'Epoch {epoch_idx + 1} | Total Loss: {np.mean(total_losses):.4f} | '
              f'Distribution Matching: {np.mean(dist_matching_losses):.4f} | '
              f'Distillation: {np.mean(distillation_losses):.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(train_config['task_name'], 
                                      'distribution_matching_controlnet_distilled_ckpt.pth')
        torch.save(model.student.state_dict(), checkpoint_path)
    
    print('Distribution matching distillation training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distilled Distribution Matching ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args) 