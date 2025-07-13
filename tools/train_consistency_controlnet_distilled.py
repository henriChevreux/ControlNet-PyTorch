import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from dataset.cifar_dataset import CifarDataset
from torch.utils.data import DataLoader
from models.consistency_controlnet_distilled import ConsistencyControlNetDistilled
from scheduler.consistency_scheduler import ConsistencyScheduler
# Suppress OpenCV parallel backend info messages
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import cv2
cv2.setLogLevel(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def timestep_to_sigma(t, sigma_min=0.002, sigma_max=80.0, num_timesteps=1000):
    """Convert discrete timestep to continuous noise level"""
    t = t.float()
    alpha = t / (num_timesteps - 1)
    sigma = sigma_min * (sigma_max / sigma_min) ** alpha
    return sigma


def train(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create consistency scheduler
    scheduler = ConsistencyScheduler(
        num_timesteps=diffusion_config['num_timesteps'])
    
    # Create the dataset
    if train_config['task_name'] == 'mnist':
        dataset = MnistDataset('train', im_path=dataset_config['im_path'], return_hints=True)
    elif train_config['task_name'] == 'cifar10':
        dataset = CifarDataset('train', im_path=dataset_config['im_path'], download=dataset_config['download'], return_hints=True)
    else:
        raise ValueError(f"Invalid dataset name: {train_config['task_name']}")
    
    dataset_loader = DataLoader(dataset,
                              batch_size=train_config['batch_size'],
                              shuffle=True)
    
    # Create distilled model - use ControlNet checkpoint, not DDPM checkpoint
    teacher_ckpt_path = os.path.join(train_config['task_name'], 
                                    train_config['controlnet_ckpt_name'])
    
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}. "
                               "Please train ControlNet first.")
    
    model = ConsistencyControlNetDistilled(
        model_config, 
        teacher_ckpt_path, 
        device=device
    ).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Training parameters
    num_epochs = train_config.get('consistency_epochs', 10)
    optimizer = Adam(model.student.parameters(), lr=train_config.get('consistency_lr', 0.0001))
    
    # Training mode selection
    use_distillation = train_config.get('use_ddpm_distillation', True)
    use_consistency_only = train_config.get('use_consistency_only', False)
    
    # Training loop
    for epoch_idx in range(num_epochs):
        total_losses = []
        consistency_losses = []
        distillation_losses = []
        
        for im, hint in tqdm(dataset_loader):
            optimizer.zero_grad()
            
            im = im.float().to(device)
            hint = hint.float().to(device)
            batch_size = im.shape[0]
            
            if use_consistency_only:
                # Pure consistency training (no DDPM teacher)
                loss_dict = model.training_step(im, hint, use_ddpm_teacher=False)
                total_loss = loss_dict['consistency_loss']
                total_losses.append(total_loss.item())
                
            elif use_distillation:
                # Distillation training with DDPM teacher
                loss_dict = model.training_step(im, hint, use_ddpm_teacher=True)
                total_loss = loss_dict['total_loss']
                recon_loss = loss_dict['recon_loss']
                distill_loss = loss_dict['distill_loss']
                
                total_losses.append(total_loss.item())
                consistency_losses.append(recon_loss.item())
                distillation_losses.append(distill_loss.item())
                
            else:
                # Manual distillation loss (alternative approach)
                # Sample timesteps with bias toward high noise levels (for pure noise sampling)
                if np.random.rand() < 0.5:  # 50% high noise training
                    # Sample heavily from high noise levels (750-999)
                    t = torch.randint(int(0.75 * diffusion_config['num_timesteps']), 
                                     diffusion_config['num_timesteps'], (batch_size,)).to(device)
                else:
                    # Sample from all timesteps
                    t = torch.randint(0, diffusion_config['num_timesteps'], (batch_size,)).to(device)
                
                sigma = timestep_to_sigma(t, 
                                        sigma_min=model.sigma_min, 
                                        sigma_max=model.sigma_max,
                                        num_timesteps=diffusion_config['num_timesteps'])
                
                # Compute distillation loss
                total_loss, cons_loss, dist_loss = model.distillation_loss(im, hint, sigma, alpha=0.5)
                
                total_losses.append(total_loss.item())
                consistency_losses.append(cons_loss.item())
                distillation_losses.append(dist_loss.item())
            
            total_loss.backward()
            optimizer.step()
            
            # Update EMA teacher (if not done in training_step)
            if not (use_consistency_only or use_distillation):
                model.update_ema_teacher()
        
        # Print epoch statistics
        if use_consistency_only:
            print(f'Epoch {epoch_idx + 1} | Consistency Loss: {np.mean(total_losses):.4f}')
        else:
            print(f'Epoch {epoch_idx + 1} | Total Loss: {np.mean(total_losses):.4f} | '
                  f'Consistency: {np.mean(consistency_losses):.4f} | '
                  f'Distillation: {np.mean(distillation_losses):.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(train_config['task_name'], 
                                            'consistency_controlnet_distilled.pth')
        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.student.state_dict(),
            'ema_teacher_state_dict': model.ema_teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': model_config,
        }, checkpoint_path)
    
    print('Distillation training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distilled Consistency ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)