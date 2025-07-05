import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.consistency_controlnet_distilled import ConsistencyControlNetDistilled
from scheduler.consistency_scheduler import ConsistencyScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    
    # Create dataset
    mnist = MnistDataset('train',
                         im_path=dataset_config['im_path'],
                         return_hints=True)
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True)
    
    # Create distilled model - use ControlNet checkpoint, not DDPM checkpoint
    teacher_ckpt_path = os.path.join(train_config['task_name'], 
                                    train_config['controlnet_ckpt_name'])  # Use DDPM checkpoint
    
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}. "
                               "Please train ControlNet first.")
    
    model = ConsistencyControlNetDistilled(
        model_config, 
        teacher_ckpt_path, 
    ).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Training parameters
    num_epochs = train_config.get('consistency_epochs', 10)
    optimizer = Adam(model.student.parameters(), lr=train_config.get('consistency_lr', 0.0001))
    
    # Training loop
    for epoch_idx in range(num_epochs):
        consistency_losses = []
        distillation_losses = []
        total_losses = []
        
        for im, hint in tqdm(mnist_loader):
            optimizer.zero_grad()
            
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Sample random timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images
            x_t, _ = scheduler.add_noise(im, t)
            
            # Compute distillation loss
            total_loss, cons_loss, dist_loss = model.distillation_loss(x_t, t, hint, im)
            
            total_losses.append(total_loss.item())
            consistency_losses.append(cons_loss.item())
            distillation_losses.append(dist_loss.item())
            
            total_loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch_idx + 1} | Total Loss: {np.mean(total_losses):.4f} | '
              f'Consistency: {np.mean(consistency_losses):.4f} | '
              f'Distillation: {np.mean(distillation_losses):.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(train_config['task_name'], 
                                      'consistency_controlnet_distilled_ckpt.pth')
        torch.save(model.student.state_dict(), checkpoint_path)
    
    print('Distillation training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distilled Consistency ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)