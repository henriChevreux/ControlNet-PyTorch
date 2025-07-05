import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.consistency_controlnet import ConsistencyControlNet
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
        num_timesteps=diffusion_config['num_timesteps']
    )
    
    # Create dataset
    mnist = MnistDataset('train',
                         im_path=dataset_config['im_path'],
                         return_hints=True)
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True)
    
    # Create model
    model = ConsistencyControlNet(model_config).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(train_config['task_name'], 'consistency_controlnet_ckpt.pth')
    if os.path.exists(checkpoint_path):
        print('Loading checkpoint')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Training parameters
    num_epochs = train_config.get('consistency_epochs', 10)
    optimizer = Adam(model.parameters(), lr=train_config.get('consistency_lr', 0.0001))
    
    # Training loop
    for epoch_idx in range(num_epochs):
        losses = []
        for im, hint in tqdm(mnist_loader):
            optimizer.zero_grad()
            
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Sample random timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images
            x_t, _ = scheduler.add_noise(im, t)
            
            # Compute consistency loss
            loss = scheduler.consistency_loss(model, x_t, t, hint, im)
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch_idx + 1} | Loss: {np.mean(losses):.4f}')
        
        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)
    
    print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Consistency ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)