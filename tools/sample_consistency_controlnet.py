import torch
import yaml
import argparse
import os
from tqdm import tqdm
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.consistency_controlnet import ConsistencyControlNet
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
    model = ConsistencyControlNet(model_config).to(device)
    model.eval()
    
    # Load checkpoint
    checkpoint_path = os.path.join(train_config['task_name'], 'consistency_controlnet_ckpt.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Create test dataset
    test_dataset = MnistDataset('test',
                               im_path=dataset_config['im_test_path'],
                               return_hints=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Sampling
    samples = []
    hints = []
    
    with torch.no_grad():
        for im, hint in tqdm(test_loader):
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Start from pure noise
            x_t = torch.randn_like(im).to(device)
            t = torch.full((im.shape[0],), scheduler.num_timesteps - 1, device=device)
            
            # Single-step sampling
            x_0_pred = scheduler.sample(model, x_t, t, hint)
            
            samples.append(x_0_pred.cpu())
            hints.append(hint.cpu())
            
            if len(samples) >= 4:  # Generate 4 batches
                break
    
    # Save results
    samples = torch.cat(samples, dim=0)
    hints = torch.cat(hints, dim=0)
    
    # Save grid
    grid = vutils.make_grid(samples, nrow=8, normalize=True)
    vutils.save_image(grid, os.path.join(train_config['task_name'], 'consistency_samples.png'))
    
    # Save hints
    grid_hints = vutils.make_grid(hints, nrow=8, normalize=True)
    vutils.save_image(grid_hints, os.path.join(train_config['task_name'], 'consistency_hints.png'))
    
    print('Sampling completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from Consistency ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    sample(args)