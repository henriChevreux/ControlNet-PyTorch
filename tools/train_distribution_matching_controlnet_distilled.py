import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset.mnist_dataset import MnistDataset
from dataset.cifar_dataset import CifarDataset
from torch.utils.data import DataLoader
from models.distribution_matching_controlnet import DistributionMatchingControlNetDistilled
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import matplotlib.pyplot as plt
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DMDTrainer:
    def __init__(self, config, teacher_ckpt_path):
        self.config = config
        self.diffusion_config = config['diffusion_params']
        self.dataset_config = config['dataset_params']
        self.model_config = config['model_params']
        self.train_config = config['train_params']
        
        # Create scheduler
        self.scheduler = LinearNoiseScheduler(
            num_timesteps=self.diffusion_config['num_timesteps'],
            beta_start=self.diffusion_config['beta_start'],
            beta_end=self.diffusion_config['beta_end']
        )
        
        # Create model
        self.model = DistributionMatchingControlNetDistilled(
            self.model_config, 
            teacher_ckpt_path, 
        ).to(device)
        self.model.train()
        
        # Optimizer and scheduler
        self.optimizer = Adam(
            self.model.student.parameters(), 
            lr=self.train_config.get('distribution_matching_lr', 0.0001),
            weight_decay=1e-6  # Small weight decay for regularization
        )
        
        # Learning rate scheduler
        num_epochs = self.train_config.get('distribution_matching_epochs', 20)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
    def validate_batch(self, val_loader):
        """Run validation to monitor DMD progress"""
        self.model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for im, hint in val_loader:
                im = im.float().to(device)
                hint = hint.float().to(device)
                
                # Sample random timestep
                t = torch.randint(0, self.diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
                
                # Add noise
                noise = torch.randn_like(im)
                x_t = self.scheduler.add_noise(im, noise, t)
                
                # Get predictions
                total_loss, dist_matching_loss, dist_loss, dist_components = self.model.distillation_loss(x_t, t, hint, im)
                
                # Collect metrics
                val_metrics['total_loss'].append(total_loss.item())
                val_metrics['dist_matching_loss'].append(dist_matching_loss.item())
                val_metrics['distillation_loss'].append(dist_loss.item())
                
                for key, value in dist_components.items():
                    val_metrics[f'val_{key}'].append(value.item())
                
                # Only validate on a few batches
                if len(val_metrics['total_loss']) >= 5:
                    break
        
        self.model.train()
        return {k: np.mean(v) for k, v in val_metrics.items()}
    
    def save_sample_predictions(self, epoch, im, hint, save_dir):
        """Save sample predictions to monitor visual quality"""
        self.model.eval()
        
        with torch.no_grad():
            # Take only first 8 samples
            im_sample = im[:8]
            hint_sample = hint[:8]
            
            # Sample at different timesteps to see denoising capability
            for t_val in [50, 200, 500]:
                t = torch.full((im_sample.shape[0],), t_val).to(device)
                
                noise = torch.randn_like(im_sample)
                x_t = self.scheduler.add_noise(im_sample, noise, t)
                
                # Student prediction
                x_0_pred = self.model.student(x_t, t, hint_sample)
                
                # Teacher prediction for comparison
                x_0_teacher = self.model.get_teacher_prediction(x_t, t, hint_sample)
                
                # Create comparison plot
                fig, axes = plt.subplots(4, 8, figsize=(16, 8))
                
                for i in range(8):
                    # Original
                    axes[0, i].imshow(im_sample[i, 0].cpu().numpy(), cmap='gray')
                    axes[0, i].set_title('Original')
                    axes[0, i].axis('off')
                    
                    # Noisy
                    axes[1, i].imshow(x_t[i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title(f'Noisy t={t_val}')
                    axes[1, i].axis('off')
                    
                    # Student prediction
                    axes[2, i].imshow(x_0_pred[i, 0].cpu().numpy(), cmap='gray')
                    axes[2, i].set_title('Student')
                    axes[2, i].axis('off')
                    
                    # Teacher prediction
                    axes[3, i].imshow(x_0_teacher[i, 0].cpu().numpy(), cmap='gray')
                    axes[3, i].set_title('Teacher')
                    axes[3, i].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_t_{t_val}_predictions.png'))
                plt.close()
        
        self.model.train()
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch with detailed logging"""
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (im, hint) in enumerate(pbar):
            self.optimizer.zero_grad()
            
            im = im.float().to(device)
            hint = hint.float().to(device)
            
            # Sample random timestep
            t = torch.randint(0, self.diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images
            noise = torch.randn_like(im)
            x_t = self.scheduler.add_noise(im, noise, t)
            
            # Compute distillation loss
            total_loss, dist_matching_loss, dist_loss, dist_components = self.model.distillation_loss(x_t, t, hint, im)
            
            # Check for NaN values
            if torch.isnan(total_loss) or torch.isnan(dist_matching_loss) or torch.isnan(dist_loss):
                print(f"Warning: NaN detected in loss computation at batch {batch_idx}!")
                print(f"Total loss: {total_loss}, Dist matching: {dist_matching_loss}, Distillation: {dist_loss}")
                continue
            
            # Collect detailed metrics
            epoch_metrics['total_loss'].append(total_loss.item())
            epoch_metrics['dist_matching_loss'].append(dist_matching_loss.item())
            epoch_metrics['distillation_loss'].append(dist_loss.item())
            
            # Log distribution components
            for key, value in dist_components.items():
                epoch_metrics[key].append(value.item())
            
            total_loss.backward()
            
            # Gradient clipping with monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), max_norm=1.0)
            epoch_metrics['grad_norm'].append(grad_norm.item())
            
            self.optimizer.step()
            
            # Update progress bar with key metrics
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'Total': f"{total_loss.item():.4f}",
                    'DMD': f"{dist_matching_loss.item():.4f}",
                    'Distill': f"{dist_loss.item():.4f}",
                    'Feature': f"{dist_components.get('feature_dist', 0):.4f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        num_epochs = self.train_config.get('distribution_matching_epochs', 20)
        save_dir = self.train_config['task_name']
        
        # Create output directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
        
        print(f"Starting DMD training for {num_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.student.parameters())} trainable parameters")
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate_batch(val_loader)
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Log metrics
            for key, value in train_metrics.items():
                self.metrics[f'train_{key}'].append(value)
            
            for key, value in val_metrics.items():
                self.metrics[key].append(value)
            
            # Print epoch summary
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'  Total Loss: {train_metrics["total_loss"]:.4f}')
            print(f'  Distribution Matching: {train_metrics["dist_matching_loss"]:.4f}')
            print(f'  Distillation: {train_metrics["distillation_loss"]:.4f}')
            print(f'  Feature Dist: {train_metrics.get("feature_dist", 0):.4f}')
            print(f'  Wasserstein: {train_metrics.get("wasserstein", 0):.4f}')
            print(f'  Gram: {train_metrics.get("gram", 0):.4f}')
            print(f'  Pixel: {train_metrics.get("pixel", 0):.4f}')
            print(f'  Grad Norm: {train_metrics.get("grad_norm", 0):.4f}')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if val_metrics:
                print(f'  Val Total Loss: {val_metrics.get("total_loss", 0):.4f}')
            
            # Save sample predictions every 5 epochs
            if (epoch + 1) % 1 == 0:
                sample_batch = next(iter(train_loader))
                self.save_sample_predictions(epoch + 1, sample_batch[0][:8].to(device), 
                                           sample_batch[1][:8].to(device), 
                                           os.path.join(save_dir, 'samples'))
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, 'distribution_matching_controlnet_distilled_ckpt.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': dict(self.metrics),
                'config': self.config
            }, checkpoint_path)
            
            # Save best model based on distribution matching loss
            if len(self.metrics['train_dist_matching_loss']) > 0:
                if train_metrics["dist_matching_loss"] == min(self.metrics['train_dist_matching_loss']):
                    best_path = os.path.join(save_dir, 'best_distribution_matching_model.pth')
                    torch.save(self.model.student.state_dict(), best_path)
                    print(f'  -> Saved best model (DMD loss: {train_metrics["dist_matching_loss"]:.4f})')
        
        print('Distribution matching distillation training completed!')
        return self.metrics


def train(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    train_config = config['train_params']
    dataset_config = config['dataset_params']
    
    # Create datasets
    if train_config['task_name'] == 'mnist':
        train_dataset = MnistDataset('train',
                                    im_path=dataset_config['im_path'],
                                    return_hints=True)
    elif train_config['task_name'] == 'cifar10':
        train_dataset = CifarDataset('train',
                                    im_path=dataset_config['im_path'],
                                    return_hints=True,
                                    download=dataset_config['download'])
    else:
        raise ValueError(f"Invalid dataset name: {train_config['task_name']}")
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=train_config['batch_size'], 
                             shuffle=True, 
                             num_workers=4)
    
    # Create validation dataset (optional) - use test split for validation
    if train_config['task_name'] == 'mnist':
        val_dataset = MnistDataset('test',
                                  im_path=dataset_config['im_test_path'],
                                  return_hints=True)
    elif train_config['task_name'] == 'cifar10':
        val_dataset = CifarDataset('test',
                                  im_path=dataset_config['im_test_path'],
                                  return_hints=True,
                                  download=dataset_config['download'])
    else:
        raise ValueError(f"Invalid dataset name: {train_config['task_name']}")
    
    val_loader = DataLoader(val_dataset, 
                           batch_size=train_config['batch_size'], 
                           shuffle=False, 
                           num_workers=4)
    
    # Teacher checkpoint path
    teacher_ckpt_path = os.path.join(train_config['task_name'], 
                                    train_config['controlnet_ckpt_name'])
    
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}. "
                               "Please train ControlNet first.")
    
    # Create trainer and start training
    trainer = DMDTrainer(config, teacher_ckpt_path)
    metrics = trainer.train(train_loader, val_loader)
    
    # Plot training curves
    plot_training_curves(metrics, train_config['task_name'])


def plot_training_curves(metrics, save_dir):
    """Plot training curves for analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(metrics['train_total_loss'], label='Total Loss')
    axes[0, 0].plot(metrics['train_dist_matching_loss'], label='DMD Loss')
    axes[0, 0].plot(metrics['train_distillation_loss'], label='Distillation Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    
    # Distribution components
    if 'train_feature_dist' in metrics:
        axes[0, 1].plot(metrics['train_feature_dist'], label='Feature Dist')
        axes[0, 1].plot(metrics['train_wasserstein'], label='Wasserstein')
        axes[0, 1].plot(metrics['train_gram'], label='Gram Matrix')
        axes[0, 1].set_title('Distribution Components')
        axes[0, 1].legend()
    
    # Pixel loss
    if 'train_pixel' in metrics:
        axes[0, 2].plot(metrics['train_pixel'])
        axes[0, 2].set_title('Pixel Loss')
    
    # Gradient norm
    if 'train_grad_norm' in metrics:
        axes[1, 0].plot(metrics['train_grad_norm'])
        axes[1, 0].set_title('Gradient Norm')
    
    # Validation losses (if available)
    if 'val_total_loss' in metrics:
        axes[1, 1].plot(metrics['val_total_loss'], label='Val Total')
        axes[1, 1].plot(metrics['val_dist_matching_loss'], label='Val DMD')
        axes[1, 1].set_title('Validation Losses')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distilled Distribution Matching ControlNet')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)