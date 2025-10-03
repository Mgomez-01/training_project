"""
Training script for the inverse cVAE model (S-parameters -> Pattern)

This should be run AFTER training the forward model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import argparse

from data.dataset import get_dataloaders
from models.inverse import InversecVAE
from models.forward import ForwardModelResNet
from models.utils import gumbel_softmax_binary, temperature_schedule


def vae_loss(logits, target_pattern, mu, logvar, kl_weight=0.01):
    """
    VAE loss = Reconstruction + KL divergence
    
    Args:
        logits: [B, 1, H, W]
        target_pattern: [B, 1, H, W] binary {0, 1}
        mu, logvar: [B, latent_dim]
        kl_weight: weight for KL term
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy_with_logits(
        logits, target_pattern, reduction='mean'
    )
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= logits.size(0)  # Normalize by batch size
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


def train_epoch(inverse_model, forward_model, train_loader, optimizer, 
                criterion_S, kl_weight, forward_weight, temperature, device):
    """Train for one epoch"""
    inverse_model.train()
    forward_model.eval()
    
    metrics = {'total': 0, 'recon': 0, 'kl': 0, 'forward': 0}
    
    for batch in tqdm(train_loader, desc='Training'):
        pattern_true = batch['pattern'].to(device)
        S_target = batch['S_params'].to(device)
        
        optimizer.zero_grad()
        
        # Generate pattern from inverse model
        logits, mu, logvar = inverse_model(S_target)
        
        # Gumbel-Softmax (differentiable)
        pattern_soft = gumbel_softmax_binary(logits, temperature=temperature, hard=False)
        
        # VAE loss
        loss_vae, loss_recon, loss_kl = vae_loss(
            logits, pattern_true, mu, logvar, kl_weight=kl_weight
        )
        
        # Forward validation loss
        S_pred = forward_model(pattern_soft)
        loss_forward = criterion_S(S_pred, S_target)
        
        # Total loss
        loss = loss_vae + forward_weight * loss_forward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inverse_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        metrics['total'] += loss.item()
        metrics['recon'] += loss_recon.item()
        metrics['kl'] += loss_kl.item()
        metrics['forward'] += loss_forward.item()
    
    for key in metrics:
        metrics[key] /= len(train_loader)
    
    return metrics


def validate(inverse_model, forward_model, val_loader, criterion_S, 
             kl_weight, forward_weight, device):
    """Validate the model"""
    inverse_model.eval()
    forward_model.eval()
    
    metrics = {'total': 0, 'recon': 0, 'kl': 0, 'forward': 0, 'accuracy': 0}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            pattern_true = batch['pattern'].to(device)
            S_target = batch['S_params'].to(device)
            
            logits, mu, logvar = inverse_model(S_target)
            pattern_binary = (torch.sigmoid(logits) > 0.5).float()
            
            loss_vae, loss_recon, loss_kl = vae_loss(
                logits, pattern_true, mu, logvar, kl_weight=kl_weight
            )
            
            S_pred = forward_model(pattern_binary)
            loss_forward = criterion_S(S_pred, S_target)
            
            loss = loss_vae + forward_weight * loss_forward
            
            # Pattern accuracy
            accuracy = (pattern_binary == pattern_true).float().mean()
            
            metrics['total'] += loss.item()
            metrics['recon'] += loss_recon.item()
            metrics['kl'] += loss_kl.item()
            metrics['forward'] += loss_forward.item()
            metrics['accuracy'] += accuracy.item()
    
    for key in metrics:
        metrics[key] /= len(val_loader)
    
    return metrics


def train_inverse_model(
    arrays_dir,
    data_dir,
    forward_checkpoint='checkpoints/forward/best_model.pt',
    save_dir='checkpoints/inverse',
    epochs=200,
    batch_size=32,
    lr=1e-4,
    latent_dim=128,
    kl_weight=0.001,
    forward_weight=10.0,
    temperature_start=5.0,
    temperature_end=0.5,
    device='cuda'
):
    """
    Main training function for inverse cVAE model
    
    Args:
        arrays_dir: Path to sparse array files
        data_dir: Path to S-parameter .pkl files
        forward_checkpoint: Path to trained forward model
        save_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        latent_dim: Dimension of latent space
        kl_weight: Weight for KL divergence loss
        forward_weight: Weight for forward validation loss
        temperature_start: Starting temperature for Gumbel-Softmax
        temperature_end: Ending temperature for Gumbel-Softmax
        device: 'cuda' or 'cpu'
    """
    # Setup
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader, S_mean, S_std = get_dataloaders(
        arrays_dir, data_dir, batch_size=batch_size
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Load forward model (frozen)
    print(f"\nLoading forward model from: {forward_checkpoint}")
    forward_model = ForwardModelResNet().to(device)
    checkpoint = torch.load(forward_checkpoint, map_location=device)
    forward_model.load_state_dict(checkpoint['model_state_dict'])
    forward_model.eval()
    for param in forward_model.parameters():
        param.requires_grad = False
    print("✓ Forward model loaded and frozen")
    
    # Inverse model
    inverse_model = InversecVAE(latent_dim=latent_dim).to(device)
    print(f"\nInitialized InversecVAE with latent_dim={latent_dim}")
    
    # Count parameters
    total_params = sum(p.numel() for p in inverse_model.parameters())
    trainable_params = sum(p.numel() for p in inverse_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(inverse_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    criterion_S = nn.MSELoss()
    best_val_loss = float('inf')
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print(f"Hyperparameters: kl_weight={kl_weight}, forward_weight={forward_weight}")
    print(f"Temperature annealing: {temperature_start} -> {temperature_end}")
    print("="*80)
    
    for epoch in range(epochs):
        # Temperature annealing for Gumbel-Softmax
        temperature = temperature_schedule(
            epoch, epochs, temperature_start, temperature_end
        )
        
        print(f"\nEpoch {epoch+1}/{epochs} (Temperature: {temperature:.3f})")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            inverse_model, forward_model, train_loader, optimizer,
            criterion_S, kl_weight, forward_weight, temperature, device
        )
        
        # Validate
        val_metrics = validate(
            inverse_model, forward_model, val_loader,
            criterion_S, kl_weight, forward_weight, device
        )
        
        # Scheduler step
        scheduler.step(val_metrics['total'])
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train: Total={train_metrics['total']:.4f}, "
              f"Recon={train_metrics['recon']:.4f}, "
              f"KL={train_metrics['kl']:.4f}, "
              f"Forward={train_metrics['forward']:.4f}")
        print(f"Val:   Total={val_metrics['total']:.4f}, "
              f"Recon={val_metrics['recon']:.4f}, "
              f"KL={val_metrics['kl']:.4f}, "
              f"Forward={val_metrics['forward']:.4f}, "
              f"Accuracy={val_metrics['accuracy']:.4f}")
        print(f"LR:    {current_lr:.2e}")
        
        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': inverse_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total'],
                'val_metrics': val_metrics,
                'config': {
                    'latent_dim': latent_dim,
                    'kl_weight': kl_weight,
                    'forward_weight': forward_weight,
                }
            }, f'{save_dir}/best_model.pt')
            print(f"✓ Saved best model")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': inverse_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total'],
                'val_metrics': val_metrics,
                'config': {
                    'latent_dim': latent_dim,
                    'kl_weight': kl_weight,
                    'forward_weight': forward_weight,
                }
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pt')
    
    print("\n" + "="*80)
    print(f'Training complete! Best validation loss: {best_val_loss:.4f}')
    print(f"Model saved to: {save_dir}/best_model.pt")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train inverse cVAE model')
    parser.add_argument('--arrays_dir', type=str, required=True,
                        help='Path to sparse array directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to S-parameter data directory')
    parser.add_argument('--forward_checkpoint', type=str, 
                        default='checkpoints/forward/best_model.pt',
                        help='Path to trained forward model checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints/inverse',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--kl_weight', type=float, default=0.001,
                        help='Weight for KL divergence loss')
    parser.add_argument('--forward_weight', type=float, default=10.0,
                        help='Weight for forward validation loss')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    train_inverse_model(
        arrays_dir=args.arrays_dir,
        data_dir=args.data_dir,
        forward_checkpoint=args.forward_checkpoint,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        forward_weight=args.forward_weight,
        device=args.device
    )
