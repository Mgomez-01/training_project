"""
Training script for the forward model (Pattern -> S-parameters)

This should be run first to create a reliable forward model that can
validate the inverse model during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse

from data.dataset import get_dataloaders
from models.forward import ForwardModel, ForwardModelResNet


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        pattern = batch['pattern'].to(device)
        S_true = batch['S_params'].to(device)
        
        optimizer.zero_grad()
        S_pred = model(pattern)
        loss = criterion(S_pred, S_true)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            pattern = batch['pattern'].to(device)
            S_true = batch['S_params'].to(device)
            
            S_pred = model(pattern)
            loss = criterion(S_pred, S_true)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_forward_model(
    arrays_dir,
    data_dir,
    save_dir='checkpoints/forward',
    epochs=100,
    batch_size=32,
    lr=1e-3,
    use_resnet=True,
    device='cuda'
):
    """
    Main training function for forward model
    
    Args:
        arrays_dir: Path to sparse array files
        data_dir: Path to S-parameter .pkl files
        save_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        use_resnet: Whether to use ResNet architecture (recommended)
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
    
    # Model
    if use_resnet:
        model = ForwardModelResNet().to(device)
        print("\nUsing ForwardModelResNet")
    else:
        model = ForwardModel().to(device)
        print("\nUsing ForwardModel (simple)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    # Save normalization stats
    torch.save({
        'S_mean': S_mean,
        'S_std': S_std
    }, f'{save_dir}/normalization.pt')
    print(f"\nSaved normalization stats to {save_dir}/normalization.pt")
    
    # Training loop
    best_val_loss = float('inf')
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"LR:         {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, f'{save_dir}/best_model.pt')
            print(f"✓ Saved best model (val_loss={val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pt')
    
    print("\n" + "="*60)
    print(f'Training complete! Best validation loss: {best_val_loss:.6f}')
    print(f"Model saved to: {save_dir}/best_model.pt")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train forward model')
    parser.add_argument('--arrays_dir', type=str, required=True,
                        help='Path to sparse array directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to S-parameter data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/forward',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--no_resnet', action='store_true',
                        help='Use simple model instead of ResNet')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    train_forward_model(
        arrays_dir=args.arrays_dir,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_resnet=not args.no_resnet,
        device=args.device
    )
