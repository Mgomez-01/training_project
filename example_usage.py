"""
Example usage script showing how to use the models programmatically
"""

import torch
import pandas as pd
import numpy as np

# Import your modules
from data.dataset import get_dataloaders
from models import ForwardModelResNet, InversecVAE
from generate import PatternGenerator


def example_1_load_data():
    """Example: Load and inspect the dataset"""
    print("="*60)
    print("Example 1: Loading Data")
    print("="*60)
    
    arrays_dir = "python/deep_archive/arrays/"
    data_dir = "python/deep_archive/data/"
    
    train_loader, val_loader, S_mean, S_std = get_dataloaders(
        arrays_dir, data_dir, batch_size=8
    )
    
    print(f"\nDataset info:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Normalization mean: {S_mean[0]}")
    print(f"  Normalization std: {S_std[0]}")
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Patterns: {batch['pattern'].shape}")  # [B, 1, 48, 32]
    print(f"  S-params: {batch['S_params'].shape}")  # [B, 201, 4]
    
    return train_loader, val_loader


def example_2_forward_model():
    """Example: Use forward model for prediction"""
    print("\n" + "="*60)
    print("Example 2: Forward Model Prediction")
    print("="*60)
    
    # Create a dummy pattern
    pattern = torch.rand(1, 1, 48, 32) > 0.5  # Random binary pattern
    pattern = pattern.float()
    
    # Load model
    model = ForwardModelResNet()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        S_pred = model(pattern)
    
    print(f"\nPrediction shape: {S_pred.shape}")  # [1, 201, 4]
    print(f"Sample S-parameters (first 5 freqs):")
    print(S_pred[0, :5, :])


def example_3_inverse_model():
    """Example: Generate pattern from target S-parameters"""
    print("\n" + "="*60)
    print("Example 3: Inverse Model Generation")
    print("="*60)
    
    # Create dummy target S-parameters
    target_S = torch.randn(1, 201, 4)
    
    # Load model
    model = InversecVAE(latent_dim=128)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate pattern
    model.eval()
    with torch.no_grad():
        logits, mu, logvar = model(target_S)
        pattern = (torch.sigmoid(logits) > 0.5).float()
    
    print(f"\nGenerated pattern shape: {pattern.shape}")  # [1, 1, 48, 32]
    print(f"Latent mean: {mu[0, :5]}")
    print(f"Latent logvar: {logvar[0, :5]}")
    print(f"Pattern sparsity: {pattern.mean().item():.2%}")


def example_4_full_pipeline():
    """Example: Complete generation pipeline"""
    print("\n" + "="*60)
    print("Example 4: Complete Generation Pipeline")
    print("="*60)
    
    # This would use trained models
    print("\nTo run the complete pipeline:")
    print("1. Train forward model:")
    print("   python train_forward.py --arrays_dir <path> --data_dir <path>")
    print("\n2. Train inverse model:")
    print("   python train_inverse.py --arrays_dir <path> --data_dir <path>")
    print("\n3. Generate patterns:")
    print("   python generate.py --target_file <path>")
    
    print("\nOr use programmatically:")
    print("""
    generator = PatternGenerator(
        inverse_checkpoint='checkpoints/inverse/best_model.pt',
        forward_checkpoint='checkpoints/forward/best_model.pt',
        normalization_file='checkpoints/forward/normalization.pt'
    )
    
    target = pd.read_pickle('target.pkl')
    candidates = generator.generate_candidates(target, n_candidates=50)
    generator.save_patterns(candidates[:10], 'output_dir')
    """)


def example_5_custom_loss():
    """Example: Custom loss function for specific optimization goals"""
    print("\n" + "="*60)
    print("Example 5: Custom Loss Functions")
    print("="*60)
    
    print("\nExample: Emphasize specific frequency range")
    print("""
    # In train_inverse.py, modify the forward loss:
    
    # Standard MSE
    loss_forward = criterion_S(S_pred, S_target)
    
    # Weighted MSE for specific frequencies
    freq_weights = torch.ones(201)
    freq_weights[50:150] = 2.0  # Emphasize center frequencies
    weighted_loss = (freq_weights.unsqueeze(0).unsqueeze(-1) * 
                     (S_pred - S_target)**2).mean()
    
    # Or emphasize specific S-parameters
    param_weights = torch.tensor([1.0, 2.0, 1.0, 1.0])  # Emphasize S21
    weighted_loss = (param_weights.unsqueeze(0).unsqueeze(0) * 
                     (S_pred - S_target)**2).mean()
    """)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Pattern Generation - Example Usage")
    print("="*60)
    
    try:
        # Note: These examples use dummy data
        # Replace with actual paths for real usage
        
        example_2_forward_model()
        example_3_inverse_model()
        example_4_full_pipeline()
        example_5_custom_loss()
        
        print("\n" + "="*60)
        print("Examples complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\nNote: Some examples may fail without trained models.")
        print(f"Error: {e}")
        print("\nThis is expected if you haven't trained the models yet.")
