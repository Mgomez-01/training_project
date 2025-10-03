"""
Hyperparameter tuning script for inverse cVAE

This script tests different combinations of kl_weight and forward_weight
to find the best configuration for your dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from pathlib import Path
import argparse
import itertools

from data.dataset import get_dataloaders
from models.inverse import InversecVAE
from models.forward import ForwardModelResNet
from models.utils import gumbel_softmax_binary
import torch.nn.functional as F


def vae_loss(logits, target_pattern, mu, logvar, kl_weight=0.01):
    """VAE loss function"""
    recon_loss = F.binary_cross_entropy_with_logits(
        logits, target_pattern, reduction='mean'
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= logits.size(0)
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


def quick_train_eval(inverse_model, forward_model, train_loader, val_loader,
                      kl_weight, forward_weight, n_epochs=10, device='cuda'):
    """
    Quick training for hyperparameter evaluation
    
    Returns validation metrics
    """
    optimizer = optim.AdamW(inverse_model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion_S = nn.MSELoss()
    
    inverse_model.train()
    forward_model.eval()
    
    # Train for a few epochs
    for epoch in range(n_epochs):
        for batch in train_loader:
            pattern_true = batch['pattern'].to(device)
            S_target = batch['S_params'].to(device)
            
            optimizer.zero_grad()
            
            logits, mu, logvar = inverse_model(S_target)
            pattern_soft = gumbel_softmax_binary(logits, temperature=1.0, hard=False)
            
            loss_vae, loss_recon, loss_kl = vae_loss(
                logits, pattern_true, mu, logvar, kl_weight=kl_weight
            )
            
            S_pred = forward_model(pattern_soft)
            loss_forward = criterion_S(S_pred, S_target)
            
            loss = loss_vae + forward_weight * loss_forward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inverse_model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Evaluate on validation set
    inverse_model.eval()
    val_metrics = {'total': 0, 'recon': 0, 'kl': 0, 'forward': 0, 'accuracy': 0}
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
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
            accuracy = (pattern_binary == pattern_true).float().mean()
            
            val_metrics['total'] += loss.item()
            val_metrics['recon'] += loss_recon.item()
            val_metrics['kl'] += loss_kl.item()
            val_metrics['forward'] += loss_forward.item()
            val_metrics['accuracy'] += accuracy.item()
            n_batches += 1
    
    for key in val_metrics:
        val_metrics[key] /= n_batches
    
    return val_metrics


def tune_hyperparameters(
    arrays_dir,
    data_dir,
    forward_checkpoint,
    kl_weights=[0.0001, 0.0005, 0.001, 0.005],
    forward_weights=[5.0, 10.0, 20.0],
    latent_dim=128,
    n_epochs=10,
    batch_size=32,
    output_file='tuning_results.json',
    device='cuda'
):
    """
    Test different hyperparameter combinations
    
    Args:
        arrays_dir: Path to array files
        data_dir: Path to data files
        forward_checkpoint: Path to trained forward model
        kl_weights: List of KL weights to try
        forward_weights: List of forward weights to try
        latent_dim: Latent dimension
        n_epochs: Number of epochs for quick training
        batch_size: Batch size
        output_file: Where to save results
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(
        arrays_dir, data_dir, batch_size=batch_size
    )
    
    # Load forward model (frozen)
    print(f"Loading forward model from: {forward_checkpoint}")
    forward_model = ForwardModelResNet().to(device)
    checkpoint = torch.load(forward_checkpoint, map_location=device)
    forward_model.load_state_dict(checkpoint['model_state_dict'])
    forward_model.eval()
    for param in forward_model.parameters():
        param.requires_grad = False
    
    # Grid search
    results = []
    total_combinations = len(kl_weights) * len(forward_weights)
    
    print("\n" + "="*80)
    print(f"HYPERPARAMETER TUNING")
    print(f"Testing {total_combinations} combinations")
    print(f"Quick training: {n_epochs} epochs per combination")
    print("="*80 + "\n")
    
    for i, (kl_w, fwd_w) in enumerate(itertools.product(kl_weights, forward_weights)):
        print(f"\n[{i+1}/{total_combinations}] Testing: kl_weight={kl_w}, forward_weight={fwd_w}")
        print("-" * 60)
        
        # Initialize fresh model
        inverse_model = InversecVAE(latent_dim=latent_dim).to(device)
        
        # Quick training
        metrics = quick_train_eval(
            inverse_model, forward_model, train_loader, val_loader,
            kl_w, fwd_w, n_epochs, device
        )
        
        result = {
            'kl_weight': kl_w,
            'forward_weight': fwd_w,
            'metrics': metrics
        }
        results.append(result)
        
        print(f"  Total Loss:    {metrics['total']:.4f}")
        print(f"  Recon Loss:    {metrics['recon']:.4f}")
        print(f"  KL Loss:       {metrics['kl']:.4f}")
        print(f"  Forward Loss:  {metrics['forward']:.4f}")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    
    # Sort by total loss
    results.sort(key=lambda x: x['metrics']['total'])
    
    # Print summary
    print("\n" + "="*80)
    print("TUNING RESULTS (Top 5)")
    print("="*80)
    
    for i, result in enumerate(results[:5]):
        print(f"\nRank {i+1}:")
        print(f"  kl_weight:     {result['kl_weight']}")
        print(f"  forward_weight: {result['forward_weight']}")
        print(f"  Total Loss:    {result['metrics']['total']:.4f}")
        print(f"  Forward Loss:  {result['metrics']['forward']:.4f}")
        print(f"  Accuracy:      {result['metrics']['accuracy']:.4f}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Full results saved to: {output_file}")
    print(f"\nRecommended hyperparameters:")
    best = results[0]
    print(f"  --kl_weight {best['kl_weight']}")
    print(f"  --forward_weight {best['forward_weight']}")
    print(f"{'='*80}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Tune hyperparameters for inverse cVAE')
    parser.add_argument('--arrays_dir', type=str, required=True,
                        help='Path to sparse array directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to S-parameter data directory')
    parser.add_argument('--forward_checkpoint', type=str,
                        default='checkpoints/forward/best_model.pt',
                        help='Path to trained forward model')
    parser.add_argument('--kl_weights', type=float, nargs='+',
                        default=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                        help='List of KL weights to try')
    parser.add_argument('--forward_weights', type=float, nargs='+',
                        default=[5.0, 10.0, 15.0, 20.0],
                        help='List of forward weights to try')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs per configuration')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output', type=str, default='tuning_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    tune_hyperparameters(
        arrays_dir=args.arrays_dir,
        data_dir=args.data_dir,
        forward_checkpoint=args.forward_checkpoint,
        kl_weights=args.kl_weights,
        forward_weights=args.forward_weights,
        latent_dim=args.latent_dim,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        output_file=args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
