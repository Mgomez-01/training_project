"""
Evaluation script to assess model performance on validation set
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import json

from data.dataset import get_dataloaders
from models import ForwardModelResNet, InversecVAE


def evaluate_forward_model(model, val_loader, device='cuda'):
    """
    Evaluate forward model performance
    
    Returns:
        dict with metrics
    """
    model.eval()
    device = torch.device(device)
    
    all_errors = []
    param_errors = {f'S{i}{j}': [] for i in [1,2] for j in [1,2]}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating Forward Model'):
            pattern = batch['pattern'].to(device)
            S_true = batch['S_params'].to(device)
            
            S_pred = model(pattern)
            
            # Overall MSE
            mse = torch.mean((S_pred - S_true) ** 2, dim=[1, 2])
            all_errors.extend(mse.cpu().numpy())
            
            # Per-parameter MSE
            param_names = ['S11', 'S21', 'S22', 'S12']
            for i, param in enumerate(param_names):
                param_mse = torch.mean((S_pred[:, :, i] - S_true[:, :, i]) ** 2, dim=1)
                param_errors[param].extend(param_mse.cpu().numpy())
    
    metrics = {
        'overall_mse': float(np.mean(all_errors)),
        'overall_std': float(np.std(all_errors)),
        'overall_median': float(np.median(all_errors)),
        'param_mse': {k: float(np.mean(v)) for k, v in param_errors.items()},
    }
    
    return metrics


def evaluate_inverse_model(inverse_model, forward_model, val_loader, 
                           device='cuda', n_samples=5):
    """
    Evaluate inverse model performance
    
    Args:
        inverse_model: Trained inverse cVAE
        forward_model: Trained forward model
        val_loader: Validation data loader
        device: Device to use
        n_samples: Number of samples to generate per target
    
    Returns:
        dict with metrics
    """
    inverse_model.eval()
    forward_model.eval()
    device = torch.device(device)
    
    reconstruction_errors = []
    forward_errors = []
    pattern_accuracies = []
    diversities = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating Inverse Model'):
            pattern_true = batch['pattern'].to(device)
            S_target = batch['S_params'].to(device)
            batch_size = pattern_true.size(0)
            
            # Generate multiple samples per target
            all_patterns = []
            for _ in range(n_samples):
                logits, mu, logvar = inverse_model(S_target)
                pattern_pred = (torch.sigmoid(logits) > 0.5).float()
                all_patterns.append(pattern_pred)
                
                # Pattern reconstruction accuracy
                accuracy = (pattern_pred == pattern_true).float().mean()
                pattern_accuracies.append(accuracy.item())
            
            # Stack patterns
            all_patterns = torch.cat(all_patterns, dim=0)  # [n_samples*B, 1, H, W]
            
            # Forward validation
            S_pred = forward_model(all_patterns)
            S_target_repeated = S_target.repeat(n_samples, 1, 1)
            
            forward_error = torch.mean((S_pred - S_target_repeated) ** 2, dim=[1, 2])
            forward_errors.extend(forward_error.cpu().numpy())
            
            # Diversity: pairwise pattern differences within each batch
            for i in range(batch_size):
                batch_patterns = all_patterns[i::batch_size]  # [n_samples, 1, H, W]
                pairwise_diffs = []
                for j in range(n_samples):
                    for k in range(j+1, n_samples):
                        diff = torch.mean((batch_patterns[j] != batch_patterns[k]).float())
                        pairwise_diffs.append(diff.item())
                if pairwise_diffs:
                    diversities.append(np.mean(pairwise_diffs))
    
    metrics = {
        'pattern_accuracy': float(np.mean(pattern_accuracies)),
        'forward_mse': float(np.mean(forward_errors)),
        'forward_std': float(np.std(forward_errors)),
        'forward_median': float(np.median(forward_errors)),
        'diversity': float(np.mean(diversities)),
        'best_forward_mse': float(np.min(forward_errors)),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--arrays_dir', type=str, required=True,
                        help='Path to sparse array directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to S-parameter data directory')
    parser.add_argument('--forward_checkpoint', type=str,
                        default='checkpoints/forward/best_model.pt',
                        help='Path to forward model checkpoint')
    parser.add_argument('--inverse_checkpoint', type=str,
                        default='checkpoints/inverse/best_model.pt',
                        help='Path to inverse model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of samples per target for inverse model')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    _, val_loader, _, _ = get_dataloaders(
        args.arrays_dir, args.data_dir, 
        batch_size=args.batch_size,
        num_workers=4
    )
    print(f"Validation samples: {len(val_loader.dataset)}\n")
    
    results = {}
    
    # Evaluate forward model
    if Path(args.forward_checkpoint).exists():
        print("="*60)
        print("Evaluating Forward Model")
        print("="*60)
        
        forward_model = ForwardModelResNet().to(device)
        checkpoint = torch.load(args.forward_checkpoint, map_location=device, weights_only=False)
        forward_model.load_state_dict(checkpoint['model_state_dict'])
        
        forward_metrics = evaluate_forward_model(forward_model, val_loader, device)
        results['forward'] = forward_metrics
        
        print("\nForward Model Metrics:")
        print(f"  Overall MSE:    {forward_metrics['overall_mse']:.6f}")
        print(f"  Overall Std:    {forward_metrics['overall_std']:.6f}")
        print(f"  Overall Median: {forward_metrics['overall_median']:.6f}")
        print(f"\n  Per-parameter MSE:")
        for param, mse in forward_metrics['param_mse'].items():
            print(f"    {param}: {mse:.6f}")
    else:
        print(f"Forward checkpoint not found: {args.forward_checkpoint}")
        forward_model = None
    
    # Evaluate inverse model
    if Path(args.inverse_checkpoint).exists() and forward_model is not None:
        print("\n" + "="*60)
        print("Evaluating Inverse Model")
        print("="*60)
        
        inv_checkpoint = torch.load(args.inverse_checkpoint, map_location=device, weights_only=False)
        latent_dim = inv_checkpoint['config']['latent_dim']
        inverse_model = InversecVAE(latent_dim=latent_dim).to(device)
        inverse_model.load_state_dict(inv_checkpoint['model_state_dict'])
        
        inverse_metrics = evaluate_inverse_model(
            inverse_model, forward_model, val_loader, 
            device, n_samples=args.n_samples
        )
        results['inverse'] = inverse_metrics
        
        print(f"\nInverse Model Metrics (n_samples={args.n_samples}):")
        print(f"  Pattern Accuracy:  {inverse_metrics['pattern_accuracy']:.4f}")
        print(f"  Forward MSE:       {inverse_metrics['forward_mse']:.6f}")
        print(f"  Forward Std:       {inverse_metrics['forward_std']:.6f}")
        print(f"  Forward Median:    {inverse_metrics['forward_median']:.6f}")
        print(f"  Best Forward MSE:  {inverse_metrics['best_forward_mse']:.6f}")
        print(f"  Diversity:         {inverse_metrics['diversity']:.4f}")
    elif not Path(args.inverse_checkpoint).exists():
        print(f"\nInverse checkpoint not found: {args.inverse_checkpoint}")
    
    # Save results
    if results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Results saved to: {args.output}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
