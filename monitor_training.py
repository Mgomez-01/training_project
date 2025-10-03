"""
Training monitor - Track and visualize training progress in real-time

This script monitors training checkpoints and displays progress.
Useful for long training runs.
"""

import json
import time
from pathlib import Path
import argparse


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def monitor_training(checkpoint_dir, interval=30):
    """
    Monitor training progress by watching checkpoint directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        interval: Check interval in seconds
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    print("="*80)
    print("TRAINING MONITOR")
    print("="*80)
    print(f"Watching: {checkpoint_dir}")
    print(f"Update interval: {interval}s")
    print(f"Press Ctrl+C to stop")
    print("="*80)
    
    last_checkpoint = None
    start_time = time.time()
    
    try:
        while True:
            # Find latest checkpoint
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                
                if latest != last_checkpoint:
                    # New checkpoint found
                    import torch
                    ckpt = torch.load(latest, map_location='cpu')
                    
                    elapsed = time.time() - start_time
                    epoch = ckpt.get('epoch', 0) + 1
                    train_loss = ckpt.get('train_loss', 0)
                    val_loss = ckpt.get('val_loss', 0)
                    
                    print(f"\n[{format_time(elapsed)}] Epoch {epoch}")
                    print(f"  Checkpoint: {latest.name}")
                    print(f"  Train Loss: {train_loss:.6f}")
                    print(f"  Val Loss:   {val_loss:.6f}")
                    
                    # Check for best model
                    best_model = checkpoint_dir / "best_model.pt"
                    if best_model.exists():
                        best_ckpt = torch.load(best_model, map_location='cpu')
                        best_val = best_ckpt.get('val_loss', float('inf'))
                        
                        if val_loss <= best_val:
                            print(f"  ✓ NEW BEST MODEL!")
                    
                    last_checkpoint = latest
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("="*80)


def summarize_training(checkpoint_dir):
    """
    Summarize training progress from checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    print("="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Directory: {checkpoint_dir}\n")
    
    # Load best model
    best_model = checkpoint_dir / "best_model.pt"
    if best_model.exists():
        import torch
        ckpt = torch.load(best_model, map_location='cpu')
        
        print("Best Model:")
        print(f"  Epoch:      {ckpt.get('epoch', 0) + 1}")
        print(f"  Train Loss: {ckpt.get('train_loss', 0):.6f}")
        print(f"  Val Loss:   {ckpt.get('val_loss', 0):.6f}")
        
        # Show config if available
        if 'config' in ckpt:
            print(f"\n  Configuration:")
            for key, val in ckpt['config'].items():
                print(f"    {key}: {val}")
    else:
        print("No best model found!")
    
    # Count checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    print(f"\nCheckpoints: {len(checkpoints)} saved")
    
    # Show progression
    if len(checkpoints) >= 2:
        print("\nTraining Progression (last 5 checkpoints):")
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        import torch
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12}")
        print("-" * 35)
        
        for ckpt_file in checkpoints[-5:]:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            epoch = ckpt.get('epoch', 0) + 1
            train = ckpt.get('train_loss', 0)
            val = ckpt.get('val_loss', 0)
            print(f"{epoch:<8} {train:<12.6f} {val:<12.6f}")
    
    print("="*80)


def compare_models(forward_dir, inverse_dir):
    """
    Compare forward and inverse model performance
    
    Args:
        forward_dir: Forward model checkpoint directory
        inverse_dir: Inverse model checkpoint directory
    """
    import torch
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Forward model
    forward_best = Path(forward_dir) / "best_model.pt"
    if forward_best.exists():
        fwd_ckpt = torch.load(forward_best, map_location='cpu')
        print("\nForward Model:")
        print(f"  Epoch:    {fwd_ckpt.get('epoch', 0) + 1}")
        print(f"  Val Loss: {fwd_ckpt.get('val_loss', 0):.6f}")
        
        # Count parameters
        n_params = sum(p.numel() for p in fwd_ckpt['model_state_dict'].values())
        print(f"  Parameters: {n_params:,}")
    else:
        print("\nForward Model: Not found")
    
    # Inverse model
    inverse_best = Path(inverse_dir) / "best_model.pt"
    if inverse_best.exists():
        inv_ckpt = torch.load(inverse_best, map_location='cpu')
        print("\nInverse Model:")
        print(f"  Epoch:    {inv_ckpt.get('epoch', 0) + 1}")
        
        # Show all metrics if available
        if 'val_metrics' in inv_ckpt:
            metrics = inv_ckpt['val_metrics']
            print(f"  Total Loss:    {metrics.get('total', 0):.6f}")
            print(f"  Recon Loss:    {metrics.get('recon', 0):.6f}")
            print(f"  KL Loss:       {metrics.get('kl', 0):.6f}")
            print(f"  Forward Loss:  {metrics.get('forward', 0):.6f}")
            print(f"  Accuracy:      {metrics.get('accuracy', 0):.4f}")
        else:
            print(f"  Val Loss: {inv_ckpt.get('val_loss', 0):.6f}")
        
        # Show config
        if 'config' in inv_ckpt:
            print(f"\n  Configuration:")
            for key, val in inv_ckpt['config'].items():
                print(f"    {key}: {val}")
        
        # Count parameters
        n_params = sum(p.numel() for p in inv_ckpt['model_state_dict'].values())
        print(f"  Parameters: {n_params:,}")
    else:
        print("\nInverse Model: Not found")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('command', choices=['monitor', 'summary', 'compare'],
                        help='Command to run')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/inverse',
                        help='Checkpoint directory to monitor')
    parser.add_argument('--forward_dir', type=str,
                        default='checkpoints/forward',
                        help='Forward model checkpoint directory')
    parser.add_argument('--inverse_dir', type=str,
                        default='checkpoints/inverse',
                        help='Inverse model checkpoint directory')
    parser.add_argument('--interval', type=int, default=30,
                        help='Monitor update interval in seconds')
    
    args = parser.parse_args()
    
    if args.command == 'monitor':
        monitor_training(args.checkpoint_dir, args.interval)
    elif args.command == 'summary':
        summarize_training(args.checkpoint_dir)
    elif args.command == 'compare':
        compare_models(args.forward_dir, args.inverse_dir)


if __name__ == '__main__':
    main()
