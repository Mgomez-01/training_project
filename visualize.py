"""
Visualization utilities for patterns and S-parameters

Note: Requires matplotlib. Install with: pip install matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_pattern(pattern, title="Pattern", save_path=None):
    """
    Plot a binary pattern
    
    Args:
        pattern: numpy array [H, W] or [1, H, W]
        title: Plot title
        save_path: Optional path to save figure
    """
    if len(pattern.shape) == 3:
        pattern = pattern[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pattern, cmap='binary', interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X dimension')
    ax.set_ylabel('Y dimension')
    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_s_parameters(s_params_df, title="S-Parameters", save_path=None):
    """
    Plot S-parameters vs frequency
    
    Args:
        s_params_df: DataFrame with columns [Frequency, S11 dB, S21 dB, S22 dB, S12 dB]
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    freq = s_params_df['Frequency'].values
    ax.plot(freq, s_params_df['S11 dB'], label='S11', linewidth=2)
    ax.plot(freq, s_params_df['S21 dB'], label='S21', linewidth=2)
    ax.plot(freq, s_params_df['S22 dB'], label='S22', linewidth=2)
    ax.plot(freq, s_params_df['S12 dB'], label='S12', linewidth=2)
    
    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_s_parameters(target_df, predicted_df, title="S-Parameter Comparison", 
                        save_path=None):
    """
    Compare target vs predicted S-parameters
    
    Args:
        target_df: Target S-parameters DataFrame
        predicted_df: Predicted S-parameters DataFrame
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    params = ['S11 dB', 'S21 dB', 'S22 dB', 'S12 dB']
    freq = target_df['Frequency'].values
    
    for i, param in enumerate(params):
        ax = axes[i]
        ax.plot(freq, target_df[param], 'b-', label='Target', linewidth=2)
        ax.plot(freq, predicted_df[param], 'r--', label='Predicted', linewidth=2)
        
        # Compute error
        error = np.mean((target_df[param].values - predicted_df[param].values) ** 2)
        
        ax.set_xlabel('Frequency (GHz)', fontsize=10)
        ax.set_ylabel('Magnitude (dB)', fontsize=10)
        ax.set_title(f'{param} (MSE: {error:.4f})', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_candidate_gallery(candidates, n_show=9, save_path=None):
    """
    Plot a gallery of candidate patterns
    
    Args:
        candidates: List of candidate dicts from generator
        n_show: Number of candidates to show
        save_path: Optional path to save figure
    """
    n_show = min(n_show, len(candidates))
    n_cols = 3
    n_rows = (n_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_show):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        pattern = candidates[idx]['pattern']
        error = candidates[idx]['error']
        
        ax.imshow(pattern, cmap='binary', interpolation='nearest')
        ax.set_title(f"Rank {idx+1}\nError: {error:.4f}", fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_show, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Candidate Patterns (Sorted by Error)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_generation_results(output_dir, n_show=5):
    """
    Visualize results from generate.py output directory
    
    Args:
        output_dir: Directory containing generated patterns
        n_show: Number of top candidates to visualize
    """
    output_dir = Path(output_dir)
    
    # Find all generated patterns
    pattern_files = sorted(list(output_dir.glob("*.array")))[:n_show]
    
    if len(pattern_files) == 0:
        print(f"No pattern files found in {output_dir}")
        return
    
    print(f"Visualizing top {len(pattern_files)} candidates from {output_dir}/")
    
    for i, pattern_file in enumerate(pattern_files):
        # Load pattern (text format)
        try:
            pattern = np.loadtxt(pattern_file, dtype=np.float32)
            if pattern.ndim == 1:
                pattern = pattern.reshape(48, 32)
        except:
            # Fallback to binary
            pattern = np.fromfile(pattern_file, dtype=np.float32).reshape(48, 32)
        
        # Load S-parameters
        s_file = pattern_file.with_suffix('.pkl')
        if s_file.exists():
            s_params = pd.read_pickle(s_file)
        else:
            s_params = None
        
        # Create figure with both pattern and S-parameters
        fig = plt.figure(figsize=(14, 5))
        
        # Plot pattern
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(pattern, cmap='binary', interpolation='nearest')
        ax1.set_title(f'Pattern: {pattern_file.name}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Plot S-parameters
        if s_params is not None:
            ax2 = plt.subplot(1, 2, 2)
            freq = s_params['Frequency'].values
            ax2.plot(freq, s_params['S11 dB'], label='S11', linewidth=2)
            ax2.plot(freq, s_params['S21 dB'], label='S21', linewidth=2)
            ax2.plot(freq, s_params['S22 dB'], label='S22', linewidth=2)
            ax2.plot(freq, s_params['S12 dB'], label='S12', linewidth=2)
            ax2.set_xlabel('Frequency (GHz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.set_title('Predicted S-Parameters', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / f"visualization_{i:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    print(f"\n✓ Visualizations saved to {output_dir}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize patterns and S-parameters')
    parser.add_argument('--output_dir', type=str, default='generated_patterns',
                        help='Directory containing generated patterns')
    parser.add_argument('--n_show', type=int, default=5,
                        help='Number of candidates to visualize')
    
    args = parser.parse_args()
    
    try:
        visualize_generation_results(args.output_dir, args.n_show)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure matplotlib is installed: pip install matplotlib")
