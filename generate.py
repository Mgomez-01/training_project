"""
Pattern generation script using trained inverse cVAE model

This script generates multiple candidate patterns for a given target S-parameter specification.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from models.inverse import InversecVAE
from models.forward import ForwardModelResNet


class PatternGenerator:
    """Class for generating patterns from trained models"""
    
    def __init__(self, 
                 inverse_checkpoint,
                 forward_checkpoint,
                 normalization_file,
                 device='cuda'):
        """
        Initialize the pattern generator
        
        Args:
            inverse_checkpoint: Path to trained inverse model
            forward_checkpoint: Path to trained forward model
            normalization_file: Path to normalization statistics
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load normalization
        print(f"Loading normalization from: {normalization_file}")
        norm = torch.load(normalization_file, map_location=self.device)
        self.S_mean = norm['S_mean']
        self.S_std = norm['S_std']
        print(f"  Mean: {self.S_mean[0]}")
        print(f"  Std:  {self.S_std[0]}")
        
        # Load inverse model
        print(f"\nLoading inverse model from: {inverse_checkpoint}")
        inv_ckpt = torch.load(inverse_checkpoint, map_location=self.device)
        latent_dim = inv_ckpt['config']['latent_dim']
        self.inverse_model = InversecVAE(latent_dim=latent_dim).to(self.device)
        self.inverse_model.load_state_dict(inv_ckpt['model_state_dict'])
        self.inverse_model.eval()
        print(f"  Latent dim: {latent_dim}")
        print(f"  Trained for {inv_ckpt['epoch']+1} epochs")
        print(f"  Val loss: {inv_ckpt['val_loss']:.4f}")
        
        # Load forward model
        print(f"\nLoading forward model from: {forward_checkpoint}")
        self.forward_model = ForwardModelResNet().to(self.device)
        fwd_ckpt = torch.load(forward_checkpoint, map_location=self.device)
        self.forward_model.load_state_dict(fwd_ckpt['model_state_dict'])
        self.forward_model.eval()
        print(f"  Trained for {fwd_ckpt['epoch']+1} epochs")
        print(f"  Val loss: {fwd_ckpt['val_loss']:.6f}")
        
        print("\n✓ Models loaded successfully!")
    
    def normalize_S(self, S_params):
        """Normalize S-parameters using stored statistics"""
        return (S_params - self.S_mean) / self.S_std
    
    def denormalize_S(self, S_params_norm):
        """Denormalize S-parameters"""
        return S_params_norm * self.S_std + self.S_mean
    
    @torch.no_grad()
    def generate_candidates(self, target_S_params, n_candidates=20, latent_std=1.0, 
                           return_probs=False):
        """
        Generate multiple candidate patterns for a target S-parameter
        
        Args:
            target_S_params: pandas DataFrame or numpy array [201, 4]
            n_candidates: number of patterns to generate
            latent_std: std for sampling latent space (1.0 = standard normal)
            return_probs: whether to return pattern probabilities
        
        Returns:
            list of dicts with 'pattern', 'predicted_S', 'error', 'candidate_id'
        """
        print(f"\nGenerating {n_candidates} candidates...")
        
        # Prepare target
        if isinstance(target_S_params, pd.DataFrame):
            target_S = target_S_params[['S11 dB', 'S21 dB', 'S22 dB', 'S12 dB']].values
        else:
            target_S = target_S_params
        
        target_S = target_S.astype(np.float32)
        target_S_norm = self.normalize_S(target_S)
        target_S_tensor = torch.from_numpy(target_S_norm).unsqueeze(0).to(self.device)
        
        candidates = []
        
        for i in range(n_candidates):
            # Sample from latent space
            z = torch.randn(1, self.inverse_model.latent_dim, device=self.device) * latent_std
            
            # Generate pattern
            logits = self.inverse_model.decode(z, target_S_tensor)
            pattern_probs = torch.sigmoid(logits)
            pattern_binary = (pattern_probs > 0.5).float()
            
            # Validate with forward model
            S_pred_norm = self.forward_model(pattern_binary)
            S_pred = self.denormalize_S(S_pred_norm.cpu().numpy()[0])
            
            # Compute error (MSE across all S-parameters and frequencies)
            error = np.mean((S_pred - target_S) ** 2)
            
            result = {
                'pattern': pattern_binary.cpu().numpy()[0, 0],  # [48, 32]
                'predicted_S': S_pred,
                'error': error,
                'candidate_id': i
            }
            
            if return_probs:
                result['pattern_probs'] = pattern_probs.cpu().numpy()[0, 0]
            
            candidates.append(result)
        
        # Sort by error
        candidates.sort(key=lambda x: x['error'])
        
        print(f"✓ Generated {n_candidates} candidates")
        print(f"  Best error: {candidates[0]['error']:.6f}")
        print(f"  Worst error: {candidates[-1]['error']:.6f}")
        print(f"  Mean error: {np.mean([c['error'] for c in candidates]):.6f}")
        
        return candidates
    
    def save_patterns(self, candidates, output_dir, prefix='candidate', top_k=None):
        """
        Save candidate patterns to files
        
        Args:
            candidates: List of candidate dicts
            output_dir: Directory to save patterns
            prefix: Prefix for filenames
            top_k: Only save top k candidates (None = save all)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if top_k is not None:
            candidates = candidates[:top_k]
        
        print(f"\nSaving {len(candidates)} patterns to {output_dir}/")
        
        for i, cand in enumerate(candidates):
            pattern = cand['pattern'].astype(np.int32)  # Convert to int for cleaner text output
            
            # Save pattern array as space-separated text file (matching input format)
            filename = output_dir / f"{prefix}_{i:03d}_error_{cand['error']:.4f}.array"
            np.savetxt(filename, pattern, fmt='%d')  # Save as integers with spaces
            
            # Save predicted S-parameters
            S_df = pd.DataFrame(
                cand['predicted_S'], 
                columns=['S11 dB', 'S21 dB', 'S22 dB', 'S12 dB']
            )
            S_df.insert(0, 'Frequency', np.linspace(190, 200, 201))
            S_df.to_pickle(filename.with_suffix('.pkl'))
        
        print(f"✓ Saved {len(candidates)} patterns")


def main():
    parser = argparse.ArgumentParser(description='Generate patterns from trained models')
    parser.add_argument('--target_file', type=str, required=True,
                        help='Path to target S-parameter .pkl file')
    parser.add_argument('--inverse_checkpoint', type=str, 
                        default='checkpoints/inverse/best_model.pt',
                        help='Path to trained inverse model')
    parser.add_argument('--forward_checkpoint', type=str, 
                        default='checkpoints/forward/best_model.pt',
                        help='Path to trained forward model')
    parser.add_argument('--normalization_file', type=str,
                        default='checkpoints/forward/normalization.pt',
                        help='Path to normalization statistics')
    parser.add_argument('--output_dir', type=str, default='generated_patterns',
                        help='Directory to save generated patterns')
    parser.add_argument('--n_candidates', type=int, default=50,
                        help='Number of candidates to generate')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top candidates to save')
    parser.add_argument('--latent_std', type=float, default=1.0,
                        help='Standard deviation for latent sampling')
    parser.add_argument('--prefix', type=str, default='candidate',
                        help='Prefix for output filenames')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for generation')
    
    args = parser.parse_args()
    
    # Initialize generator
    print("="*80)
    print("Pattern Generator")
    print("="*80)
    
    generator = PatternGenerator(
        inverse_checkpoint=args.inverse_checkpoint,
        forward_checkpoint=args.forward_checkpoint,
        normalization_file=args.normalization_file,
        device=args.device
    )
    
    # Load target S-parameters
    print("\n" + "="*80)
    print(f"Loading target from: {args.target_file}")
    target_S = pd.read_pickle(args.target_file)
    print(f"Target shape: {target_S.shape}")
    print("\nTarget S-parameters (first 5 rows):")
    print(target_S.head())
    
    # Generate candidates
    print("\n" + "="*80)
    candidates = generator.generate_candidates(
        target_S, 
        n_candidates=args.n_candidates,
        latent_std=args.latent_std
    )
    
    # Print top 10
    print("\n" + "="*80)
    print(f"Top {min(10, len(candidates))} candidates by error:")
    print("-"*80)
    for i, cand in enumerate(candidates[:10]):
        print(f"  {i+1:2d}. Error: {cand['error']:.6f}")
    
    # Save patterns
    print("\n" + "="*80)
    generator.save_patterns(
        candidates, 
        args.output_dir, 
        prefix=args.prefix,
        top_k=args.top_k
    )
    
    print("\n" + "="*80)
    print("Generation complete!")
    print(f"Next steps:")
    print(f"  1. Run simulations on patterns in: {args.output_dir}/")
    print(f"  2. Compare simulation results with predicted S-parameters")
    print(f"  3. Select best performing patterns")
    print("="*80)


if __name__ == '__main__':
    main()
