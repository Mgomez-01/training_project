"""
Data inspection and analysis utilities

Use this script to understand your dataset before training
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict


def inspect_arrays(arrays_dir, n_samples=10):
    """Inspect sparse array files"""
    arrays_dir = Path(arrays_dir)
    array_files = list(arrays_dir.glob("*_sparse_array"))
    
    print("="*60)
    print("SPARSE ARRAY ANALYSIS")
    print("="*60)
    print(f"Directory: {arrays_dir}")
    print(f"Total files: {len(array_files)}")
    
    if len(array_files) == 0:
        print("No array files found!")
        return
    
    # Analyze sample files
    print(f"\nAnalyzing {min(n_samples, len(array_files))} samples...")
    
    shapes = []
    sparsities = []
    file_sizes = []
    
    for i, file_path in enumerate(array_files[:n_samples]):
        data = np.fromfile(file_path, dtype=np.float32)
        file_sizes.append(file_path.stat().st_size)
        
        # Try to infer shape
        total_elements = len(data)
        if total_elements == 1536:  # 48*32
            shape = (48, 32)
        else:
            # Try common shapes
            for h in [32, 48, 64, 128]:
                if total_elements % h == 0:
                    w = total_elements // h
                    shape = (h, w)
                    break
            else:
                shape = None
        
        shapes.append(shape)
        
        # Calculate sparsity
        sparsity = np.mean(data)
        sparsities.append(sparsity)
        
        if i < 3:  # Show details for first 3
            print(f"\nFile {i+1}: {file_path.name}")
            print(f"  Elements: {total_elements}")
            print(f"  Shape: {shape}")
            print(f"  Sparsity: {sparsity:.4f} (fraction of 1s)")
            print(f"  Min value: {data.min()}")
            print(f"  Max value: {data.max()}")
            print(f"  Unique values: {np.unique(data)}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"File size: {np.mean(file_sizes):.0f} ± {np.std(file_sizes):.0f} bytes")
    print(f"Most common shape: {max(set(shapes), key=shapes.count)}")
    print(f"Average sparsity: {np.mean(sparsities):.4f} ± {np.std(sparsities):.4f}")
    print(f"Sparsity range: [{np.min(sparsities):.4f}, {np.max(sparsities):.4f}]")


def inspect_data_files(data_dir, n_samples=10):
    """Inspect S-parameter data files"""
    data_dir = Path(data_dir)
    pkl_files = list(data_dir.glob("*.pkl"))
    
    print("\n" + "="*60)
    print("S-PARAMETER DATA ANALYSIS")
    print("="*60)
    print(f"Directory: {data_dir}")
    print(f"Total files: {len(pkl_files)}")
    
    if len(pkl_files) == 0:
        print("No .pkl files found!")
        return
    
    # Analyze sample files
    print(f"\nAnalyzing {min(n_samples, len(pkl_files))} samples...")
    
    all_columns = []
    shapes = []
    freq_ranges = []
    s_param_stats = defaultdict(list)
    
    for i, file_path in enumerate(pkl_files[:n_samples]):
        df = pd.read_pickle(file_path)
        
        all_columns.append(list(df.columns))
        shapes.append(df.shape)
        
        # Check frequency column
        if 'Frequency' in df.columns:
            freq_ranges.append((df['Frequency'].min(), df['Frequency'].max()))
        
        # S-parameter statistics
        for col in df.columns:
            if 'dB' in col or col.startswith('S'):
                s_param_stats[col].append({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                })
        
        if i < 3:  # Show details for first 3
            print(f"\nFile {i+1}: {file_path.name}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            if 'Frequency' in df.columns:
                print(f"  Frequency range: {df['Frequency'].min():.2f} - {df['Frequency'].max():.2f}")
            print(f"  Sample data (first 3 rows):")
            print(df.head(3).to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    most_common_shape = max(set(shapes), key=shapes.count)
    print(f"Most common shape: {most_common_shape}")
    print(f"Most common columns: {max(set(map(tuple, all_columns)), key=all_columns.count)}")
    
    if freq_ranges:
        all_mins = [r[0] for r in freq_ranges]
        all_maxs = [r[1] for r in freq_ranges]
        print(f"Frequency range: {np.mean(all_mins):.2f} - {np.mean(all_maxs):.2f} GHz")
    
    print(f"\nS-Parameter Statistics (averaged over {len(pkl_files[:n_samples])} files):")
    for param, stats_list in s_param_stats.items():
        mean_vals = [s['mean'] for s in stats_list]
        min_vals = [s['min'] for s in stats_list]
        max_vals = [s['max'] for s in stats_list]
        print(f"  {param}:")
        print(f"    Mean: {np.mean(mean_vals):8.2f} dB (std: {np.std(mean_vals):.2f})")
        print(f"    Range: [{np.mean(min_vals):6.2f}, {np.mean(max_vals):6.2f}] dB")


def check_pairing(arrays_dir, data_dir):
    """Check if array and data files are properly paired"""
    arrays_dir = Path(arrays_dir)
    data_dir = Path(data_dir)
    
    print("\n" + "="*60)
    print("FILE PAIRING ANALYSIS")
    print("="*60)
    
    array_files = {f.stem: f for f in arrays_dir.glob("*_sparse_array")}
    pkl_files = {f.stem: f for f in data_dir.glob("*.pkl")}
    
    print(f"Array files: {len(array_files)}")
    print(f"Data files:  {len(pkl_files)}")
    
    # Find paired files
    paired = []
    unpaired_arrays = []
    unpaired_data = []
    
    for array_name, array_path in array_files.items():
        # Expected pkl name
        expected_pkl_name = array_name.replace('_sparse_array', '_dataframe')
        
        if expected_pkl_name in pkl_files:
            paired.append((array_name, expected_pkl_name))
        else:
            unpaired_arrays.append(array_name)
    
    # Check for unpaired pkl files
    for pkl_name in pkl_files:
        expected_array_name = pkl_name.replace('_dataframe', '_sparse_array')
        if expected_array_name not in array_files:
            unpaired_data.append(pkl_name)
    
    print(f"\n✓ Paired files: {len(paired)}")
    
    if unpaired_arrays:
        print(f"\n⚠ Unpaired array files: {len(unpaired_arrays)}")
        for name in unpaired_arrays[:5]:
            print(f"  - {name}")
        if len(unpaired_arrays) > 5:
            print(f"  ... and {len(unpaired_arrays)-5} more")
    
    if unpaired_data:
        print(f"\n⚠ Unpaired data files: {len(unpaired_data)}")
        for name in unpaired_data[:5]:
            print(f"  - {name}")
        if len(unpaired_data) > 5:
            print(f"  ... and {len(unpaired_data)-5} more")
    
    if not unpaired_arrays and not unpaired_data:
        print("\n✓ All files are properly paired!")
    
    return len(paired)


def main():
    parser = argparse.ArgumentParser(description='Inspect dataset before training')
    parser.add_argument('--arrays_dir', type=str, required=True,
                        help='Path to sparse array directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to S-parameter data directory')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to analyze in detail')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DATASET INSPECTION TOOL")
    print("="*60)
    
    # Inspect arrays
    inspect_arrays(args.arrays_dir, args.n_samples)
    
    # Inspect data files
    inspect_data_files(args.data_dir, args.n_samples)
    
    # Check pairing
    n_paired = check_pairing(args.arrays_dir, args.data_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("DATASET READY CHECK")
    print("="*60)
    
    if n_paired > 0:
        print(f"✓ Found {n_paired} paired samples")
        print(f"✓ Dataset is ready for training!")
        print(f"\nNext steps:")
        print(f"  1. Train forward model:")
        print(f"     python train_forward.py --arrays_dir {args.arrays_dir} --data_dir {args.data_dir}")
        print(f"  2. Train inverse model:")
        print(f"     python train_inverse.py --arrays_dir {args.arrays_dir} --data_dir {args.data_dir}")
    else:
        print(f"✗ No paired samples found!")
        print(f"✗ Cannot train models without paired data")
        print(f"\nPlease check:")
        print(f"  - File naming convention")
        print(f"  - Directory paths")
        print(f"  - File formats")
    
    print("="*60)


if __name__ == '__main__':
    main()
