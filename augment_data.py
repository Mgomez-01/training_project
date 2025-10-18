"""
Data augmentation using 2-port network symmetry

For a 2-port network, flipping the pattern vertically swaps the ports:
- Port 1 becomes Port 2
- Port 2 becomes Port 1

This means:
- S11 (reflection at port 1) ↔ S22 (reflection at port 2)
- S21 (transmission 1→2) ↔ S12 (transmission 2→1)

This doubles the effective dataset size.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def flip_pattern_vertical(pattern):
    """
    Flip pattern vertically (reverse row order)
    
    Args:
        pattern: numpy array [H, W]
    
    Returns:
        Flipped pattern [H, W]
    """
    return np.flipud(pattern)


def swap_s_parameters(s_params_df):
    """
    Swap S-parameters for port reversal
    
    S11 ↔ S22 (reflections)
    S21 ↔ S12 (transmissions)
    
    Args:
        s_params_df: DataFrame with columns [Frequency, S11 dB, S21 dB, S22 dB, S12 dB]
    
    Returns:
        DataFrame with swapped S-parameters
    """
    swapped_df = s_params_df.copy()
    
    # Swap S11 ↔ S22
    swapped_df['S11 dB'] = s_params_df['S22 dB'].values
    swapped_df['S22 dB'] = s_params_df['S11 dB'].values
    
    # Swap S21 ↔ S12
    swapped_df['S21 dB'] = s_params_df['S12 dB'].values
    swapped_df['S12 dB'] = s_params_df['S21 dB'].values
    
    return swapped_df


def augment_dataset(arrays_dir, data_dir, output_arrays_dir, output_data_dir):
    """
    Create augmented dataset by flipping patterns and swapping S-parameters
    
    Args:
        arrays_dir: Source directory for patterns
        data_dir: Source directory for S-parameters
        output_arrays_dir: Output directory for augmented patterns
        output_data_dir: Output directory for augmented S-parameters
    """
    arrays_dir = Path(arrays_dir)
    data_dir = Path(data_dir)
    output_arrays_dir = Path(output_arrays_dir)
    output_data_dir = Path(output_data_dir)
    
    # Create output directories
    output_arrays_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all paired files
    array_files = list(arrays_dir.glob("*_sparse_array"))
    
    print("="*80)
    print("2-PORT NETWORK DATA AUGMENTATION")
    print("="*80)
    print(f"Source arrays: {arrays_dir}")
    print(f"Source data:   {data_dir}")
    print(f"Output arrays: {output_arrays_dir}")
    print(f"Output data:   {output_data_dir}")
    print(f"\nFound {len(array_files)} original samples")
    print(f"Will create {len(array_files)} augmented samples")
    print(f"Total dataset: {2 * len(array_files)} samples")
    print("="*80)
    print()
    
    augmented_count = 0
    skipped_count = 0
    
    for array_file in tqdm(array_files, desc="Augmenting"):
        # Get base name and construct corresponding data file
        base_name = array_file.stem  # e.g., "20251006_063646_lab2_6_results_sparse_array"
        
        # Replace _sparse_array with _dataframe to get pkl filename
        if base_name.endswith('_sparse_array'):
            pkl_base = base_name[:-len('_sparse_array')] + '_dataframe'
        else:
            pkl_base = base_name + '_dataframe'
        
        pkl_file = data_dir / f"{pkl_base}.pkl"
        
        if not pkl_file.exists():
            print(f"Warning: Missing data file for {base_name}")
            skipped_count += 1
            continue
        
        try:
            # Load original pattern
            pattern = np.loadtxt(array_file, dtype=np.float32)
            if pattern.ndim == 1:
                pattern = pattern.reshape(48, 32)
            
            # Load original S-parameters
            s_params = pd.read_pickle(pkl_file)
            
            # Flip pattern vertically
            flipped_pattern = flip_pattern_vertical(pattern)
            
            # Swap S-parameters
            swapped_s_params = swap_s_parameters(s_params)
            
            # Create augmented filenames
            # Strip _sparse_array from base_name for the core filename
            core_name = base_name[:-len('_sparse_array')] if base_name.endswith('_sparse_array') else base_name
            
            aug_array_file = output_arrays_dir / f"{core_name}_flipped_sparse_array"
            aug_pkl_file = output_data_dir / f"{core_name}_flipped_dataframe.pkl"
            
            # Save augmented pattern (text format)
            np.savetxt(aug_array_file, flipped_pattern.astype(np.int32), fmt='%d')
            
            # Save augmented S-parameters
            swapped_s_params.to_pickle(aug_pkl_file)
            
            augmented_count += 1
            
        except Exception as e:
            print(f"\nError processing {base_name}: {e}")
            skipped_count += 1
    
    print()
    print("="*80)
    print("AUGMENTATION COMPLETE")
    print("="*80)
    print(f"Successfully augmented: {augmented_count} samples")
    print(f"Skipped:                {skipped_count} samples")
    print(f"\nOriginal dataset:   {len(array_files)} samples")
    print(f"Augmented dataset:  {augmented_count} samples")
    print(f"Total after merge:  {len(array_files) + augmented_count} samples")
    print()
    print("Augmented files created in:")
    print(f"  Arrays: {output_arrays_dir}")
    print(f"  Data:   {output_data_dir}")
    print()
    print("To use augmented data:")
    print("  Option 1: Copy augmented files into original directories")
    print(f"    cp {output_arrays_dir}/* {arrays_dir}/")
    print(f"    cp {output_data_dir}/* {data_dir}/")
    print("  Option 2: Use Makefile: 'make augment' (copies automatically)")
    print("="*80)


def verify_augmentation(array_file, pkl_file, aug_array_file, aug_pkl_file):
    """
    Verify that augmentation is correct
    
    Args:
        array_file: Original pattern file
        pkl_file: Original S-parameter file
        aug_array_file: Augmented pattern file
        aug_pkl_file: Augmented S-parameter file
    """
    print("="*80)
    print("AUGMENTATION VERIFICATION")
    print("="*80)
    
    # Load original
    pattern = np.loadtxt(array_file, dtype=np.float32)
    if pattern.ndim == 1:
        pattern = pattern.reshape(48, 32)
    s_params = pd.read_pickle(pkl_file)
    
    # Load augmented
    aug_pattern = np.loadtxt(aug_array_file, dtype=np.float32)
    if aug_pattern.ndim == 1:
        aug_pattern = aug_pattern.reshape(48, 32)
    aug_s_params = pd.read_pickle(aug_pkl_file)
    
    print(f"\nOriginal pattern file: {array_file}")
    print(f"Augmented pattern file: {aug_array_file}")
    
    # Check pattern flip
    expected_flip = np.flipud(pattern)
    pattern_match = np.array_equal(aug_pattern, expected_flip)
    print(f"\nPattern flip correct: {pattern_match}")
    
    if not pattern_match:
        print("  WARNING: Pattern flip does not match!")
    
    # Check S-parameter swap
    s11_match = np.allclose(aug_s_params['S11 dB'], s_params['S22 dB'])
    s22_match = np.allclose(aug_s_params['S22 dB'], s_params['S11 dB'])
    s21_match = np.allclose(aug_s_params['S21 dB'], s_params['S12 dB'])
    s12_match = np.allclose(aug_s_params['S12 dB'], s_params['S21 dB'])
    
    print(f"\nS-parameter swaps correct:")
    print(f"  S11 ↔ S22: {s11_match and s22_match}")
    print(f"  S21 ↔ S12: {s21_match and s12_match}")
    
    if not all([s11_match, s22_match, s21_match, s12_match]):
        print("  WARNING: S-parameter swaps do not match!")
    
    # Show sample values
    print(f"\nOriginal S-parameters (first 3 rows):")
    print(s_params[['S11 dB', 'S21 dB', 'S22 dB', 'S12 dB']].head(3))
    
    print(f"\nAugmented S-parameters (first 3 rows):")
    print(aug_s_params[['S11 dB', 'S21 dB', 'S22 dB', 'S12 dB']].head(3))
    
    print("\n" + "="*80)
    
    if pattern_match and all([s11_match, s22_match, s21_match, s12_match]):
        print("✓ VERIFICATION PASSED")
    else:
        print("✗ VERIFICATION FAILED")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Augment dataset using 2-port network symmetry'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Augment command
    aug_parser = subparsers.add_parser('augment', help='Create augmented dataset')
    aug_parser.add_argument('--arrays_dir', type=str, required=True,
                           help='Source directory for patterns')
    aug_parser.add_argument('--data_dir', type=str, required=True,
                           help='Source directory for S-parameters')
    aug_parser.add_argument('--output_arrays_dir', type=str, required=True,
                           help='Output directory for augmented patterns')
    aug_parser.add_argument('--output_data_dir', type=str, required=True,
                           help='Output directory for augmented S-parameters')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify augmentation')
    verify_parser.add_argument('--array_file', type=str, required=True,
                              help='Original pattern file')
    verify_parser.add_argument('--pkl_file', type=str, required=True,
                              help='Original S-parameter file')
    verify_parser.add_argument('--aug_array_file', type=str, required=True,
                              help='Augmented pattern file')
    verify_parser.add_argument('--aug_pkl_file', type=str, required=True,
                              help='Augmented S-parameter file')
    
    args = parser.parse_args()
    
    if args.command == 'augment':
        augment_dataset(
            args.arrays_dir,
            args.data_dir,
            args.output_arrays_dir,
            args.output_data_dir
        )
    elif args.command == 'verify':
        verify_augmentation(
            args.array_file,
            args.pkl_file,
            args.aug_array_file,
            args.aug_pkl_file
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
