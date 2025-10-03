import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


class SimulationDataset(Dataset):
    """Dataset for paired pattern and S-parameter data"""
    
    def __init__(self, arrays_dir, data_dir, normalize_S=True):
        """
        Args:
            arrays_dir: Path to directory containing sparse array files
            data_dir: Path to directory containing .pkl S-parameter files
            normalize_S: Whether to normalize S-parameters
        """
        self.arrays_dir = Path(arrays_dir)
        self.data_dir = Path(data_dir)
        
        # Find all paired files
        self.samples = []
        for array_file in self.arrays_dir.glob("*_sparse_array"):
            # Extract base name
            base_name = array_file.stem  # removes extension
            pkl_file = self.data_dir / f"{base_name.replace('_sparse_array', '_dataframe')}.pkl"
            
            if pkl_file.exists():
                self.samples.append({
                    'array_path': array_file,
                    'data_path': pkl_file,
                    'name': base_name
                })
        
        print(f"Found {len(self.samples)} paired samples")
        
        if len(self.samples) == 0:
            raise ValueError(f"No paired samples found in {arrays_dir} and {data_dir}")
        
        # Compute normalization statistics if needed
        self.normalize_S = normalize_S
        if normalize_S:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean/std across all S-parameters"""
        print("Computing normalization statistics...")
        all_S = []
        
        # Sample subset for speed (or all if small dataset)
        sample_size = min(100, len(self.samples))
        for sample in self.samples[:sample_size]:
            df = pd.read_pickle(sample['data_path'])
            S = df[['S11 dB', 'S21 dB', 'S22 dB', 'S12 dB']].values
            all_S.append(S)
        
        all_S = np.concatenate(all_S, axis=0)
        self.S_mean = all_S.mean(axis=0, keepdims=True)  # [1, 4]
        self.S_std = all_S.std(axis=0, keepdims=True) + 1e-8
        
        print(f"S-param normalization:")
        print(f"  Mean: {self.S_mean[0]}")
        print(f"  Std:  {self.S_std[0]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load pattern (binary array)
        pattern = np.fromfile(sample['array_path'], dtype=np.float32)
        pattern = pattern.reshape(48, 32)  # Adjust if needed
        pattern = pattern.astype(np.float32)  # Ensure binary {0, 1}
        
        # Load S-parameters
        df = pd.read_pickle(sample['data_path'])
        S_params = df[['S11 dB', 'S21 dB', 'S22 dB', 'S12 dB']].values  # [201, 4]
        S_params = S_params.astype(np.float32)
        
        # Normalize S-parameters
        if self.normalize_S:
            S_params = (S_params - self.S_mean) / self.S_std
        
        # Convert to torch tensors
        pattern = torch.from_numpy(pattern).unsqueeze(0)  # [1, 48, 32]
        S_params = torch.from_numpy(S_params)  # [201, 4]
        
        return {
            'pattern': pattern,
            'S_params': S_params,
            'name': sample['name']
        }


def get_dataloaders(arrays_dir, data_dir, batch_size=32, train_split=0.8, num_workers=4):
    """
    Create train and validation dataloaders
    
    Args:
        arrays_dir: Path to sparse array directory
        data_dir: Path to S-parameter data directory
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, S_mean, S_std
    """
    dataset = SimulationDataset(arrays_dir, data_dir)
    
    # Train/val split
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset.S_mean, dataset.S_std
