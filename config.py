"""
Configuration file for training hyperparameters

You can modify these values or load them programmatically
"""

# Data paths
ARRAYS_DIR = "python/deep_archive/arrays/"
DATA_DIR = "python/deep_archive/data/"

# Training configuration
FORWARD_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-3,
    'use_resnet': True,
    'save_dir': 'checkpoints/forward',
}

INVERSE_CONFIG = {
    'epochs': 200,
    'batch_size': 32,
    'lr': 1e-4,
    'latent_dim': 128,
    'kl_weight': 0.001,  # Start small, can increase if patterns look random
    'forward_weight': 10.0,  # Important for physical consistency
    'temperature_start': 5.0,
    'temperature_end': 0.5,
    'save_dir': 'checkpoints/inverse',
}

# Generation configuration
GENERATION_CONFIG = {
    'n_candidates': 50,  # Number of patterns to generate
    'top_k': 10,  # Number to save for simulation
    'latent_std': 1.0,  # Sampling variance
    'output_dir': 'generated_patterns',
}

# Hardware
DEVICE = 'cuda'  # or 'cpu'
NUM_WORKERS = 4  # For data loading

# Dataset
TRAIN_SPLIT = 0.8  # 80% train, 20% validation
