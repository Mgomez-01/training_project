#!/bin/bash
# Quick start script for training the models

set -e  # Exit on error

echo "=========================================="
echo "Pattern Generation Training Pipeline"
echo "=========================================="
echo ""

# Configuration
ARRAYS_DIR="python/deep_archive/arrays/"
DATA_DIR="python/deep_archive/data/"
FORWARD_CHECKPOINT="checkpoints/forward/best_model.pt"
INVERSE_CHECKPOINT="checkpoints/inverse/best_model.pt"
DEVICE="cuda"

# Parse command line arguments
STAGE=${1:-"all"}

case $STAGE in
  "forward"|"1")
    echo "Stage 1: Training Forward Model"
    echo "=========================================="
    python train_forward.py \
      --arrays_dir $ARRAYS_DIR \
      --data_dir $DATA_DIR \
      --epochs 100 \
      --batch_size 32 \
      --lr 0.001 \
      --device $DEVICE
    echo ""
    echo "✓ Forward model training complete!"
    ;;
    
  "inverse"|"2")
    echo "Stage 2: Training Inverse cVAE"
    echo "=========================================="
    
    if [ ! -f "$FORWARD_CHECKPOINT" ]; then
      echo "Error: Forward model checkpoint not found at $FORWARD_CHECKPOINT"
      echo "Please train the forward model first: ./run_training.sh forward"
      exit 1
    fi
    
    python train_inverse.py \
      --arrays_dir $ARRAYS_DIR \
      --data_dir $DATA_DIR \
      --forward_checkpoint $FORWARD_CHECKPOINT \
      --epochs 200 \
      --batch_size 32 \
      --lr 0.0001 \
      --latent_dim 128 \
      --kl_weight 0.001 \
      --forward_weight 10.0 \
      --device $DEVICE
    echo ""
    echo "✓ Inverse model training complete!"
    ;;
    
  "generate"|"3")
    echo "Stage 3: Generate Patterns"
    echo "=========================================="
    
    if [ ! -f "$INVERSE_CHECKPOINT" ]; then
      echo "Error: Inverse model checkpoint not found at $INVERSE_CHECKPOINT"
      echo "Please train the inverse model first: ./run_training.sh inverse"
      exit 1
    fi
    
    TARGET_FILE=${2:-"path/to/target.pkl"}
    
    if [ ! -f "$TARGET_FILE" ]; then
      echo "Error: Target file not found: $TARGET_FILE"
      echo "Usage: ./run_training.sh generate <target_file>"
      exit 1
    fi
    
    python generate.py \
      --target_file $TARGET_FILE \
      --inverse_checkpoint $INVERSE_CHECKPOINT \
      --forward_checkpoint $FORWARD_CHECKPOINT \
      --normalization_file checkpoints/forward/normalization.pt \
      --n_candidates 50 \
      --top_k 10 \
      --device $DEVICE
    echo ""
    echo "✓ Pattern generation complete!"
    ;;
    
  "all")
    echo "Running complete pipeline..."
    echo ""
    
    # Stage 1: Forward
    echo "Stage 1/2: Training Forward Model"
    echo "=========================================="
    python train_forward.py \
      --arrays_dir $ARRAYS_DIR \
      --data_dir $DATA_DIR \
      --epochs 100 \
      --batch_size 32 \
      --device $DEVICE
    echo ""
    echo "✓ Forward model training complete!"
    echo ""
    
    # Stage 2: Inverse
    echo "Stage 2/2: Training Inverse cVAE"
    echo "=========================================="
    python train_inverse.py \
      --arrays_dir $ARRAYS_DIR \
      --data_dir $DATA_DIR \
      --forward_checkpoint $FORWARD_CHECKPOINT \
      --epochs 200 \
      --batch_size 32 \
      --device $DEVICE
    echo ""
    echo "✓ Inverse model training complete!"
    echo ""
    
    echo "=========================================="
    echo "Training pipeline complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Generate patterns:"
    echo "     ./run_training.sh generate <target_file>"
    echo "  2. Run simulations on generated patterns"
    echo "  3. Compare with predictions"
    echo "=========================================="
    ;;
    
  *)
    echo "Usage: ./run_training.sh [stage]"
    echo ""
    echo "Stages:"
    echo "  forward  (or 1) - Train forward model only"
    echo "  inverse  (or 2) - Train inverse model only"
    echo "  generate (or 3) - Generate patterns (requires target file)"
    echo "  all             - Train both models sequentially (default)"
    echo ""
    echo "Examples:"
    echo "  ./run_training.sh                    # Train both models"
    echo "  ./run_training.sh forward            # Train forward model"
    echo "  ./run_training.sh inverse            # Train inverse model"
    echo "  ./run_training.sh generate target.pkl  # Generate patterns"
    exit 1
    ;;
esac
