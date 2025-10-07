# Makefile for Pattern Generation Project
# Provides convenient shortcuts for common tasks

# Configuration
ARRAYS_DIR := python/deep_archive/arrays/
DATA_DIR := python/deep_archive/data/
FORWARD_CKPT := checkpoints/forward/best_model.pt
INVERSE_CKPT := checkpoints/inverse/best_model.pt
DEVICE := cuda

.PHONY: help install test inspect augment train-forward train-inverse train-all generate evaluate visualize tune clean

help:
	@echo "Pattern Generation Project - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run unit tests"
	@echo "  make inspect      - Inspect dataset"
	@echo ""
	@echo "Data Augmentation:"
	@echo "  make augment          - Create augmented dataset (2x data using 2-port symmetry)"
	@echo "  make augment-verify   - Verify augmentation is correct"
	@echo ""
	@echo "Training:"
	@echo "  make train-forward    - Train forward model only"
	@echo "  make train-inverse    - Train inverse model only"
	@echo "  make train-all        - Train both models sequentially"
	@echo ""
	@echo "Generation & Evaluation:"
	@echo "  make generate TARGET=path/to/target.pkl  - Generate patterns"
	@echo "  make evaluate         - Evaluate trained models"
	@echo "  make visualize        - Visualize generated patterns"
	@echo ""
	@echo "Advanced:"
	@echo "  make tune            - Tune hyperparameters"
	@echo "  make clean           - Clean generated files"
	@echo ""
	@echo "Quick Workflows:"
	@echo "  make workflow            - inspect + train + evaluate"
	@echo "  make workflow-augmented  - inspect + augment + train + evaluate (recommended!)"
	@echo ""
	@echo "Configuration (can override):"
	@echo "  ARRAYS_DIR = $(ARRAYS_DIR)"
	@echo "  DATA_DIR   = $(DATA_DIR)"
	@echo "  DEVICE     = $(DEVICE)"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete!"

test:
	@echo "Running unit tests..."
	python test_models.py
	@echo "✓ Tests complete!"

inspect:
	@echo "Inspecting dataset..."
	python inspect_data.py \
		--arrays_dir $(ARRAYS_DIR) \
		--data_dir $(DATA_DIR)

augment:
	@echo "Creating augmented dataset (2-port symmetry)..."
	@echo "This will double your dataset size"
	@echo ""
	python augment_data.py augment \
		--arrays_dir $(ARRAYS_DIR) \
		--data_dir $(DATA_DIR) \
		--output_arrays_dir $(ARRAYS_DIR)_augmented \
		--output_data_dir $(DATA_DIR)_augmented
	@echo ""
	@echo "Copying augmented files to original directories..."
	cp $(ARRAYS_DIR)_augmented/* $(ARRAYS_DIR)/
	cp $(DATA_DIR)_augmented/* $(DATA_DIR)/
	@echo ""
	@echo "✓ Dataset augmented!"
	@echo "Check the output above for the new dataset size."
	@echo "You can now train with 2x more data."

augment-verify:
	@echo "Verifying augmentation on first sample..."
	@bash -c 'FIRST_ARRAY=$(ls $(ARRAYS_DIR)/*_sparse_array | head -1); \
	FIRST_BASE=$(basename $FIRST_ARRAY _sparse_array); \
	echo "Checking: $FIRST_BASE"; \
	python augment_data.py verify \
		--array_file $(ARRAYS_DIR)/${FIRST_BASE}_sparse_array \
		--pkl_file $(DATA_DIR)/${FIRST_BASE}_dataframe.pkl \
		--aug_array_file $(ARRAYS_DIR)_augmented/${FIRST_BASE}_flipped_sparse_array \
		--aug_pkl_file $(DATA_DIR)_augmented/${FIRST_BASE}_flipped_dataframe.pkl'

train-forward:
	@echo "Training forward model..."
	python train_forward.py \
		--arrays_dir $(ARRAYS_DIR) \
		--data_dir $(DATA_DIR) \
		--epochs 100 \
		--batch_size 32 \
		--device $(DEVICE)
	@echo "✓ Forward model training complete!"

train-inverse:
	@echo "Training inverse cVAE..."
	@if [ ! -f "$(FORWARD_CKPT)" ]; then \
		echo "Error: Forward model not found at $(FORWARD_CKPT)"; \
		echo "Please run 'make train-forward' first"; \
		exit 1; \
	fi
	python train_inverse.py \
		--arrays_dir $(ARRAYS_DIR) \
		--data_dir $(DATA_DIR) \
		--forward_checkpoint $(FORWARD_CKPT) \
		--epochs 200 \
		--batch_size 32 \
		--device $(DEVICE)
	@echo "✓ Inverse model training complete!"

train-all: train-forward train-inverse
	@echo ""
	@echo "=========================================="
	@echo "✓ Training pipeline complete!"
	@echo "=========================================="
	@echo "Next steps:"
	@echo "  1. Generate patterns: make generate TARGET=target.pkl"
	@echo "  2. Evaluate models: make evaluate"
	@echo "  3. Visualize results: make visualize"

generate:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET not specified"; \
		echo "Usage: make generate TARGET=path/to/target.pkl"; \
		exit 1; \
	fi
	@if [ ! -f "$(INVERSE_CKPT)" ]; then \
		echo "Error: Inverse model not found at $(INVERSE_CKPT)"; \
		echo "Please run 'make train-all' first"; \
		exit 1; \
	fi
	@echo "Generating patterns for target: $(TARGET)"
	python generate.py \
		--target_file $(TARGET) \
		--inverse_checkpoint $(INVERSE_CKPT) \
		--forward_checkpoint $(FORWARD_CKPT) \
		--normalization_file checkpoints/forward/normalization.pt \
		--n_candidates 50 \
		--top_k 10 \
		--device $(DEVICE)
	@echo "✓ Pattern generation complete!"

evaluate:
	@echo "Evaluating models..."
	python evaluate.py \
		--arrays_dir $(ARRAYS_DIR) \
		--data_dir $(DATA_DIR) \
		--forward_checkpoint $(FORWARD_CKPT) \
		--inverse_checkpoint $(INVERSE_CKPT) \
		--device $(DEVICE)
	@echo "✓ Evaluation complete! See evaluation_results.json"

visualize:
	@echo "Visualizing generated patterns..."
	python visualize.py \
		--output_dir generated_patterns \
		--n_show 10
	@echo "✓ Visualizations saved to generated_patterns/"

tune:
	@echo "Tuning hyperparameters..."
	@if [ ! -f "$(FORWARD_CKPT)" ]; then \
		echo "Error: Forward model not found at $(FORWARD_CKPT)"; \
		echo "Please run 'make train-forward' first"; \
		exit 1; \
	fi
	python tune_hyperparameters.py \
		--arrays_dir $(ARRAYS_DIR) \
		--data_dir $(DATA_DIR) \
		--forward_checkpoint $(FORWARD_CKPT) \
		--n_epochs 10 \
		--device $(DEVICE)
	@echo "✓ Hyperparameter tuning complete! See tuning_results.json"

clean:
	@echo "Cleaning generated files..."
	rm -rf generated_patterns/*
	rm -f evaluation_results.json
	rm -f tuning_results.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete!"

clean-all: clean
	@echo "Cleaning checkpoints (this will delete trained models)..."
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf checkpoints/*; \
		echo "✓ All checkpoints deleted!"; \
	else \
		echo "Cancelled."; \
	fi

# Quick workflow
workflow: inspect train-all evaluate
	@echo ""
	@echo "=========================================="
	@echo "✓ Complete workflow finished!"
	@echo "=========================================="
	@echo "Ready to generate patterns!"
	@echo "  make generate TARGET=your_target.pkl"

workflow-augmented: inspect augment train-all evaluate visualize
	@echo ""
	@echo "=========================================="
	@echo "✓ Complete workflow with augmentation finished!"
	@echo "=========================================="
	@echo "Trained with 2x dataset (augmented)"
	@echo "Ready to generate patterns!"
	@echo "  make generate TARGET=your_target.pkl"
