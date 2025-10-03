# Pattern Generation for S-Parameter Optimization

This project implements a **Conditional Variational Autoencoder (cVAE)** with **tandem forward model validation** to generate device patterns that achieve target S-parameter specifications.

## 🎯 Problem Overview

Given target S-parameters (frequency response characteristics), generate binary spatial patterns (48×32) that will produce those S-parameters when simulated.

**Input**: S-parameters (201 frequencies × 4 parameters)  
**Output**: Binary pattern (48×32 array)  
**Validation**: Forward model + actual simulation

## 🏗️ Architecture

### Two-Stage Approach

1. **Forward Model** (Pattern → S-parameters)
   - Pre-trained ResNet-style 2D CNN
   - Validates that generated patterns are physically realizable
   - Frozen during inverse model training

2. **Inverse cVAE** (S-parameters → Pattern)
   - Conditional VAE with Gumbel-Softmax for binary outputs
   - Uses forward model for physics-based validation
   - Generates multiple diverse candidates per target

```
Target S-params → [cVAE Encoder] → Latent Space (z ~ N(μ,σ))
                                          ↓
                     [cVAE Decoder] ← [z + condition]
                                          ↓
                                  Generated Pattern
                                          ↓
                              [Forward Model] (frozen)
                                          ↓
                                  Predicted S-params
                                          ↓
                            Loss = Reconstruction + KL + Forward
```

## 📁 Project Structure

```
training_project/
├── data/
│   ├── __init__.py
│   └── dataset.py          # Data loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── forward.py          # Forward model architectures
│   ├── inverse.py          # Inverse cVAE model
│   └── utils.py            # Gumbel-Softmax and utilities
├── checkpoints/            # Saved models (created during training)
│   ├── forward/
│   └── inverse/
├── train_forward.py        # Train forward model
├── train_inverse.py        # Train inverse cVAE
├── generate.py             # Generate patterns from targets
├── config.py               # Hyperparameters
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Ensure you have paired data files:
- **Patterns**: `*_sparse_array` files (48×32 binary arrays)
- **S-parameters**: `*_dataframe.pkl` files (201×5 DataFrames with Frequency, S11, S21, S22, S12)

Example naming convention:
```
20251003_022220_lab2_30_results_sparse_array
20251003_022220_lab2_30_results_dataframe.pkl
```

### 3. Train Forward Model

```bash
python train_forward.py \
    --arrays_dir python/deep_archive/arrays/ \
    --data_dir python/deep_archive/data/ \
    --epochs 100 \
    --batch_size 32 \
    --device cuda
```

**Expected outcome**: Val loss < 1.0 (normalized MSE)

### 4. Train Inverse cVAE

```bash
python train_inverse.py \
    --arrays_dir python/deep_archive/arrays/ \
    --data_dir python/deep_archive/data/ \
    --forward_checkpoint checkpoints/forward/best_model.pt \
    --epochs 200 \
    --batch_size 32 \
    --latent_dim 128 \
    --kl_weight 0.001 \
    --forward_weight 10.0 \
    --device cuda
```

**Key hyperparameters to tune**:
- `kl_weight`: Controls diversity (lower = more diverse patterns)
- `forward_weight`: Controls physical consistency (higher = closer to target S-params)

### 5. Generate Patterns

```bash
python generate.py \
    --target_file path/to/target_Sparams.pkl \
    --n_candidates 50 \
    --top_k 10 \
    --output_dir generated_patterns
```

This generates 50 candidates, saves the top 10, and outputs:
- `candidate_000_error_0.1234.array` (binary pattern)
- `candidate_000_error_0.1234.pkl` (predicted S-parameters)

## 📊 Training Tips

### Forward Model

✅ **Good signs**:
- Val loss decreasing steadily
- R² > 0.95 on validation set
- Per-frequency error < 1 dB²

⚠️ **Warning signs**:
- Val loss plateaus early (increase model capacity)
- Large gap between train/val loss (add regularization)

### Inverse cVAE

✅ **Good signs**:
- KL loss stabilizes around 10-50
- Forward loss decreases below 2.0
- Pattern accuracy > 80%
- Generated patterns show diversity

⚠️ **Warning signs**:
- All patterns look identical → decrease `kl_weight`
- Patterns look random → increase `forward_weight`
- KL loss explodes → decrease learning rate or `kl_weight`

### Hyperparameter Tuning

Start with defaults, then adjust:

| Issue | Solution |
|-------|----------|
| Patterns too similar | Decrease `kl_weight` (0.001 → 0.0005) |
| Patterns too random | Increase `forward_weight` (10 → 20) |
| Poor S-param match | Increase `forward_weight` |
| Training unstable | Decrease learning rate |

## 🔬 Workflow

1. **Train forward model** → Get val loss < 1.0
2. **Train inverse cVAE** → Monitor KL + Forward losses
3. **Generate candidates** → 20-50 patterns per target
4. **Pre-screen with forward model** → Keep top 10-20
5. **Run full simulations** → 4 min/pattern × 10 = 40 min
6. **Select best patterns** → Add to training set (optional)

## 🎛️ Advanced Options

### Data Augmentation

Currently not implemented, but you can add in `dataset.py`:
- Random flips (if patterns have symmetry)
- Random rotations (if rotationally invariant)

### Alternative Architectures

- **Simple Forward Model**: Set `--no_resnet` flag (faster, less accurate)
- **Larger Latent Space**: Increase `--latent_dim` to 256 (more expressive)
- **Different Sampling**: Modify `latent_std` in generate.py (explore design space)

### Loss Function Variants

In `train_inverse.py`, you can modify:
```python
# Weighted MSE for specific frequencies
weights = torch.ones(201)
weights[50:150] = 2.0  # Emphasize center frequencies
loss_forward = (weights * (S_pred - S_target)**2).mean()
```

## 📈 Monitoring Training

Key metrics to watch:

**Forward Model**:
- Train/Val MSE
- R² score
- Per-parameter error (S11, S21, S22, S12)

**Inverse cVAE**:
- Total loss
- Reconstruction loss (pattern accuracy)
- KL divergence (latent space regularization)
- Forward validation loss (S-parameter match)
- Temperature (Gumbel-Softmax annealing)

## 🐛 Troubleshooting

### "No paired samples found"
- Check that array and pkl files are in correct directories
- Verify naming convention matches (stem names should align)

### CUDA out of memory
- Reduce `batch_size` to 16 or 8
- Use `--device cpu` (slower but works)

### Forward model not converging
- Increase epochs to 150-200
- Check data normalization statistics
- Verify patterns are actually binary {0, 1}

### Generated patterns all black/white
- Issue with Gumbel-Softmax temperature
- Check sigmoid output range
- Inspect `pattern_probs` in generate.py

### Poor S-parameter match
- Increase `forward_weight` in inverse training
- Train forward model longer
- Check if target S-params are within training distribution

## 🔮 Future Enhancements

- [ ] Add visualization scripts for patterns and S-parameters
- [ ] Implement active learning (retrain with best simulated patterns)
- [ ] Add GAN variant for comparison
- [ ] Multi-objective optimization (bandwidth + loss + size)
- [ ] Gradient-based optimization starting from cVAE outputs
- [ ] Pattern constraints (connectivity, manufacturability)

## 📚 References

Key concepts used:
- **Conditional VAE**: Generate diverse outputs conditioned on target
- **Gumbel-Softmax**: Differentiable sampling for discrete variables
- **Tandem Networks**: Forward model validates inverse model predictions
- **Temperature Annealing**: Gradually make outputs more discrete

## 📝 Citation

If you use this code, please cite your simulation software and relevant papers on inverse design for electromagnetic structures.

## 📧 Support

For questions or issues:
1. Check this README
2. Review error messages carefully
3. Adjust hyperparameters based on "Training Tips"
4. Consult the code comments for implementation details

---

**Happy pattern generation! 🎨**
