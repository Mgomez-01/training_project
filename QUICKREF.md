# Pattern Generation Project - Quick Reference

## 🚀 Quick Start Commands

### 1. **Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python test_models.py

# Inspect your data
python inspect_data.py \
    --arrays_dir python/deep_archive/arrays/ \
    --data_dir python/deep_archive/data/
```

### 2. **Training**

```bash
# Method 1: Use automated script
./run_training.sh all

# Method 2: Train step-by-step
python train_forward.py \
    --arrays_dir python/deep_archive/arrays/ \
    --data_dir python/deep_archive/data/ \
    --epochs 100 \
    --device cuda

python train_inverse.py \
    --arrays_dir python/deep_archive/arrays/ \
    --data_dir python/deep_archive/data/ \
    --forward_checkpoint checkpoints/forward/best_model.pt \
    --epochs 200 \
    --device cuda
```

### 3. **Generate Patterns**

```bash
python generate.py \
    --target_file path/to/target_Sparams.pkl \
    --n_candidates 50 \
    --top_k 10 \
    --output_dir generated_patterns
```

### 4. **Evaluation**

```bash
# Evaluate trained models
python evaluate.py \
    --arrays_dir python/deep_archive/arrays/ \
    --data_dir python/deep_archive/data/ \
    --forward_checkpoint checkpoints/forward/best_model.pt \
    --inverse_checkpoint checkpoints/inverse/best_model.pt

# Visualize results
python visualize.py --output_dir generated_patterns --n_show 10
```

---

## 📊 Hyperparameter Tuning

### When to Tune
- Patterns look too similar (no diversity)
- Patterns look random (poor S-param match)
- Poor convergence during training

### Quick Tuning
```bash
python tune_hyperparameters.py \
    --arrays_dir python/deep_archive/arrays/ \
    --data_dir python/deep_archive/data/ \
    --forward_checkpoint checkpoints/forward/best_model.pt \
    --kl_weights 0.0001 0.0005 0.001 0.005 0.01 \
    --forward_weights 5.0 10.0 15.0 20.0 \
    --n_epochs 10
```

### Hyperparameter Guide

| Parameter | Default | Low Value | High Value | Effect |
|-----------|---------|-----------|------------|--------|
| `kl_weight` | 0.001 | 0.0001 | 0.01 | Lower = more diverse patterns |
| `forward_weight` | 10.0 | 5.0 | 20.0 | Higher = better S-param match |
| `latent_dim` | 128 | 64 | 256 | Higher = more expressive |
| `learning_rate` | 1e-4 | 5e-5 | 5e-4 | Adjust if training unstable |

---

## 🐛 Troubleshooting

### Problem: Models not learning
**Solution:**
```bash
# Check data
python inspect_data.py --arrays_dir <path> --data_dir <path>

# Reduce learning rate
python train_inverse.py ... --lr 5e-5

# Increase forward weight
python train_inverse.py ... --forward_weight 20.0
```

### Problem: All patterns look the same
**Solution:**
```bash
# Decrease KL weight for more diversity
python train_inverse.py ... --kl_weight 0.0005

# Try larger latent space
python train_inverse.py ... --latent_dim 256
```

### Problem: Patterns look random
**Solution:**
```bash
# Increase forward weight
python train_inverse.py ... --forward_weight 15.0

# Train forward model longer
python train_forward.py ... --epochs 150
```

### Problem: CUDA out of memory
**Solution:**
```bash
# Reduce batch size
python train_*.py ... --batch_size 16

# Or use CPU
python train_*.py ... --device cpu
```

---

## 📈 Monitoring Training

### Good Signs
- ✅ Forward model: Val loss < 1.0, decreasing steadily
- ✅ Inverse model: KL loss stabilizes (10-50), Forward loss < 2.0
- ✅ Pattern accuracy > 80%
- ✅ Generated patterns show diversity

### Warning Signs
- ⚠️ Val loss >> Train loss → Add regularization
- ⚠️ Loss plateaus early → Increase model capacity
- ⚠️ KL loss explodes → Decrease `kl_weight` or learning rate
- ⚠️ All patterns identical → Decrease `kl_weight`

---

## 🔬 Workflow Summary

```
1. Inspect Data
   ↓
2. Train Forward Model (100 epochs)
   ↓ (achieve val_loss < 1.0)
3. Train Inverse cVAE (200 epochs)
   ↓ (monitor KL + Forward losses)
4. Generate Candidates (50 patterns)
   ↓ (use forward model to pre-screen)
5. Simulate Top 10 Patterns (40 minutes)
   ↓
6. Select Best Patterns
   ↓
7. (Optional) Retrain with new data
```

---

## 📁 Directory Structure After Training

```
training_project/
├── checkpoints/
│   ├── forward/
│   │   ├── best_model.pt          # Best forward model
│   │   ├── normalization.pt       # Data normalization stats
│   │   └── checkpoint_epoch_*.pt
│   └── inverse/
│       ├── best_model.pt          # Best inverse model
│       └── checkpoint_epoch_*.pt
├── generated_patterns/
│   ├── candidate_000_error_*.array  # Top patterns
│   ├── candidate_000_error_*.pkl    # Predicted S-params
│   └── visualization_*.png          # Visualizations
├── evaluation_results.json        # Model metrics
└── tuning_results.json            # Hyperparameter tuning results
```

---

## 💡 Advanced Usage

### Custom Loss Functions

Edit `train_inverse.py` to add frequency-weighted loss:

```python
# Emphasize specific frequency range
freq_weights = torch.ones(201)
freq_weights[50:150] = 2.0  # Center frequencies

loss_forward = (freq_weights.unsqueeze(0).unsqueeze(-1) * 
                (S_pred - S_target)**2).mean()
```

### Active Learning Loop

```bash
# 1. Generate patterns
python generate.py --target_file target.pkl --n_candidates 100

# 2. Simulate top candidates
# ... run your simulator ...

# 3. Add best results to training data
# ... copy to python/deep_archive/ ...

# 4. Retrain models with expanded dataset
python train_forward.py ...
python train_inverse.py ...
```

### Batch Generation

```python
# Generate for multiple targets
from generate import PatternGenerator

generator = PatternGenerator(...)

for target_file in target_files:
    target = pd.read_pickle(target_file)
    candidates = generator.generate_candidates(target, n_candidates=50)
    generator.save_patterns(candidates[:10], f'output/{target_file.stem}')
```

---

## 📞 Support Checklist

Before asking for help, check:

1. ✅ Ran `python test_models.py` successfully
2. ✅ Ran `python inspect_data.py` and data looks correct
3. ✅ Forward model val loss < 1.0
4. ✅ Tried adjusting hyperparameters
5. ✅ Checked GPU memory (reduce batch_size if needed)
6. ✅ Read error messages carefully

---

## 🎯 Expected Performance

### Forward Model
- Training time: ~30-60 min (100 epochs, 2800 samples)
- Val MSE: < 1.0 (normalized)
- Per-parameter error: < 2 dB²

### Inverse cVAE
- Training time: ~60-120 min (200 epochs, 2800 samples)
- Pattern accuracy: > 80%
- Forward validation MSE: < 2.0
- Diversity: > 0.3 (30% different between samples)

### Generation
- Time per candidate: ~0.1 seconds
- 50 candidates: ~5 seconds
- Top-10 typically within 2x of best possible error

---

## 🔗 File Dependencies

```
Forward Training:
  arrays/ + data/ → train_forward.py → checkpoints/forward/

Inverse Training:
  arrays/ + data/ + checkpoints/forward/ → train_inverse.py → checkpoints/inverse/

Generation:
  target.pkl + checkpoints/ → generate.py → generated_patterns/
```

---

## 📝 Checklist for Production

- [ ] Forward model val loss < 1.0
- [ ] Inverse model forward loss < 2.0
- [ ] Tested on at least 10 different targets
- [ ] Verified generated patterns with simulations
- [ ] Documented hyperparameter choices
- [ ] Saved best checkpoints
- [ ] Created backup of training data

---

**Last Updated**: October 2025  
**Project**: Pattern Generation for S-Parameter Optimization
