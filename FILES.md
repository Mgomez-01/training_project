# Project File Reference

Complete reference of all project files and their purposes.

## 📁 Core Files

### Configuration & Setup

| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `config.py` | Centralized hyperparameter configuration |
| `.gitignore` | Git ignore patterns |
| `Makefile` | Convenient command shortcuts |
| `run_training.sh` | Automated training pipeline script |

### Data Processing

| File | Purpose |
|------|---------|
| `data/__init__.py` | Data module initialization |
| `data/dataset.py` | Dataset loader and preprocessing |

### Model Architectures

| File | Purpose |
|------|---------|
| `models/__init__.py` | Models module initialization |
| `models/forward.py` | Forward model (Pattern → S-params) |
| `models/inverse.py` | Inverse cVAE (S-params → Pattern) |
| `models/utils.py` | Gumbel-Softmax and utility functions |

### Training Scripts

| File | Purpose |
|------|---------|
| `train_forward.py` | Train forward model |
| `train_inverse.py` | Train inverse cVAE model |

### Generation & Evaluation

| File | Purpose |
|------|---------|
| `generate.py` | Generate patterns from target S-params |
| `evaluate.py` | Evaluate trained models on validation set |
| `visualize.py` | Visualize patterns and S-parameters |

### Analysis & Utilities

| File | Purpose |
|------|---------|
| `inspect_data.py` | Inspect and validate dataset |
| `tune_hyperparameters.py` | Hyperparameter grid search |
| `test_models.py` | Unit tests for models |
| `example_usage.py` | Example code snippets |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `QUICKREF.md` | Quick reference guide |
| `DOCUMENTATION.md` | Comprehensive technical documentation |
| `FILES.md` | This file - complete file reference |

---

## 📂 Generated Directories

### checkpoints/
Created during training, contains saved models.

```
checkpoints/
├── forward/
│   ├── best_model.pt           # Best forward model weights
│   ├── normalization.pt        # Data normalization statistics
│   └── checkpoint_epoch_*.pt   # Periodic checkpoints
└── inverse/
    ├── best_model.pt           # Best inverse model weights
    └── checkpoint_epoch_*.pt   # Periodic checkpoints
```

### generated_patterns/
Created during generation, contains output patterns.

```
generated_patterns/
├── candidate_000_error_0.1234.array  # Top pattern (binary file)
├── candidate_000_error_0.1234.pkl    # Predicted S-parameters
├── candidate_001_error_0.1456.array  # Second best pattern
├── ...
└── visualization_*.png               # Visual outputs
```

---

## 🎯 File Usage by Task

### Initial Setup
```bash
requirements.txt     # Install dependencies
test_models.py       # Verify installation
inspect_data.py      # Check your data
```

### Training
```bash
config.py            # Review/modify hyperparameters
train_forward.py     # Stage 1: Train forward model
train_inverse.py     # Stage 2: Train inverse model
run_training.sh      # Or use this for both
```

### Generation
```bash
generate.py          # Generate candidate patterns
visualize.py         # Visualize results
```

### Evaluation & Tuning
```bash
evaluate.py              # Assess model performance
tune_hyperparameters.py  # Find optimal hyperparameters
```

### Development
```bash
example_usage.py     # See usage examples
test_models.py       # Run unit tests
models/*.py          # Modify architectures
```

---

## 📋 File Dependencies

### Training Dependencies
```
train_forward.py
  └─ requires: data/dataset.py, models/forward.py
  └─ produces: checkpoints/forward/

train_inverse.py
  └─ requires: data/dataset.py, models/inverse.py, models/forward.py
  └─ requires: checkpoints/forward/best_model.pt
  └─ produces: checkpoints/inverse/
```

### Generation Dependencies
```
generate.py
  └─ requires: models/inverse.py, models/forward.py
  └─ requires: checkpoints/forward/ and checkpoints/inverse/
  └─ produces: generated_patterns/
```

### Evaluation Dependencies
```
evaluate.py
  └─ requires: data/dataset.py, models/
  └─ requires: checkpoints/
  └─ produces: evaluation_results.json

visualize.py
  └─ requires: matplotlib
  └─ reads: generated_patterns/
  └─ produces: visualization PNG files
```

---

## 🔧 Modifying the Project

### To change model architecture:
- Edit `models/forward.py` or `models/inverse.py`
- Update hyperparameters in `config.py`
- Re-run training

### To add custom loss:
- Edit loss function in `train_inverse.py`
- See `example_usage.py` for examples

### To change data format:
- Modify `data/dataset.py`
- Update `inspect_data.py` for new format
- Re-verify with test suite

### To add new features:
- Create new script in project root
- Update this FILES.md
- Add tests to `test_models.py` if applicable

---

## 📏 File Sizes (Approximate)

### Source Code
```
Python files:     ~15 KB each
Total code:       ~200 KB
Documentation:    ~100 KB
```

### Generated Files
```
Forward checkpoint:   ~50 MB
Inverse checkpoint:   ~30 MB
Pattern file:         3 KB (48×32×4 bytes)
S-params file:        9 KB (pickle overhead)
```

### Expected Disk Usage
```
Project source:       < 1 MB
Checkpoints:          ~100 MB
Generated patterns:   ~1 MB per 100 candidates
Training data:        Depends on dataset size
```

---

## 🔍 Quick File Lookup

### "I want to..."

**...train models**
→ `train_forward.py`, `train_inverse.py`, or `run_training.sh`

**...generate patterns**
→ `generate.py`

**...visualize results**
→ `visualize.py`

**...understand the models**
→ `models/forward.py`, `models/inverse.py`, `DOCUMENTATION.md`

**...check my data**
→ `inspect_data.py`

**...tune hyperparameters**
→ `tune_hyperparameters.py`

**...see examples**
→ `example_usage.py`

**...run tests**
→ `test_models.py`

**...find commands**
→ `QUICKREF.md`, `Makefile`

**...understand the theory**
→ `DOCUMENTATION.md`

**...troubleshoot**
→ `QUICKREF.md` (troubleshooting section)

---

## 📝 File Creation Order

When starting from scratch:

1. ✅ `requirements.txt` → Install dependencies
2. ✅ `config.py` → Configure paths
3. ✅ `data/dataset.py` → Set up data loading
4. ✅ `models/*.py` → Define architectures
5. ✅ `train_forward.py` → Create training pipeline
6. ✅ `train_inverse.py` → Create inverse training
7. ✅ `generate.py` → Create generation script
8. ✅ `evaluate.py`, `visualize.py` → Add analysis tools
9. ✅ `test_models.py` → Add unit tests
10. ✅ Documentation files → Document everything

---

## 🎓 Learning Path

### Beginner
1. Read `README.md`
2. Run `test_models.py`
3. Try `example_usage.py`
4. Use `run_training.sh`

### Intermediate
1. Read `QUICKREF.md`
2. Understand `models/*.py`
3. Modify `config.py`
4. Try `tune_hyperparameters.py`

### Advanced
1. Read `DOCUMENTATION.md`
2. Modify model architectures
3. Implement custom losses
4. Add new features

---

## 🔐 File Permissions

### Executable Files
```bash
chmod +x run_training.sh
chmod +x Makefile  # If using make
```

### Read-Only (Recommended)
```bash
# Protect trained models from accidental deletion
chmod 444 checkpoints/*/best_model.pt
```

---

## 🗺️ Navigation Guide

```
Start Here
    ↓
README.md ──────→ Understand project
    ↓
QUICKREF.md ────→ Find commands
    ↓
inspect_data.py ─→ Verify data
    ↓
train_*.py ─────→ Train models
    ↓
generate.py ────→ Create patterns
    ↓
evaluate.py ────→ Assess performance
    ↓
DOCUMENTATION.md → Deep dive

Parallel paths:
├─ tune_hyperparameters.py → Optimize settings
├─ visualize.py ───────────→ See results
├─ test_models.py ─────────→ Debug issues
└─ example_usage.py ───────→ Learn API
```

---

**Last Updated**: October 2025  
**Total Files**: 30+ Python files + 4 documentation files  
**Lines of Code**: ~5,000 (excluding comments and docs)
