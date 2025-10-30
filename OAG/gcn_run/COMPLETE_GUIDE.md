# Complete Guide: Testing GCN on Tiny OAG Dataset
## Everything You Need - Steps, Commands, Fixes, and Documentation

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Environment Setup](#environment-setup)
4. [Step-by-Step Instructions](#step-by-step-instructions)
5. [All Fixes Applied](#all-fixes-applied)
6. [Complete Commands](#complete-commands)
7. [Troubleshooting](#troubleshooting)
8. [Expected Results](#expected-results)

---

## Quick Start

### 1. Setup Environment
```bash
cd "/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT"
source venv/bin/activate
cd OAG
```

### 2. Create Tiny Dataset
```bash
python gcn_run/create_tiny_oag.py --n_papers 50 --n_authors 20 --n_venues 5 --n_fields 10
```

### 3. Train GCN Model
```bash
# IMPORTANT: Run from OAG directory (not from gcn_run folder)
# Set PYTHONPATH so Python can find the pyHGT module
cd "/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG"
PYTHONPATH="/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG:$PYTHONPATH" python gcn_run/train_paper_field.py \
    --conv_name gcn \
    --data_dir ./dataset \
    --model_dir ./model_save \
    --task_name PF_tiny \
    --domain _CS \
    --cuda -1 \
    --n_hid 64 \
    --n_layers 2 \
    --n_epoch 5 \
    --n_batch 2 \
    --batch_size 5 \
    --n_pool 1 \
    --sample_depth 2 \
    --sample_width 10
```

**⚠️ Note**: Must run from `OAG` directory with `PYTHONPATH` set, so Python can find the `pyHGT` module.

**✅ That's it! The training should run successfully.**

---

## Overview

### What We Did
- Created a tiny synthetic OAG dataset (50 papers, 20 authors, 5 venues, 10 fields)
- Fixed NumPy 2.0 compatibility issues
- Fixed macOS multiprocessing issues
- Fixed PyTorch 2.6 loading issues
- Successfully trained GCN model on the tiny dataset

### Key Achievement
✅ **NDCG > 0.76** on test set with tiny synthetic data

### Files in gcn_run Folder
- `create_tiny_oag.py` - Dataset creation script (all required fields included)
- `train_paper_field.py` - Training script with all fixes applied
- `COMPLETE_GUIDE.md` - This document (everything in one place)

---

## Environment Setup

### Prerequisites
- Python 3.x with virtual environment
- PyHGT codebase
- All dependencies installed (see requirements.txt)

### Verify Setup
```bash
# Navigate to project
cd "/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT"

# Activate virtual environment
source venv/bin/activate

# Verify Python
python --version

# Navigate to OAG directory
cd OAG
```

---

## Step-by-Step Instructions

### Step 1: Create Tiny OAG Dataset

**Command:**
```bash
cd "/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG"
# Run from OAG directory (scripts use relative imports)
python gcn_run/create_tiny_oag.py --n_papers 50 --n_authors 20 --n_venues 5 --n_fields 10
```

**What It Does:**
- Creates synthetic graph with 50 papers, 20 authors, 5 venues, 10 fields
- Includes all required fields: `emb`, `title`, `citation`, `year`
- Saves to: `./dataset/graph_CS.pk`

**Expected Output:**
```
🚀 Creating TINY synthetic OAG dataset:
  Papers: 50
  Authors: 20
  Venues: 5
  Fields: 10
Creating edges...
✅ Dataset created successfully!
  Total papers: 50
  Total fields (L2): 10
  Papers with fields: 50
💾 Saving to ./dataset/graph_CS.pk...
✅ TINY OAG dataset created successfully!
```

---

### Step 2: Train GCN Model

**Command:**
```bash
# IMPORTANT: Must run from OAG directory, not from gcn_run folder!
# Set PYTHONPATH so Python can find the pyHGT module
cd "/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG"
PYTHONPATH="/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG:$PYTHONPATH" python gcn_run/train_paper_field.py \
    --conv_name gcn \
    --data_dir ./dataset \
    --model_dir ./model_save \
    --task_name PF_tiny \
    --domain _CS \
    --cuda -1 \
    --n_hid 64 \
    --n_layers 2 \
    --n_epoch 5 \
    --n_batch 2 \
    --batch_size 5 \
    --n_pool 1 \
    --sample_depth 2 \
    --sample_width 10
```

**Key Parameters:**
- `--conv_name gcn`: **CRITICAL** - Use GCN model (default is 'hgt')
- `--cuda -1`: Use CPU (use 0, 1, etc. for GPU)
- `--n_pool 1`: **CRITICAL** - Disable multiprocessing (required for macOS/tiny dataset)
- `--batch_size 5`: Small batch size for tiny dataset (must be ≤ available data)
- `--domain _CS`: Matches dataset filename `graph_CS.pk`

---

## All Fixes Applied

### Fix 1: Missing 'title' Field in Dataset
**Error**: `'title'`  
**Location**: `create_tiny_oag.py` (line ~34)  
**Fix Applied:**
```python
'title': ['Paper_' + str(i) for i in range(n_papers)]
```

### Fix 2: Missing 'citation' Field
**Location**: `create_tiny_oag.py`  
**Fix Applied:**
```python
'citation': np.random.randint(0, 100, n_papers)  # For papers
'citation': np.random.randint(0, 50, n_authors)  # For authors
'citation': np.random.randint(0, 200, n_venues)  # For venues
'citation': np.random.randint(0, 150, n_fields)  # For fields
```

### Fix 3: NumPy 2.0 - `np.str` Deprecated
**Error**: `module 'numpy' has no attribute 'str'`  
**Location**: `pyHGT/utils.py` and `OAG/pyHGT/utils.py` (line ~69)  
**Fix Applied:**
```python
# Before:
texts = np.array(list(...), dtype=np.str)

# After:
texts = np.array(list(...), dtype=str)
```

### Fix 4: NumPy 2.0 - `np.asfarray` Removed
**Error**: `np.asfarray was removed in the NumPy 2.0 release`  
**Location**: `pyHGT/utils.py` and `OAG/pyHGT/utils.py` (line ~6)  
**Fix Applied:**
```python
# Before:
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]

# After:
def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
```

### Fix 5: macOS Multiprocessing Issue
**Error**: `RuntimeError: An attempt has been made to start a new process...`  
**Location**: `train_paper_field.py` (multiple locations)  
**Fix Applied:**

**Change 5a - Pool Creation (line ~224):**
```python
# Before:
pool = mp.Pool(args.n_pool)

# After:
if args.n_pool > 1:
    pool = mp.Pool(args.n_pool)
else:
    pool = None
```

**Change 5b - prepare_data Function (line ~141):**
```python
def prepare_data(pool):
    if pool is None:
        # No multiprocessing - run directly
        train_data = []
        for batch_id in range(args.n_batch):
            data = node_classification_sample(randint(), sel_train_pairs, train_range)
            train_data.append(data)
        valid_data = node_classification_sample(randint(), sel_valid_pairs, valid_range)
        return train_data, valid_data
    else:
        # Use multiprocessing (original code)
        jobs = []
        for batch_id in range(args.n_batch):
            p = pool.apply_async(node_classification_sample, args=(randint(), ...))
            jobs.append(p)
        # ... rest of original code
        return jobs
```

**Change 5c - Training Loop (line ~235):**
```python
# Inside training loop
if pool is None:
    train_data, valid_data = prepare_data(pool)
else:
    jobs = prepare_data(pool)
    train_data = [job.get() for job in jobs[:-1]]
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    pool = mp.Pool(args.n_pool)
```

**Change 5d - Pool Cleanup (after training loop):**
```python
# After training loop, before test evaluation
if pool is not None:
    pool.close()
    pool.join()
```

### Fix 6: PyTorch 2.6 Loading Issue
**Error**: `WeightsUnpickler error: weights_only=True`  
**Location**: `train_paper_field.py` (line ~330)  
**Fix Applied:**
```python
# Before:
best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))

# After:
best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name), weights_only=False)
```

---

## Complete Commands

### Full Command Sequence (Copy-Paste Ready)

```bash
# 1. Navigate and activate environment
cd "/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT"
source venv/bin/activate
cd OAG

# 2. Create tiny dataset
python gcn_run/create_tiny_oag.py --n_papers 50 --n_authors 20 --n_venues 5 --n_fields 10

# 3. Train GCN model (set PYTHONPATH to find pyHGT module)
PYTHONPATH="/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG:$PYTHONPATH" python gcn_run/train_paper_field.py \
    --conv_name gcn \
    --data_dir ./dataset \
    --model_dir ./model_save \
    --task_name PF_tiny \
    --domain _CS \
    --cuda -1 \
    --n_hid 64 \
    --n_layers 2 \
    --n_epoch 5 \
    --n_batch 2 \
    --batch_size 5 \
    --n_pool 1 \
    --sample_depth 2 \
    --sample_width 10
```

---

## Expected Results

### Training Output Example
**Actual Run Results:**
```
Data Preparation: 0.0s
UPDATE!!!
Epoch: 1 (0.0s)  LR: 0.00051 Train Loss: 1.88  Valid Loss: 1.75  Valid NDCG: 0.4768
Data Preparation: 0.0s
UPDATE!!!
Epoch: 2 (0.0s)  LR: 0.00051 Train Loss: 1.75  Valid Loss: 1.84  Valid NDCG: 0.6521
Data Preparation: 0.0s
UPDATE!!!
Epoch: 3 (0.0s)  LR: 0.00052 Train Loss: 1.62  Valid Loss: 1.55  Valid NDCG: 0.6536
Data Preparation: 0.0s
UPDATE!!!
Epoch: 4 (0.0s)  LR: 0.00053 Train Loss: 1.82  Valid Loss: 1.80  Valid NDCG: 0.6633
Data Preparation: 0.0s
UPDATE!!!
Epoch: 5 (0.0s)  LR: 0.00053 Train Loss: 1.52  Valid Loss: 1.47  Valid NDCG: 0.6963
Last Test NDCG: 0.8030
Last Test MRR:  0.7086
Best Test NDCG: 0.7669
Best Test MRR:  0.6305
```

**Key Observations:**
- ✅ Training completed successfully in 5 epochs
- ✅ Training loss decreased from 1.88 → 1.52
- ✅ Validation loss improved from 1.75 → 1.47
- ✅ Validation NDCG improved from 0.4768 → 0.6963
- ✅ Final test metrics: NDCG=0.8030, MRR=0.7086

### Metrics Explanation
- **Train Loss**: Lower is better (model learning)
- **Valid Loss**: Lower is better (generalization)
- **Valid NDCG**: Higher is better (0.0-1.0, ranking quality)
- **Test NDCG**: Final ranking performance on unseen data
- **Test MRR**: Mean Reciprocal Rank (higher = better)

---

## Troubleshooting

### Issue 1: "Cannot take a larger sample than population"
**Cause**: Batch size larger than available data  
**Solution**: Reduce `--batch_size` (use 3, 4, or 5 for tiny dataset)

```bash
--batch_size 3  # Instead of 5 or 8
```

### Issue 2: Multiprocessing Errors on macOS
**Cause**: macOS spawn method incompatibility  
**Solution**: Always use `--n_pool 1` for tiny datasets

```bash
--n_pool 1  # Single process mode
```

### Issue 3: NumPy Errors
**Error**: `module 'numpy' has no attribute 'str'` or `np.asfarray was removed`  
**Solution**: All fixes already applied in:
- `pyHGT/utils.py`
- `OAG/pyHGT/utils.py`

If errors persist, manually apply fixes from [Fix 3](#fix-3-numpy-20---npstr-deprecated) and [Fix 4](#fix-4-numpy-20---npasfarray-removed).

### Issue 4: PyTorch Loading Errors
**Error**: `WeightsUnpickler error: weights_only=True`  
**Solution**: Fix already applied in `gcn_run/train_paper_field.py` (line ~330)

### Issue 5: "No valid data, skipping..."
**Cause**: Validation pairs don't exist or time ranges incorrect  
**Solution**: Check that dataset was created correctly and has papers with years in 2015-2016 range

### Issue 6: "ModuleNotFoundError: No module named 'pyHGT'"
**Cause**: Python cannot find the pyHGT module  
**Solution**: Set PYTHONPATH to include OAG directory

```bash
# ✅ CORRECT - Set PYTHONPATH and run from OAG directory
cd "/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG"
PYTHONPATH="/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG:$PYTHONPATH" python gcn_run/train_paper_field.py ...

# ❌ WRONG - Running without PYTHONPATH
cd /path/to/pyHGT/OAG
python gcn_run/train_paper_field.py ...  # Will fail with ModuleNotFoundError

# ❌ WRONG - Running from gcn_run folder
cd /path/to/pyHGT/OAG/gcn_run
python train_paper_field.py ...  # This will also fail!
```

---

## Parameter Tuning

### Adjust Dataset Size
```bash
# Smaller (for faster debugging)
python gcn_run/create_tiny_oag.py --n_papers 25 --n_authors 10 --n_venues 3 --n_fields 5

# Larger (for better results)
python gcn_run/create_tiny_oag.py --n_papers 100 --n_authors 50 --n_venues 10 --n_fields 20
```

### Adjust Model Architecture
```bash
# Larger model
--n_hid 128      # Larger hidden dimension
--n_layers 3     # More GCN layers

# Smaller model (for testing)
--n_hid 32       # Smaller hidden dimension
--n_layers 1     # Fewer layers
```

### Adjust Training Settings
```bash
# More training
--n_epoch 10     # More epochs
--n_batch 4      # More batches per epoch

# Less training (for quick tests)
--n_epoch 1      # Single epoch
--n_batch 1      # Single batch
```

### Try Different Models
```bash
# GCN (fast, baseline)
--conv_name gcn

# HGT (heterogeneous-aware, best for real OAG)
--conv_name hgt

# GAT (attention-based)
--conv_name gat
```

---

## Files Summary

### Scripts in gcn_run/
- ✅ `create_tiny_oag.py` - Creates tiny synthetic OAG dataset
- ✅ `train_paper_field.py` - Training script with all fixes applied

### Documentation
- ✅ `COMPLETE_GUIDE.md` - This file (everything in one place)

### Generated Files
- `dataset/graph_CS.pk` - Tiny dataset (created by script)
- `model_save/PF_tiny_gcn` - Saved model (created during training)

---

## Summary of All Fixes

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | Missing 'title' | `create_tiny_oag.py` | Added `'title'` field to papers |
| 2 | Missing 'citation' | `create_tiny_oag.py` | Added `'citation'` to all node types |
| 3 | `np.str` deprecated | `utils.py` (2 files) | Changed to `str` |
| 4 | `np.asfarray` removed | `utils.py` (2 files) | Changed to `np.asarray(..., dtype=float)` |
| 5 | macOS multiprocessing | `train_paper_field.py` | Added `pool=None` support for `n_pool=1` |
| 6 | PyTorch 2.6 loading | `train_paper_field.py` | Added `weights_only=False` |

**Status**: ✅ All fixes applied and tested successfully!

---

## Quick Reference Card

```bash
# IMPORTANT: All commands must be run from OAG directory!
cd "/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG"

# Create dataset
python gcn_run/create_tiny_oag.py --n_papers 50 --n_authors 20 --n_venues 5 --n_fields 10

# Train GCN (must set PYTHONPATH to find pyHGT module)
PYTHONPATH="/Users/sarasaad/Documents/BDMA /CentraleSupelec/BDRP/HGT/pyHGT/OAG:$PYTHONPATH" python gcn_run/train_paper_field.py \
    --conv_name gcn \
    --data_dir ./dataset \
    --model_dir ./model_save \
    --task_name PF_tiny \
    --domain _CS \
    --cuda -1 \
    --n_hid 64 \
    --n_layers 2 \
    --n_epoch 5 \
    --n_batch 2 \
    --batch_size 5 \
    --n_pool 1 \
    --sample_depth 2 \
    --sample_width 10
```

**🎉 Everything is ready to use!**

