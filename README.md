# CAFA6 Competition Codebase

## Overview
This repository contains the code for the CAFA6 competition.
The pipeline consists of:
1.  **ESM2 Fine-tuning**: Fine-tuning ESM2 model on CAFA6 data.
2.  **3Di Generation**: Using ProstT5 to generate 3Di structure tokens.
3.  **KNN Inference**: Using KNN on 3Di tokens.
4.  **Diamond**: Sequence alignment scores.
5.  **Ensemble**: Combining predictions from ESM2, KNN, and Diamond.

## Structure

```
CAFA6/
├── src/
│   ├── features/       # Feature generation (3Di, Diamond processing)
│   ├── training/       # Model training scripts (ESM2, etc.)
│   ├── inference/      # Inference scripts
│   ├── ensemble/       # Ensemble and submission generation
│   └── utils/          # Utility scripts
├── scripts/            # Shell scripts to run pipelines
├── data/               # Data directory
├── models/             # Saved models
├── logs/               # Execution logs
└── results/            # Final submission files
```

## Setup
The environment used is `cafa6`.

## Usage

### 1. Training
Run ESM2 training:
```bash
./scripts/run_training.sh
```

### 2. Feature Generation
Generate 3Di features:
```bash
./scripts/run_generation.sh
```

### 3. Inference
Run inference scripts in `src/inference/`.
Example:
```bash
python src/inference/predict_esm2.py
python src/inference/predict_3di_knn.py
```

### 4. Ensemble
Generate final submission:
```bash
python src/ensemble/ensemble_final.py
```
