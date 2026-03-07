# refine_u — Brain Graph Super-Resolution

A Hydra-configured training package for brain graph super-resolution. Takes a low-resolution brain connectivity matrix (160x160) and predicts the corresponding high-resolution matrix (268x268). Experiments log to Weights & Biases and produce per-fold prediction CSVs.

## Setup

### With uv (recommended)

```bash
uv sync
```

### With pip

```bash
pip install -r requirements.txt
```

### Data

Place the dataset CSVs in `data/`:

```
data/
├── lr_train.csv
├── hr_train.csv
└── lr_test.csv
```

## How to Run

All commands are run from **inside** the `refine_u/` directory:

```bash
# Default config (traditional GCN, eigen features, 5-fold ensemble)
python -m train

# Run a named experiment
python -m train experiment=refine_u           # UNet + eigen (best: 0.127483 Kaggle)
python -m train experiment=gcn_mlp_eigen       # Traditional GCN + eigen
python -m train experiment=gcn_mlp             # Traditional GCN + ones (no features)
python -m train experiment=gsrn_baseline       # GSRN baseline

# Inline overrides
python -m train model.gcn_type=unet training.epochs=100
python -m train model.feat_type=betweenness training.seeds=[42,123,456]
python -m train wandb.enabled=false
```

## Outputs

Each run produces an organized output directory:

```
outputs/{run_name}/
├── folds/
│   ├── predictions_fold_1.csv
│   ├── predictions_fold_2.csv
│   └── ...
├── plots/
│   ├── hr_distribution.png
│   ├── test_predictions_distribution.png
│   └── evaluation_barplots.png
└── {run_name}_predictions.csv       # final ensemble submission
```

Per-fold CSVs and the final submission are also uploaded to W&B as artifacts.

## Directory Structure

```
refine_u/
├── train.py                    # Single entry point (Hydra @main)
├── data.py                     # Data loading, feature computation, augmentation
├── evaluate.py                 # Loss functions, full evaluation metrics, submission CSV
├── models/
│   ├── __init__.py             # Exports GraphCentMapperModel, GSRNet
│   ├── ops.py                  # Graph UNet building blocks
│   ├── graph_cent.py           # GraphCentMapperModel — the main model
│   └── gsrn.py                 # GSRNet — baseline model
├── configs/
│   ├── config.yaml             # Root config
│   ├── model/
│   │   ├── graph_cent.yaml
│   │   └── gsrn.yaml
│   ├── training/
│   │   └── default.yaml
│   └── experiment/
│       ├── refine_u.yaml       # UNet + eigen (best Kaggle: 0.127483)
│       ├── gcn_mlp_eigen.yaml
│       ├── gcn_mlp.yaml
│       └── gsrn_baseline.yaml
├── project_original_files/     # Original assignment files
├── pyproject.toml
└── requirements.txt
```

## Models

### GraphCentMapperModel (main)

GCN backbone (traditional 2-layer or Graph UNet with TopK pooling) → flatten → MLP → reshape to 268x268 → symmetrize. Supports low-rank factorization.

### GSRNet (baseline)

Reproduces the GSR-Net paper using eigenvalue decomposition-based super-resolution. Forced to CPU due to non-module subcomponents.

## Evaluation Metrics

- **MAE** — Mean Absolute Error (vectorized matrices)
- **PCC** — Pearson Correlation Coefficient
- **JSD** — Jensen-Shannon Distance
- **BC** — MAE of Betweenness Centrality
- **EC** — MAE of Eigenvector Centrality
- **PC** — MAE of PageRank Centrality
