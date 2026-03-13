# Chicken Biometric Prediction

## Overview
Research project (CRISP-DM) using ML to predict chicken weight and classify sex from morphometric measurements.

## Structure
- `data/raw/dataset.csv` - Main dataset (2299 records, 238 animals, 13 features)
- `data/svm/` - Per-age enriched datasets (extra columns: PESO_ANTERIOR, GANHO_PESO)
- `src/` - All Python scripts (EDA, model comparison, individual models)
- `notebooks/` - Jupyter notebooks
- `results/` - Generated outputs (gitignored): figures, models, predictions

## Running
```bash
uv run python src/eda.py                # EDA
uv run python src/comparacao_peso.py    # Weight model comparison
uv run python src/comparacao_sexo.py    # Sex model comparison
```

## Key Conventions
- Dataset uses `;` separator and `.` decimal
- Always split by animal (same animal never in train and test)
- StratifiedKFold for classification, KFold for regression
- All numeric columns need `pd.to_numeric(col, errors='coerce')`
- Scripts use `ROOT = Path(__file__).resolve().parent.parent` for project-root-relative paths
- All outputs go to `results/` (gitignored)
