# Chicken Biometric Prediction

## Overview
Research project (CRISP-DM) using ML to predict chicken weight and classify sex from morphometric measurements.

## Structure
- `data/raw/dataset.csv` - Main dataset (2299 records, 238 animals, 13 features)
- `data/svm/` - Per-age enriched datasets (extra columns: PESO_ANTERIOR, GANHO_PESO)
- `src/eda.py` - Exploratory Data Analysis (CRISP-DM Phase 2), shared across experiments
- `src/experimento_1/` - Per-age models: XGBoost (weight) + SVM (sex), one model per age
- `src/experimento_2/` - Full dataset, manual 20% balanced split: XGBoost sex classifier + feature importance
- `src/experimento_3/` - Systematic multi-model comparison (split by animal): 10 regressors (weight) + 8 classifiers (sex)
- `src/experimento_4/` - Growth-trajectory features (rolling mean, growth rate, slope) for sex classification
- `src/experimento_5/` - XGBoost weight regression on full dataset using only the most impactful morphometric features (data-driven selection; IDADE excluded by default via INCLUDE_IDADE flag)
- `src/experimento_6/` - Feature importance (XGBoost) for both weight regression and sex classification + ROC curve / AUC for sex
- `notebooks/` - Jupyter notebooks (Experiment 2 unified XGBoost weight model)
- `results/` - Generated outputs (gitignored): figures, models, predictions

## Running
```bash
uv run python src/eda.py                              # EDA
uv run python src/experimento_3/comparacao_peso.py    # Weight model comparison (10 models)
uv run python src/experimento_3/comparacao_sexo.py    # Sex model comparison (8 models)
uv run python src/experimento_4/experimento_4_sexo.py # Sex via growth features
uv run python src/experimento_5/experimento_5_peso.py # Weight via XGBoost + top features
uv run python src/experimento_6/experimento_6.py      # Feature importance (both) + ROC/AUC (sex)
```

## Key Conventions
- Dataset uses `;` separator and `.` decimal
- Always split by animal (same animal never in train and test)
- StratifiedKFold for classification, KFold for regression
- All numeric columns need `pd.to_numeric(col, errors='coerce')`
- Scripts resolve the project root relative to their own location: `src/eda.py` uses `.parent.parent`; scripts inside `src/experimento_N/` use `.parent.parent.parent`
- All outputs go to `results/` (gitignored)
