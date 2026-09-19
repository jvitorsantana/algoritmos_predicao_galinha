# Chicken Biometric Prediction

## Overview
Research project (CRISP-DM) using ML to predict chicken weight and classify sex from morphometric measurements.

## Structure
- `data/raw/dataset.csv` - Main dataset (2299 records, 235 animals, 13 features). Cleaned in place: 12 decimal-point ("comma") fixes + per-animal majority SEXO (one bird = one sex). Note: `238` unique is a string-count artifact of asterisk-flagged ids (`*181` etc.); real count is 235.
- `data/raw/dataset_original.csv` - Pristine raw backup (before any cleaning); used by experimento_7
- `data/svm/` - Per-age enriched datasets (extra columns: PESO_ANTERIOR, GANHO_PESO)
- `src/eda.py` - Exploratory Data Analysis (CRISP-DM Phase 2), shared across experiments
- `src/experimento_1/` - Per-age models: XGBoost (weight) + SVM (sex), one model per age
- `src/experimento_2/` - Full dataset, manual 20% balanced split: XGBoost sex classifier + feature importance
- `src/experimento_3/` - Systematic multi-model comparison (split by animal): 10 regressors (weight) + 8 classifiers (sex)
- `src/experimento_4/` - Growth-trajectory features (rolling mean, growth rate, slope) for sex classification
- `src/experimento_5/` - XGBoost weight regression on full dataset using only the most impactful morphometric features (data-driven selection; IDADE excluded by default via INCLUDE_IDADE flag)
- `src/experimento_6/` - Feature importance (XGBoost) for both weight regression and sex classification + ROC curve / AUC for sex
- `src/experimento_7/` - Data-cleaning robustness check: sex classification on RAW vs CLEANED data (12 comma fixes + per-animal majority SEXO + dedup), GroupKFold by animal. Tests whether label noise (not lack of signal) caused the low sex AUC — result: no change, so signal is genuinely absent
- `src/experimento_8/` - Sexo por janelas cumulativas de idade (só dia 1; depois dia 1 + próxima idade; etc.). Unidade amostral = AVE (não linha), coorte fixa de 173, classes balanceadas por subamostragem (66+66), duplicatas (ANIMAL,IDADE) resolvidas mantendo o 1º registro, 70/30 estratificado repetido 30x, permutação + Benjamini-Hochberg. `preparacao.py` (dados + protocolo, compartilhado) + `experimento_8_logreg.py` + `experimento_8_xgboost.py`. Resultado: acaso até 52 d; LogReg AUC 0,701 aos 66 d e XGBoost 0,640 — o sinal existe, mas só tardiamente
- `src/figuras/` - Standalone article figures: boxplots by sex, correlation heatmaps (Spearman, aggregate + per-age), PCA and LDA by sex
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
uv run python src/experimento_7/experimento_7_rotulos_sexo.py  # Sex: raw vs cleaned data (label-noise check)
uv run python src/experimento_8/experimento_8_logreg.py    # Sex: cumulative age windows, LogReg-L2 (~2 min)
uv run python src/experimento_8/experimento_8_xgboost.py   # Sex: cumulative age windows, XGBoost (~8 min)
uv run python src/experimento_8/graficos.py                # Exp. 8 figures (run both models first)
```

## Key Conventions
- Dataset uses `;` separator and `.` decimal
- Always split by animal (same animal never in train and test)
- StratifiedKFold for classification, KFold for regression
- All numeric columns need `pd.to_numeric(col, errors='coerce')`
- Scripts resolve the project root relative to their own location: `src/eda.py` uses `.parent.parent`; scripts inside `src/experimento_N/` use `.parent.parent.parent`
- All outputs go to `results/` (gitignored)
