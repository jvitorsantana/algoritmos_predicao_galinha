# Previsao do Sexo e Peso de Galinhas D'Angola a partir das medidas corporais

Aplicacoes no manejo avicola.

## Project Structure

```
├── data/
│   ├── raw/dataset.csv          # Original dataset (2299 records, 238 animals)
│   └── svm/                     # Per-age enriched datasets for SVM models
├── src/
│   ├── eda.py                   # Exploratory Data Analysis (CRISP-DM Phase 2) - shared
│   ├── experimento_1/           # Per-age models (one model per age)
│   │   ├── xgb_peso_treino.py   #   XGBoost weight training (per age)
│   │   ├── xgb_peso_teste.py    #   XGBoost weight prediction (per age)
│   │   ├── svm_treino.py        #   SVM sex classification (per age)
│   │   └── svm_teste.py         #   SVM sex prediction (per age)
│   ├── experimento_2/           # Full dataset, manual 20% balanced split (XGBoost)
│   │   ├── xgb_sexo_treino.py   #   XGBoost sex training + feature importance
│   │   └── xgb_sexo_teste.py    #   XGBoost sex testing
│   ├── experimento_3/           # Systematic multi-model comparison (split by animal)
│   │   ├── comparacao_peso.py   #   Weight regression - 10 model comparison
│   │   └── comparacao_sexo.py   #   Sex classification - 8 model comparison
│   ├── experimento_4/           # Growth-trajectory features for sex
│   │   └── experimento_4_sexo.py
│   ├── experimento_5/           # XGBoost weight regression - most impactful features only
│   │   └── experimento_5_peso.py
│   └── experimento_6/           # Feature importance (weight & sex) + ROC/AUC (sex)
│       └── experimento_6.py
├── notebooks/
│   └── xgboost_peso.ipynb       # Experiment 2: unified XGBoost weight model (train + test)
├── results/                     # Generated outputs (gitignored)
│   ├── figures/
│   ├── models/
│   └── predictions/
├── relatorio.pdf                # Full research report (4 experiments)
├── FINDINGS.md                  # Research findings and analysis
├── pyproject.toml               # Dependencies (managed with uv)
└── .gitignore
```

## Setup

```bash
uv sync
```

## Usage

```bash
# 1. Exploratory Data Analysis
uv run python src/eda.py

# Experiment 1 - Per-age models
uv run python src/experimento_1/xgb_peso_treino.py   # weight (train)
uv run python src/experimento_1/xgb_peso_teste.py    # weight (test)
uv run python src/experimento_1/svm_treino.py        # sex (train)
uv run python src/experimento_1/svm_teste.py         # sex (test)

# Experiment 2 - Full dataset, XGBoost sex classifier
uv run python src/experimento_2/xgb_sexo_treino.py
uv run python src/experimento_2/xgb_sexo_teste.py

# Experiment 3 - Systematic model comparison
uv run python src/experimento_3/comparacao_peso.py   # weight (10 models)
uv run python src/experimento_3/comparacao_sexo.py   # sex (8 models)

# Experiment 4 - Sex from growth-trajectory features
uv run python src/experimento_4/experimento_4_sexo.py

# Experiment 5 - Weight via XGBoost using only the most impactful features
uv run python src/experimento_5/experimento_5_peso.py

# Experiment 6 - Feature importance (weight & sex) + ROC curve / AUC (sex)
uv run python src/experimento_6/experimento_6.py
```

## Dataset

- **238 animals** measured at **12 ages** (0 to 115 days)
- **13 morphometric features**: BICO, CIRCFCABECA, PESCOCO, ASA, TULIPA, DORSO, VENTRE, CIRCFABDOM, SOBRECOXA, COXA, CANELA, UNHAMAIOR
- **Targets**: PESO (weight, regression) and SEXO (sex, classification)
