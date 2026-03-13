# Previsao do Sexo e Peso de Galinhas D'Angola a partir das medidas corporais

Aplicacoes no manejo avicola.

## Project Structure

```
├── data/
│   ├── raw/dataset.csv          # Original dataset (2299 records, 238 animals)
│   └── svm/                     # Per-age enriched datasets for SVM models
├── src/
│   ├── eda.py                   # Exploratory Data Analysis (CRISP-DM Phase 2)
│   ├── comparacao_peso.py       # Weight regression - 10 model comparison
│   ├── comparacao_sexo.py       # Sex classification - 8 model comparison
│   ├── svm_treino.py            # SVM sex classification (per age)
│   ├── svm_teste.py             # SVM sex prediction (per age)
│   ├── xgb_peso_treino.py       # XGBoost weight training (per age)
│   ├── xgb_peso_teste.py        # XGBoost weight prediction (per age)
│   ├── xgb_sexo_treino.py       # XGBoost sex classification training
│   └── xgb_sexo_teste.py        # XGBoost sex classification testing
├── notebooks/
│   └── xgboost_peso.ipynb       # XGBoost unified weight model (train + test)
├── results/                     # Generated outputs (gitignored)
│   ├── figures/
│   ├── models/
│   └── predictions/
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

# 2. Model comparison - Weight prediction (10 models)
uv run python src/comparacao_peso.py

# 3. Model comparison - Sex classification (8 models)
uv run python src/comparacao_sexo.py
```

## Dataset

- **238 animals** measured at **12 ages** (0 to 115 days)
- **13 morphometric features**: BICO, CIRCFCABECA, PESCOCO, ASA, TULIPA, DORSO, VENTRE, CIRCFABDOM, SOBRECOXA, COXA, CANELA, UNHAMAIOR
- **Targets**: PESO (weight, regression) and SEXO (sex, classification)
