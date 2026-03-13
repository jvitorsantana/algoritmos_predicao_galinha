"""
CRISP-DM Phase 4: Modeling - Weight Prediction
Trains and compares multiple regression models.
"""
import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, KFold, cross_val_score
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA PREPARATION
# =============================================================================
print("=" * 80)
print("WEIGHT PREDICTION - MODEL COMPARISON")
print("=" * 80)

df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')

FEATURES = ['IDADE', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA',
            'DORSO', 'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']

for col in FEATURES + ['PESO']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove ages with too few samples
age_counts = df['IDADE'].value_counts()
valid_ages = age_counts[age_counts >= 10].index
df = df[df['IDADE'].isin(valid_ages)]

df_clean = df.dropna(subset=FEATURES + ['PESO'])
print(f"Records after cleaning: {len(df_clean)}")

# Split by animal (same animal never in both train and test)
animals = df_clean['ANIMAL'].unique()
animals_train, animals_test = train_test_split(
    animals, test_size=0.2, random_state=42
)

train = df_clean[df_clean['ANIMAL'].isin(animals_train)]
test = df_clean[df_clean['ANIMAL'].isin(animals_test)]

X_train, y_train = train[FEATURES], train['PESO']
X_test, y_test = test[FEATURES], test['PESO']

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Animals: Train={len(animals_train)} | Test={len(animals_test)}")

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    'XGBoost': {
        'pipeline': Pipeline([('model', XGBRegressor(random_state=42, n_jobs=-1))]),
        'params': {
            'model__n_estimators': randint(100, 800),
            'model__max_depth': randint(3, 8),
            'model__learning_rate': uniform(0.005, 0.15),
            'model__subsample': uniform(0.6, 0.4),
            'model__colsample_bytree': uniform(0.5, 0.5),
            'model__gamma': uniform(0, 0.5),
            'model__min_child_weight': randint(1, 15),
            'model__reg_alpha': uniform(0, 1.0),
            'model__reg_lambda': uniform(0.5, 2.0),
        },
        'n_iter': 50,
    },
    'LightGBM': {
        'pipeline': Pipeline([('model', LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))]),
        'params': {
            'model__n_estimators': randint(100, 800),
            'model__max_depth': randint(3, 10),
            'model__learning_rate': uniform(0.005, 0.15),
            'model__subsample': uniform(0.6, 0.4),
            'model__colsample_bytree': uniform(0.5, 0.5),
            'model__min_child_samples': randint(5, 30),
            'model__reg_alpha': uniform(0, 1.0),
            'model__reg_lambda': uniform(0.5, 2.0),
        },
        'n_iter': 50,
    },
    'Random Forest': {
        'pipeline': Pipeline([('model', RandomForestRegressor(random_state=42, n_jobs=-1))]),
        'params': {
            'model__n_estimators': randint(100, 600),
            'model__max_depth': randint(5, 20),
            'model__min_samples_split': randint(2, 15),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': uniform(0.3, 0.7),
        },
        'n_iter': 30,
    },
    'Extra Trees': {
        'pipeline': Pipeline([('model', ExtraTreesRegressor(random_state=42, n_jobs=-1))]),
        'params': {
            'model__n_estimators': randint(100, 600),
            'model__max_depth': randint(5, 20),
            'model__min_samples_split': randint(2, 15),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': uniform(0.3, 0.7),
        },
        'n_iter': 30,
    },
    'Gradient Boosting': {
        'pipeline': Pipeline([('model', GradientBoostingRegressor(random_state=42))]),
        'params': {
            'model__n_estimators': randint(100, 500),
            'model__max_depth': randint(3, 8),
            'model__learning_rate': uniform(0.01, 0.15),
            'model__subsample': uniform(0.6, 0.4),
            'model__min_samples_split': randint(2, 15),
            'model__min_samples_leaf': randint(1, 10),
        },
        'n_iter': 30,
    },
    'SVR': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR()),
        ]),
        'params': {
            'model__C': uniform(0.1, 100),
            'model__epsilon': uniform(0.01, 1.0),
            'model__kernel': ['rbf', 'poly'],
            'model__gamma': ['scale', 'auto'],
        },
        'n_iter': 20,
    },
    'KNN': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsRegressor()),
        ]),
        'params': {
            'model__n_neighbors': randint(3, 30),
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan'],
        },
        'n_iter': 15,
    },
    'Ridge': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge()),
        ]),
        'params': {
            'model__alpha': uniform(0.01, 100),
        },
        'n_iter': 15,
    },
    'Lasso': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(max_iter=5000)),
        ]),
        'params': {
            'model__alpha': uniform(0.01, 50),
        },
        'n_iter': 15,
    },
    'ElasticNet': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(max_iter=5000)),
        ]),
        'params': {
            'model__alpha': uniform(0.01, 50),
            'model__l1_ratio': uniform(0.1, 0.9),
        },
        'n_iter': 15,
    },
}

# =============================================================================
# TRAINING & EVALUATION
# =============================================================================
results = []

for name, config in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Training: {name} ({config['n_iter']} iterations)")
    print(f"{'='*60}")

    search = RandomizedSearchCV(
        estimator=config['pipeline'],
        param_distributions=config['params'],
        n_iter=config['n_iter'],
        cv=cv,
        scoring='r2',
        n_jobs=1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2_cv = search.best_score_

    print(f"  R2 CV:   {r2_cv:.4f}")
    print(f"  R2 Test: {r2:.4f}")
    print(f"  RMSE:    {rmse:.2f}g")
    print(f"  MAE:     {mae:.2f}g")

    # Per-age metrics
    per_age = []
    for age in sorted(test['IDADE'].unique()):
        mask = test['IDADE'] == age
        if mask.sum() < 5:
            continue
        y_r = y_test[mask]
        y_p = y_pred[mask.values]
        age_r2 = r2_score(y_r, y_p)
        age_mae = mean_absolute_error(y_r, y_p)
        per_age.append({'age': int(age), 'r2': age_r2, 'mae': age_mae, 'n': mask.sum()})

    results.append({
        'name': name,
        'r2_cv': r2_cv,
        'r2_test': r2,
        'rmse': rmse,
        'mae': mae,
        'per_age': per_age,
        'model': best,
        'y_pred': y_pred,
    })

# =============================================================================
# COMPARISON TABLE
# =============================================================================
results.sort(key=lambda x: x['r2_test'], reverse=True)

print(f"\n{'='*80}")
print("MODEL COMPARISON (sorted by R2 Test)")
print(f"{'='*80}")
print(f"{'Model':<20} | {'R2 CV':>8} | {'R2 Test':>8} | {'RMSE':>8} | {'MAE':>8}")
print("-" * 65)
for r in results:
    print(f"{r['name']:<20} | {r['r2_cv']:>8.4f} | {r['r2_test']:>8.4f} | "
          f"{r['rmse']:>7.2f}g | {r['mae']:>7.2f}g")

# Per-age comparison for top 3 models
print(f"\n{'='*80}")
print("PER-AGE R2 - TOP 3 MODELS")
print(f"{'='*80}")
top3 = results[:3]
header = f"{'Age':>5} | {'N':>4}"
for r in top3:
    header += f" | {r['name']:>14}"
print(header)
print("-" * len(header))

all_ages = sorted(set(a['age'] for r in top3 for a in r['per_age']))
for age in all_ages:
    line = f"{age:>5}"
    n_str = ""
    for r in top3:
        age_data = next((a for a in r['per_age'] if a['age'] == age), None)
        if age_data:
            if not n_str:
                n_str = f" | {age_data['n']:>4}"
            line_r2 = f"{age_data['r2']:>14.4f}"
        else:
            if not n_str:
                n_str = " |  N/A"
            line_r2 = f"{'N/A':>14}"
        line += f" | {line_r2}"
    print(f"{line[:5]}{n_str}{line[5:]}")

# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Weight Prediction - Model Comparison', fontsize=16, fontweight='bold')

# 1. R2 comparison
ax = axes[0, 0]
names = [r['name'] for r in results]
r2_cv_vals = [r['r2_cv'] for r in results]
r2_test_vals = [r['r2_test'] for r in results]
x = np.arange(len(names))
ax.bar(x - 0.2, r2_cv_vals, 0.35, label='R2 CV', alpha=0.8, color='steelblue')
ax.bar(x + 0.2, r2_test_vals, 0.35, label='R2 Test', alpha=0.8, color='coral')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('R2')
ax.set_title('R2 Score by Model')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. MAE comparison
ax = axes[0, 1]
mae_vals = [r['mae'] for r in results]
colors = ['green' if m < 50 else 'orange' if m < 100 else 'red' for m in mae_vals]
ax.bar(names, mae_vals, color=colors, alpha=0.7)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('MAE (g)')
ax.set_title('Mean Absolute Error by Model')
ax.grid(True, alpha=0.3, axis='y')

# 3. Real vs Predicted (best model)
ax = axes[1, 0]
best = results[0]
ax.scatter(y_test, best['y_pred'], alpha=0.4, s=10)
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Ideal')
ax.set_xlabel('Real Weight (g)')
ax.set_ylabel('Predicted Weight (g)')
ax.set_title(f"Best: {best['name']} (R2={best['r2_test']:.4f})")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Per-age R2 for top 3
ax = axes[1, 1]
for r in top3:
    ages_r2 = [a['age'] for a in r['per_age']]
    vals_r2 = [a['r2'] for a in r['per_age']]
    ax.plot(ages_r2, vals_r2, marker='o', label=r['name'], linewidth=2, markersize=5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Age (days)')
ax.set_ylabel('R2')
ax.set_title('Per-Age R2 - Top 3 Models')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / 'comparacao_peso.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: resultados/comparacao_peso.png")

# Save best model info
best_info = {
    'model': results[0]['name'],
    'r2_cv': results[0]['r2_cv'],
    'r2_test': results[0]['r2_test'],
    'rmse': results[0]['rmse'],
    'mae': results[0]['mae'],
    'features': FEATURES,
    'ranking': [{'name': r['name'], 'r2_test': round(r['r2_test'], 4)} for r in results]
}
with open(RESULTS / 'comparacao_peso.json', 'w') as f:
    json.dump(best_info, f, indent=2)

print(f"\n{'='*80}")
print(f"BEST MODEL: {results[0]['name']} (R2={results[0]['r2_test']:.4f}, MAE={results[0]['mae']:.2f}g)")
print(f"{'='*80}")
