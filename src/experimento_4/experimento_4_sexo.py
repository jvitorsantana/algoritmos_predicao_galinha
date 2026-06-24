"""
CRISP-DM Phase 4: Modeling - Sex Classification
Experiment 4: Growth relationship (IDADE x PESO) with rolling-mean forecast.

Idea:
  - Each animal has a time-series of (IDADE, PESO).
  - Build growth features: rolling mean of the last N weights (predicts next),
    residual between actual and predicted weight, growth rates, slope of
    weight vs age, etc.
  - Use these features to classify SEXO.
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
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# Rolling window: how many previous measurements to average when predicting next PESO
WINDOW = 3

# =============================================================================
# DATA PREPARATION
# =============================================================================
print("=" * 80)
print("SEX CLASSIFICATION - EXPERIMENT 4 (growth + rolling-mean forecast)")
print("=" * 80)

df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')

BASE_FEATURES = ['PESO', 'IDADE', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA',
                 'DORSO', 'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']

for col in BASE_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['SEXO', 'ANIMAL', 'IDADE', 'PESO'])
df = df.sort_values(['ANIMAL', 'IDADE']).reset_index(drop=True)

# =============================================================================
# FEATURE ENGINEERING: GROWTH RELATIONSHIP (IDADE x PESO)
# =============================================================================
# For each animal, build time-series features from PESO x IDADE.
# PESO_PRED_MA: mean of the last WINDOW weights — our "forecast" for next PESO.
# PESO_RESID:   actual PESO minus predicted (how much animal deviates from its own trend).
# GANHO_PESO:   weight gain since previous measurement.
# TAXA_CRESC:   growth rate = GANHO_PESO / delta(IDADE).
# TAXA_CRESC_MA:rolling mean of growth rate over last WINDOW steps.
# PESO_ACUM_MA: overall average weight per animal up to current row.
# SLOPE_IA:     linear regression slope of PESO vs IDADE up to current row.

def _rolling_past_mean(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window=window, min_periods=1).mean()


def _past_slope(group: pd.DataFrame) -> pd.Series:
    g = group.sort_index()
    slopes = [np.nan] * len(g)
    for i in range(2, len(g)):
        x = g['IDADE'].iloc[:i].values
        y = g['PESO'].iloc[:i].values
        if np.ptp(x) > 0:
            slopes[i] = np.polyfit(x, y, 1)[0]
    return pd.Series(slopes, index=g.index)


grp_peso = df.groupby('ANIMAL')['PESO']
grp_idade = df.groupby('ANIMAL')['IDADE']

# Rolling-mean forecast of next PESO using the last `window` weights (shifted so we
# only use past values — avoids leakage from the current row).
df['PESO_PRED_MA'] = grp_peso.transform(lambda s: _rolling_past_mean(s, WINDOW))
df['PESO_RESID'] = df['PESO'] - df['PESO_PRED_MA']

# Weight gain / growth rate vs previous measurement of the same animal.
prev_peso = grp_peso.shift(1)
prev_idade = grp_idade.shift(1)
df['GANHO_PESO'] = df['PESO'] - prev_peso
delta_idade = (df['IDADE'] - prev_idade).replace(0, np.nan)
df['TAXA_CRESC'] = df['GANHO_PESO'] / delta_idade
df['TAXA_CRESC_MA'] = df.groupby('ANIMAL')['TAXA_CRESC'].transform(
    lambda s: _rolling_past_mean(s, WINDOW)
)

# Cumulative mean of past weights (animal's baseline so far).
df['PESO_ACUM_MA'] = grp_peso.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

# Linear-regression slope of PESO vs IDADE over all past observations of the animal.
df['SLOPE_IA'] = df.groupby('ANIMAL', group_keys=False).apply(_past_slope)

GROWTH_FEATURES = ['IDADE', 'PESO', 'PESO_PRED_MA', 'PESO_RESID',
                   'GANHO_PESO', 'TAXA_CRESC', 'TAXA_CRESC_MA',
                   'PESO_ACUM_MA', 'SLOPE_IA']
FEATURES = GROWTH_FEATURES

# Drop rows where we don't yet have enough history for a growth signal.
df_clean = df.dropna(subset=FEATURES).copy()

le = LabelEncoder()
df_clean['SEXO_NUM'] = le.fit_transform(df_clean['SEXO'])

print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"Records after cleaning: {len(df_clean)}")
print(f"Class distribution:\n{df_clean['SEXO'].value_counts()}")
print(f"Features: {FEATURES}")

# Sanity check on forecast quality (rolling-mean as a PESO predictor).
mae = (df_clean['PESO'] - df_clean['PESO_PRED_MA']).abs().mean()
rmse = np.sqrt(((df_clean['PESO'] - df_clean['PESO_PRED_MA']) ** 2).mean())
print(f"\nRolling-mean PESO forecast — MAE: {mae:.2f} g | RMSE: {rmse:.2f} g (window={WINDOW})")

# Split by animal
animals = df_clean['ANIMAL'].unique()
animals_train, animals_test = train_test_split(
    animals, test_size=0.2, random_state=42
)

train = df_clean[df_clean['ANIMAL'].isin(animals_train)]
test = df_clean[df_clean['ANIMAL'].isin(animals_test)]

X_train, y_train = train[FEATURES], train['SEXO_NUM']
X_test, y_test = test[FEATURES], test['SEXO_NUM']

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Train class balance: {dict(y_train.value_counts())}")
print(f"scale_pos_weight: {scale_weight:.4f}")

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    'XGBoost': {
        'pipeline': Pipeline([('smote', SMOTE(random_state=42)), ('model', XGBClassifier(
            random_state=42, n_jobs=-1, eval_metric='logloss',
        ))]),
        'params': {
            'model__n_estimators': randint(50, 500),
            'model__max_depth': randint(2, 7),
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
        'pipeline': Pipeline([('smote', SMOTE(random_state=42)), ('model', LGBMClassifier(
            random_state=42, n_jobs=-1, verbose=-1,
        ))]),
        'params': {
            'model__n_estimators': randint(50, 500),
            'model__max_depth': randint(2, 10),
            'model__learning_rate': uniform(0.005, 0.15),
            'model__subsample': uniform(0.6, 0.4),
            'model__colsample_bytree': uniform(0.5, 0.5),
            'model__min_child_samples': randint(5, 50),
            'model__reg_alpha': uniform(0, 1.0),
            'model__reg_lambda': uniform(0.5, 2.0),
        },
        'n_iter': 50,
    },
    'Random Forest': {
        'pipeline': Pipeline([('smote', SMOTE(random_state=42)), ('model', RandomForestClassifier(
            random_state=42, n_jobs=-1,
        ))]),
        'params': {
            'model__n_estimators': randint(100, 600),
            'model__max_depth': randint(3, 20),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 15),
            'model__max_features': uniform(0.3, 0.7),
        },
        'n_iter': 30,
    },
    'Extra Trees': {
        'pipeline': Pipeline([('smote', SMOTE(random_state=42)), ('model', ExtraTreesClassifier(
            random_state=42, n_jobs=-1,
        ))]),
        'params': {
            'model__n_estimators': randint(100, 600),
            'model__max_depth': randint(3, 20),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 15),
            'model__max_features': uniform(0.3, 0.7),
        },
        'n_iter': 30,
    },
    'Gradient Boosting': {
        'pipeline': Pipeline([('smote', SMOTE(random_state=42)), ('model', GradientBoostingClassifier(random_state=42))]),
        'params': {
            'model__n_estimators': randint(50, 400),
            'model__max_depth': randint(2, 7),
            'model__learning_rate': uniform(0.01, 0.15),
            'model__subsample': uniform(0.6, 0.4),
            'model__min_samples_split': randint(2, 15),
            'model__min_samples_leaf': randint(1, 10),
        },
        'n_iter': 30,
    },
    'SVM': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', SVC(probability=True)),
        ]),
        'params': {
            'model__C': uniform(0.01, 100),
            'model__kernel': ['rbf', 'poly', 'linear'],
            'model__gamma': ['scale', 'auto'],
        },
        'n_iter': 20,
    },
    'KNN': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', KNeighborsClassifier()),
        ]),
        'params': {
            'model__n_neighbors': randint(3, 50),
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan'],
        },
        'n_iter': 15,
    },
    'Logistic Regression': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', LogisticRegression(max_iter=2000)),
        ]),
        'params': {
            'model__C': uniform(0.001, 100),
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['saga'],
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
        scoring='f1',
        n_jobs=1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_cv = search.best_score_

    print(f"  F1 CV:   {f1_cv:.4f}")
    print(f"  F1 Test: {f1:.4f}")
    print(f"  Acc:     {acc:.4f}")
    print(f"  Prec:    {prec:.4f}")
    print(f"  Rec:     {rec:.4f}")

    # Per-age accuracy
    per_age = []
    for age in sorted(test['IDADE'].unique()):
        mask = test['IDADE'] == age
        if mask.sum() < 5:
            continue
        y_r = y_test[mask]
        y_p = y_pred[mask.values]
        age_acc = accuracy_score(y_r, y_p)
        age_f1 = f1_score(y_r, y_p, zero_division=0)
        per_age.append({'age': int(age), 'acc': age_acc, 'f1': age_f1, 'n': int(mask.sum())})

    results.append({
        'name': name,
        'f1_cv': f1_cv,
        'f1_test': f1,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'per_age': per_age,
        'model': best,
        'y_pred': y_pred,
    })

# =============================================================================
# COMPARISON TABLE
# =============================================================================
results.sort(key=lambda x: x['f1_test'], reverse=True)

print(f"\n{'='*80}")
print("MODEL COMPARISON (sorted by F1 Test) - GROWTH FEATURES")
print(f"{'='*80}")
print(f"{'Model':<20} | {'F1 CV':>8} | {'F1 Test':>8} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8}")
print("-" * 75)
for r in results:
    print(f"{r['name']:<20} | {r['f1_cv']:>8.4f} | {r['f1_test']:>8.4f} | "
          f"{r['acc']:>8.4f} | {r['prec']:>8.4f} | {r['rec']:>8.4f}")

# Random baseline
majority_class = y_train.mode()[0]
baseline_acc = (y_test == majority_class).mean()
print(f"\n  Random baseline (majority class): Acc={baseline_acc:.4f}")

# Per-age comparison for top 3
print(f"\n{'='*80}")
print("PER-AGE ACCURACY - TOP 3 MODELS")
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
            val = f"{age_data['acc']:>14.4f}"
        else:
            if not n_str:
                n_str = " |  N/A"
            val = f"{'N/A':>14}"
        line += f" | {val}"
    print(f"{line[:5]}{n_str}{line[5:]}")

# Best model classification report
print(f"\n{'='*80}")
print(f"CLASSIFICATION REPORT - {results[0]['name']} (Best)")
print(f"{'='*80}")
print(classification_report(y_test, results[0]['y_pred'],
                            target_names=le.classes_, zero_division=0))

# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Classificação de Sexo - Experimento 4 (Features de Crescimento)', fontsize=16, fontweight='bold')

# 1. F1 comparison
ax = axes[0, 0]
names = [r['name'] for r in results]
f1_cv_vals = [r['f1_cv'] for r in results]
f1_test_vals = [r['f1_test'] for r in results]
x = np.arange(len(names))
ax.bar(x - 0.2, f1_cv_vals, 0.35, label='F1 Validação Cruzada', alpha=0.8, color='steelblue')
ax.bar(x + 0.2, f1_test_vals, 0.35, label='F1 Teste', alpha=0.8, color='coral')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Pontuação F1')
ax.set_title('Pontuação F1 por Modelo')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Linha de base aleatória')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 2. Metrics comparison (best model)
ax = axes[0, 1]
best = results[0]
metrics = ['Acurácia', 'Precisão', 'Revocação', 'F1']
vals = [best['acc'], best['prec'], best['rec'], best['f1_test']]
colors = ['steelblue', 'green', 'orange', 'coral']
ax.bar(metrics, vals, color=colors, alpha=0.7)
ax.set_ylim(0, 1)
ax.set_title(f'Melhor Modelo: {best["name"]}')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

# 3. Confusion matrix (best model)
ax = axes[1, 0]
cm = confusion_matrix(y_test, results[0]['y_pred'])
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title(f'Matriz de Confusão - {results[0]["name"]}')
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(le.classes_)
ax.set_yticklabels(le.classes_)
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black', fontsize=14)
ax.set_ylabel('Real')
ax.set_xlabel('Previsto')

# 4. Growth curves by sex (mean PESO vs IDADE)
ax = axes[1, 1]
for sexo, sub in df_clean.groupby('SEXO'):
    grp = sub.groupby('IDADE')['PESO'].mean()
    ax.plot(grp.index, grp.values, marker='o', label=f'{sexo} (média)', linewidth=2)
ax.set_xlabel('IDADE (dias)')
ax.set_ylabel('PESO (g)')
ax.set_title('Curva de Crescimento por Sexo')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / 'experimento_4_sexo.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: results/figures/experimento_4_sexo.png")

# Save results
best_info = {
    'experiment': 'experimento_4_sexo',
    'approach': f'growth features + rolling-mean (window={WINDOW}) forecast',
    'model': results[0]['name'],
    'f1_cv': results[0]['f1_cv'],
    'f1_test': results[0]['f1_test'],
    'acc': results[0]['acc'],
    'baseline_acc': float(baseline_acc),
    'forecast_mae': float(mae),
    'forecast_rmse': float(rmse),
    'features': FEATURES,
    'window': WINDOW,
    'ranking': [{'name': r['name'], 'f1_test': round(r['f1_test'], 4)} for r in results]
}
with open(RESULTS / 'experimento_4_sexo.json', 'w') as f:
    json.dump(best_info, f, indent=2)

print(f"\n{'='*80}")
print(f"BEST MODEL: {results[0]['name']} (F1={results[0]['f1_test']:.4f}, Acc={results[0]['acc']:.4f})")
print(f"Majority baseline: Acc={baseline_acc:.4f}")
if results[0]['acc'] < baseline_acc + 0.05:
    print("\nWARNING: Best model barely beats the majority baseline.")
    print("The growth features may not carry enough signal to classify sex.")
print(f"{'='*80}")
