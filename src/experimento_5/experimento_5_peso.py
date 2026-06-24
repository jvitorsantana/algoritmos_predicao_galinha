"""
CRISP-DM Phase 4: Modeling - Weight Prediction
Experiment 5: XGBoost weight regression on the FULL dataset (no per-age split),
using ONLY the most impactful features.

Approach:
  1. Train an XGBoost regressor on ALL candidate features (the 12 morphometric
     measurements; IDADE is excluded by default - see INCLUDE_IDADE) and rank
     features by gain-based importance.
  2. Select the most impactful features: the smallest set whose cumulative
     importance reaches CUMULATIVE_THRESHOLD (default 0.95).
  3. Retrain XGBoost (with hyperparameter search) using ONLY the selected features.
  4. Compare full-feature vs selected-feature models to confirm parity.

All figures are saved with the `experimento_5_` prefix.
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from scipy.stats import randint, uniform

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
MODEL_DIR = RESULTS / 'models' / 'experimento_5'
FIGURES.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Fraction of total importance the selected features must cover.
CUMULATIVE_THRESHOLD = 0.95

# =============================================================================
# DATA PREPARATION
# =============================================================================
print("=" * 80)
print("WEIGHT PREDICTION - EXPERIMENT 5 (XGBoost, full dataset, top features)")
print("=" * 80)

df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')

# IDADE is excluded by default so the model must predict weight from body
# measurements ALONE (age is what inflates the global R2). Set INCLUDE_IDADE=True
# to add age back as a predictor and compare.
INCLUDE_IDADE = False

MORPHOMETRIC_FEATURES = ['BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA',
                         'DORSO', 'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']
CANDIDATE_FEATURES = (['IDADE'] if INCLUDE_IDADE else []) + MORPHOMETRIC_FEATURES
print(f"IDADE used as predictor: {INCLUDE_IDADE}")

# Always coerce IDADE/PESO too: IDADE is still needed for the per-age breakdown.
for col in set(CANDIDATE_FEATURES + ['IDADE', 'PESO']):
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop ages with too few samples (keeps per-age breakdown meaningful).
age_counts = df['IDADE'].value_counts()
valid_ages = age_counts[age_counts >= 10].index
df = df[df['IDADE'].isin(valid_ages)]

df_clean = df.dropna(subset=list(set(CANDIDATE_FEATURES + ['PESO', 'IDADE'])))
print(f"Records after cleaning: {len(df_clean)}")

# Split by animal (same animal never in both train and test).
animals = df_clean['ANIMAL'].unique()
animals_train, animals_test = train_test_split(animals, test_size=0.2, random_state=42)
train = df_clean[df_clean['ANIMAL'].isin(animals_train)]
test = df_clean[df_clean['ANIMAL'].isin(animals_test)]

X_train_all, y_train = train[CANDIDATE_FEATURES], train['PESO']
X_test_all, y_test = test[CANDIDATE_FEATURES], test['PESO']

print(f"Train: {len(X_train_all)} rows | Test: {len(X_test_all)} rows")
print(f"Animals: Train={len(animals_train)} | Test={len(animals_test)}")

# XGBoost hyperparameter search space (shared by both stages).
PARAM_DIST = {
    'n_estimators': randint(100, 800),
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.005, 0.15),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 0.5),
    'min_child_weight': randint(1, 15),
    'reg_alpha': uniform(0, 1.0),
    'reg_lambda': uniform(0.5, 2.0),
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def tune_xgb(X, y, n_iter):
    """Run a RandomizedSearchCV over XGBoost and return the best fitted estimator + CV R2."""
    base = XGBRegressor(random_state=42, n_jobs=-1, importance_type='gain')
    search = RandomizedSearchCV(
        estimator=base, param_distributions=PARAM_DIST, n_iter=n_iter,
        cv=cv, scoring='r2', n_jobs=1, random_state=42, verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_score_


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        'r2': r2_score(y, pred),
        'rmse': float(np.sqrt(mean_squared_error(y, pred))),
        'mae': float(mean_absolute_error(y, pred)),
        'pred': pred,
    }


# =============================================================================
# STAGE 1 - FULL-FEATURE MODEL + FEATURE IMPORTANCE
# =============================================================================
print(f"\n{'='*60}\nStage 1: full-feature XGBoost (ranking importances)\n{'='*60}")
model_full, r2cv_full = tune_xgb(X_train_all, y_train, n_iter=40)
eval_full = evaluate(model_full, X_test_all, y_test)
print(f"  Full model -> R2 CV={r2cv_full:.4f} | R2 Test={eval_full['r2']:.4f} "
      f"| RMSE={eval_full['rmse']:.2f}g | MAE={eval_full['mae']:.2f}g")

# Gain-based importance, normalized to sum to 1.
imp = model_full.feature_importances_.astype(float)
imp = imp / imp.sum()
importance = (pd.Series(imp, index=CANDIDATE_FEATURES)
              .sort_values(ascending=False))

print("\nFeature importance (gain, normalized):")
for feat, val in importance.items():
    print(f"  {feat:<12} {val:.4f}")

# Select the smallest set of features whose cumulative importance >= threshold.
cumulative = importance.cumsum()
n_keep = int((cumulative < CUMULATIVE_THRESHOLD).sum()) + 1
n_keep = max(n_keep, 3)  # never keep fewer than 3 features
SELECTED_FEATURES = list(importance.index[:n_keep])
dropped = [f for f in CANDIDATE_FEATURES if f not in SELECTED_FEATURES]

print(f"\nSelected {len(SELECTED_FEATURES)} features "
      f"(cumulative importance >= {CUMULATIVE_THRESHOLD:.0%}): {SELECTED_FEATURES}")
print(f"Dropped {len(dropped)} features: {dropped}")

# =============================================================================
# STAGE 2 - FINAL MODEL ON SELECTED FEATURES
# =============================================================================
print(f"\n{'='*60}\nStage 2: final XGBoost on selected features\n{'='*60}")
X_train_sel = X_train_all[SELECTED_FEATURES]
X_test_sel = X_test_all[SELECTED_FEATURES]

model_sel, r2cv_sel = tune_xgb(X_train_sel, y_train, n_iter=60)
eval_sel = evaluate(model_sel, X_test_sel, y_test)
y_pred = eval_sel['pred']
pct_mae = eval_sel['mae'] / y_test.mean() * 100

print(f"  Selected model -> R2 CV={r2cv_sel:.4f} | R2 Test={eval_sel['r2']:.4f} "
      f"| RMSE={eval_sel['rmse']:.2f}g | MAE={eval_sel['mae']:.2f}g ({pct_mae:.1f}%)")

# Per-age breakdown (the global R2 is inflated by IDADE; per-age tells the real story).
per_age = []
for age in sorted(test['IDADE'].unique()):
    mask = (test['IDADE'] == age).values
    if mask.sum() < 5:
        continue
    per_age.append({
        'age': int(age),
        'r2': r2_score(y_test[mask], y_pred[mask]),
        'mae': float(mean_absolute_error(y_test[mask], y_pred[mask])),
        'n': int(mask.sum()),
    })

print(f"\n{'Age':>5} | {'N':>4} | {'R2':>8} | {'MAE (g)':>8}")
print("-" * 34)
for a in per_age:
    print(f"{a['age']:>5} | {a['n']:>4} | {a['r2']:>8.4f} | {a['mae']:>8.2f}")

# Contrast: the global R2 is inflated by the age->weight curve; the average
# intra-age R2 is the honest measure of within-age predictive power.
mean_per_age_r2 = float(np.mean([a['r2'] for a in per_age]))
print(f"\n  R2 GLOBAL (todas as idades juntas): {eval_sel['r2']:.4f}")
print(f"  R2 MEDIO INTRA-IDADE (media das idades): {mean_per_age_r2:.4f}")

# =============================================================================
# COMPARISON: FULL vs SELECTED
# =============================================================================
print(f"\n{'='*80}\nFULL vs SELECTED FEATURE SET\n{'='*80}")
print(f"{'Model':<22} | {'# feat':>6} | {'R2 CV':>8} | {'R2 Test':>8} | {'RMSE':>8} | {'MAE':>8}")
print("-" * 75)
print(f"{'All features':<22} | {len(CANDIDATE_FEATURES):>6} | {r2cv_full:>8.4f} | "
      f"{eval_full['r2']:>8.4f} | {eval_full['rmse']:>7.2f}g | {eval_full['mae']:>7.2f}g")
print(f"{'Selected features':<22} | {len(SELECTED_FEATURES):>6} | {r2cv_sel:>8.4f} | "
      f"{eval_sel['r2']:>8.4f} | {eval_sel['rmse']:>7.2f}g | {eval_sel['mae']:>7.2f}g")

# =============================================================================
# FIGURE 1 - experimento_5_peso.png (final model performance)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Experimento 5 - Predição de Peso (XGBoost, features selecionadas)',
             fontsize=16, fontweight='bold')

# (A) Real vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.4, s=12, color='steelblue')
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Ideal')
ax.set_xlabel('Peso Real (g)')
ax.set_ylabel('Peso Predito (g)')
ax.set_title(f"Real vs. Predito (R²={eval_sel['r2']:.4f}; RMSE={eval_sel['rmse']:.1f} g; "
             f"MAE={eval_sel['mae']:.1f} g)")
ax.legend()
ax.grid(True, alpha=0.3)

# (B) Residuals vs Predicted
ax = axes[0, 1]
residuals = y_test.values - y_pred
ax.scatter(y_pred, residuals, alpha=0.4, s=12, color='darkorange')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Peso Predito (g)')
ax.set_ylabel('Resíduo (g)')
ax.set_title('Análise de Resíduos')
ax.grid(True, alpha=0.3)

# (C) Per-age R2 (the honest, intra-age metric) - each bar labeled with its value.
ax = axes[1, 0]
ages = [a['age'] for a in per_age]
r2s = [a['r2'] for a in per_age]
FLOOR = -2.0  # clip very negative bars for readability; the true value is labeled
heights = [max(v, FLOOR) for v in r2s]
colors = ['green' if v >= 0.5 else 'orange' if v >= 0 else 'red' for v in r2s]
bars = ax.bar([str(a) for a in ages], heights, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axhline(y=mean_per_age_r2, color='purple', linestyle='--', linewidth=1.2,
           label=f'Média intra-idade = {mean_per_age_r2:.2f}')
for b, v in zip(bars, r2s):
    top = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, top + (0.06 if v >= 0 else -0.06),
            f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top',
            fontsize=9, fontweight='bold')
ax.set_ylim(FLOOR - 0.35, 1.0)
ax.set_xlabel('Idade (dias)')
ax.set_ylabel('R²')
ax.set_title(f"R² por Faixa Etária (intra-idade) — global = {eval_sel['r2']:.3f}\n"
             f"(barras cortadas em {FLOOR:.1f}; valor real rotulado)")
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

# (D) Per-age MAE
ax = axes[1, 1]
maes = [a['mae'] for a in per_age]
ax.bar([str(a) for a in ages], maes, color='coral', alpha=0.75)
ax.set_xlabel('Idade (dias)')
ax.set_ylabel('MAE (g)')
ax.set_title('Erro Médio Absoluto por Faixa Etária')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig1_path = FIGURES / 'experimento_5_peso.png'
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved: results/figures/{fig1_path.name}")

# =============================================================================
# FIGURE 2 - experimento_5_features.png (feature selection)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Experimento 5 - Seleção de Features Mais Impactantes',
             fontsize=16, fontweight='bold')

# (A) Importance ranking with selected highlighted
ax = axes[0]
feats = list(importance.index)[::-1]  # ascending for horizontal bars
vals = [importance[f] for f in feats]
bar_colors = ['seagreen' if f in SELECTED_FEATURES else 'lightgray' for f in feats]
ax.barh(feats, vals, color=bar_colors, alpha=0.9)
ax.set_xlabel('Importância (gain, normalizada)')
ax.set_title(f'Importância das Variáveis — selecionadas em verde '
             f'(soma ≥ {CUMULATIVE_THRESHOLD:.0%})')
for i, (f, v) in enumerate(zip(feats, vals)):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

# (B) Full vs Selected performance
ax = axes[1]
groups = ['Todas as\nfeatures', 'Features\nselecionadas']
r2_vals = [eval_full['r2'], eval_sel['r2']]
mae_vals = [eval_full['mae'], eval_sel['mae']]
x = np.arange(len(groups))
ax2 = ax.twinx()
b1 = ax.bar(x - 0.2, r2_vals, 0.4, label='R² Teste', color='steelblue', alpha=0.85)
b2 = ax2.bar(x + 0.2, mae_vals, 0.4, label='MAE (g)', color='coral', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('R² Teste')
ax2.set_ylabel('MAE (g)')
ax.set_ylim(0, 1)
ax.set_title('Desempenho: todas vs. selecionadas')
for xi, v in zip(x - 0.2, r2_vals):
    ax.text(xi, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
for xi, v in zip(x + 0.2, mae_vals):
    ax2.text(xi, v + 1, f'{v:.1f}', ha='center', fontsize=9)
lines = [b1, b2]
ax.legend(lines, [l.get_label() for l in lines], loc='upper center')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig2_path = FIGURES / 'experimento_5_features.png'
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: results/figures/{fig2_path.name}")

# =============================================================================
# PERSIST MODEL + RESULTS
# =============================================================================
model_sel.save_model(MODEL_DIR / 'modelo_peso_exp5.json')

result = {
    'experiment': 'experimento_5_peso',
    'approach': 'XGBoost on full dataset (no per-age split), top features by cumulative gain importance',
    'include_idade': INCLUDE_IDADE,
    'cumulative_threshold': CUMULATIVE_THRESHOLD,
    'candidate_features': CANDIDATE_FEATURES,
    'selected_features': SELECTED_FEATURES,
    'dropped_features': dropped,
    'feature_importance': {f: round(float(importance[f]), 4) for f in importance.index},
    'final_model': {
        'r2_cv': round(float(r2cv_sel), 4),
        'r2_test': round(float(eval_sel['r2']), 4),
        'rmse': round(eval_sel['rmse'], 2),
        'mae': round(eval_sel['mae'], 2),
        'pct_mae': round(float(pct_mae), 2),
    },
    'full_model': {
        'r2_cv': round(float(r2cv_full), 4),
        'r2_test': round(float(eval_full['r2']), 4),
        'rmse': round(eval_full['rmse'], 2),
        'mae': round(eval_full['mae'], 2),
    },
    'per_age': per_age,
    'mean_per_age_r2': round(mean_per_age_r2, 4),
    'figures': [fig1_path.name, fig2_path.name],
}
with open(RESULTS / 'experimento_5_peso.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*80}")
print(f"DONE — selected {len(SELECTED_FEATURES)}/{len(CANDIDATE_FEATURES)} features: {SELECTED_FEATURES}")
print(f"Final model: R2 Test={eval_sel['r2']:.4f} | MAE={eval_sel['mae']:.2f}g ({pct_mae:.1f}%)")
print(f"Results: results/experimento_5_peso.json")
print(f"{'='*80}")
