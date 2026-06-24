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
    train_test_split, RandomizedSearchCV, KFold, StratifiedKFold
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_curve, auc, roc_auc_score, accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import randint, uniform

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPERIMENT 6 - Feature importance (weight & sex) + ROC/AUC (sex)")
print("=" * 80)

# =============================================================================
# DATA PREPARATION
# =============================================================================
df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')

MORPHO = ['BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA', 'DORSO', 'VENTRE',
          'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']

for col in MORPHO + ['PESO', 'IDADE']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop ages with too few samples.
age_counts = df['IDADE'].value_counts()
valid_ages = age_counts[age_counts >= 10].index
df = df[df['IDADE'].isin(valid_ages)]

# One animal-level split shared by both tasks (no animal in train and test).
animals = df['ANIMAL'].dropna().unique()
animals_train, animals_test = train_test_split(animals, test_size=0.2, random_state=42)
print(f"Animals: train={len(animals_train)} | test={len(animals_test)}")

# XGBoost search spaces.
REG_PARAM = {
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
CLF_PARAM = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(2, 7),
    'learning_rate': uniform(0.005, 0.15),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 0.5),
    'min_child_weight': randint(1, 15),
    'reg_alpha': uniform(0, 1.0),
    'reg_lambda': uniform(0.5, 2.0),
}


def normalized_importance(model, feature_names):
    imp = model.feature_importances_.astype(float)
    total = imp.sum()
    imp = imp / total if total > 0 else imp
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


# =============================================================================
# TASK 1 - WEIGHT REGRESSION (feature importance)
# =============================================================================
print(f"\n{'='*60}\nTask 1: weight regression (XGBoost)\n{'='*60}")
REG_FEATURES = ['IDADE'] + MORPHO  # PESO is the target

reg = df.dropna(subset=REG_FEATURES + ['PESO'])
reg_train = reg[reg['ANIMAL'].isin(animals_train)]
reg_test = reg[reg['ANIMAL'].isin(animals_test)]
X_reg_tr, y_reg_tr = reg_train[REG_FEATURES], reg_train['PESO']
X_reg_te, y_reg_te = reg_test[REG_FEATURES], reg_test['PESO']

reg_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1, importance_type='gain'),
    REG_PARAM, n_iter=40, cv=KFold(5, shuffle=True, random_state=42),
    scoring='r2', n_jobs=1, random_state=42, verbose=0,
)
reg_search.fit(X_reg_tr, y_reg_tr)
reg_model = reg_search.best_estimator_
reg_pred = reg_model.predict(X_reg_te)
reg_r2 = r2_score(y_reg_te, reg_pred)
reg_rmse = float(np.sqrt(mean_squared_error(y_reg_te, reg_pred)))
reg_mae = float(mean_absolute_error(y_reg_te, reg_pred))
reg_imp = normalized_importance(reg_model, REG_FEATURES)

print(f"  R2 Test={reg_r2:.4f} | RMSE={reg_rmse:.2f}g | MAE={reg_mae:.2f}g")
print("  Importance (gain):")
for f, v in reg_imp.items():
    print(f"    {f:<12} {v:.4f}")

# =============================================================================
# TASK 2 - SEX CLASSIFICATION (feature importance + ROC/AUC)
# =============================================================================
print(f"\n{'='*60}\nTask 2: sex classification (XGBoost)\n{'='*60}")
CLF_FEATURES = ['PESO', 'IDADE'] + MORPHO

clf = df.dropna(subset=CLF_FEATURES + ['SEXO'])
clf = clf[clf['SEXO'].isin(['Macho', 'Femea'])]
clf_train = clf[clf['ANIMAL'].isin(animals_train)]
clf_test = clf[clf['ANIMAL'].isin(animals_test)]

le = LabelEncoder()
y_clf_tr = le.fit_transform(clf_train['SEXO'])
y_clf_te = le.transform(clf_test['SEXO'])
X_clf_tr = clf_train[CLF_FEATURES]
X_clf_te = clf_test[CLF_FEATURES]
pos_label = le.classes_[1]  # positive class for ROC (alphabetical: Femea=0, Macho=1)

n_neg, n_pos = (y_clf_tr == 0).sum(), (y_clf_tr == 1).sum()
scale_pos_weight = n_neg / n_pos if n_pos else 1.0
print(f"  Classes: {dict(zip(le.classes_, range(len(le.classes_))))} | positive='{pos_label}'")
print(f"  scale_pos_weight={scale_pos_weight:.4f}")

clf_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss',
                  scale_pos_weight=scale_pos_weight, importance_type='gain'),
    CLF_PARAM, n_iter=40, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=1, random_state=42, verbose=0,
)
clf_search.fit(X_clf_tr, y_clf_tr)
clf_model = clf_search.best_estimator_

y_proba = clf_model.predict_proba(X_clf_te)[:, 1]
y_pred = clf_model.predict(X_clf_te)
fpr, tpr, _ = roc_curve(y_clf_te, y_proba)
auc_test = auc(fpr, tpr)
auc_cv = clf_search.best_score_
clf_acc = accuracy_score(y_clf_te, y_pred)
clf_f1 = f1_score(y_clf_te, y_pred)
clf_imp = normalized_importance(clf_model, CLF_FEATURES)

print(f"  AUC CV={auc_cv:.4f} | AUC Test={auc_test:.4f} | Acc={clf_acc:.4f} | F1={clf_f1:.4f}")
print("  Importance (gain):")
for f, v in clf_imp.items():
    print(f"    {f:<12} {v:.4f}")

# =============================================================================
# FIGURE 1 - experimento_6_importancia_peso.png
# =============================================================================
def plot_importance(series, title, subtitle, color, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    feats = list(series.index)[::-1]
    vals = [series[f] for f in feats]
    ax.barh(feats, vals, color=color, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Importância (gain, normalizada)')
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: results/figures/{path.name}")


plot_importance(
    reg_imp,
    'Importância das Variáveis (Regressão do Peso)',
    f'XGBoost Regressor — R²={reg_r2:.3f}; MAE={reg_mae:.1f} g',
    'seagreen',
    FIGURES / 'experimento_6_importancia_peso.png',
)

# =============================================================================
# FIGURE 2 - experimento_6_importancia_sexo.png
# =============================================================================
plot_importance(
    clf_imp,
    'Importância das Variáveis (Classificação do Sexo)',
    f'XGBoost Classifier — AUC={auc_test:.3f}; Acc={clf_acc:.3f}',
    'steelblue',
    FIGURES / 'experimento_6_importancia_sexo.png',
)

# =============================================================================
# FIGURE 3 - experimento_6_roc_sexo.png
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='darkorange', linewidth=2.5,
        label=f'XGBoost (AUC = {auc_test:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
        label='Aleatório (AUC = 0.500)')
ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
ax.set_title(f'Curva ROC (Classificação do Sexo)\n'
             f'classe positiva = "{pos_label}" | AUC Teste = {auc_test:.3f} | AUC CV = {auc_cv:.3f}',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = FIGURES / 'experimento_6_roc_sexo.png'
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: results/figures/{roc_path.name}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
result = {
    'experiment': 'experimento_6',
    'regression': {
        'model': 'XGBoost Regressor',
        'features': REG_FEATURES,
        'r2_test': round(float(reg_r2), 4),
        'rmse': round(reg_rmse, 2),
        'mae': round(reg_mae, 2),
        'importance': {f: round(float(reg_imp[f]), 4) for f in reg_imp.index},
    },
    'classification': {
        'model': 'XGBoost Classifier',
        'features': CLF_FEATURES,
        'positive_class': str(pos_label),
        'auc_cv': round(float(auc_cv), 4),
        'auc_test': round(float(auc_test), 4),
        'accuracy': round(float(clf_acc), 4),
        'f1': round(float(clf_f1), 4),
        'importance': {f: round(float(clf_imp[f]), 4) for f in clf_imp.index},
    },
    'figures': [
        'experimento_6_importancia_peso.png',
        'experimento_6_importancia_sexo.png',
        'experimento_6_roc_sexo.png',
    ],
}
with open(RESULTS / 'experimento_6.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*80}")
print(f"DONE — Weight: top feature = {reg_imp.index[0]} ({reg_imp.iloc[0]:.3f}) | R²={reg_r2:.3f}")
print(f"       Sex:    top feature = {clf_imp.index[0]} ({clf_imp.iloc[0]:.3f}) | AUC={auc_test:.3f}")
print(f"Results: results/experimento_6.json")
print(f"{'='*80}")
