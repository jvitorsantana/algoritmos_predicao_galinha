import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform
import pickle

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / 'results' / 'models' / 'xgb_sexo'
FIGURES_DIR = ROOT / 'results' / 'figures'

plt.rcParams['figure.figsize'] = [16, 10]

# =============================================================================
# CONFIGURACAO
# =============================================================================
print("="*80)
print("CLASSIFICACAO DE SEXO - TREINAMENTO")
print("Usando medidas biometricas para classificar o sexo da ave")
print("="*80)

# Pasta para salvar resultados
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Carregar dataset de TREINO
dados = pd.read_csv(ROOT / 'data' / 'processed' / 'sexo_treino.csv', sep=';', encoding='utf-8')

print(f"\nTotal de registros (treino): {len(dados)}")
print(f"Animais unicos: {dados['ANIMAL'].nunique()}")

# =============================================================================
# PREPARACAO DOS DADOS
# =============================================================================
print("\n" + "="*80)
print("PREPARACAO DOS DADOS")
print("="*80)

# Garantir que colunas numericas estao no tipo correto
colunas_numericas = ['PESO', 'IDADE', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA',
                     'DORSO', 'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']
for col in colunas_numericas:
    if col in dados.columns:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')

print("\nTipos das colunas apos conversao:")
print(dados[colunas_numericas].dtypes)

# Verificar coluna SEXO
if 'SEXO' not in dados.columns:
    print("ERRO: Coluna 'SEXO' nao encontrada no dataset!")
    print(f"Colunas disponiveis: {list(dados.columns)}")
    exit(1)

print("\nDistribuicao de SEXO no dataset de treino:")
print(dados['SEXO'].value_counts())

# =============================================================================
# FEATURES
# =============================================================================
colunas_features = ['PESO', 'IDADE', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA',
                    'DORSO', 'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']

colunas_disponiveis = [col for col in colunas_features if col in dados.columns]
print(f"\nFeatures utilizadas: {colunas_disponiveis}")

dados_clean = dados.dropna(subset=colunas_disponiveis + ['SEXO'])
n_removidas = len(dados) - len(dados_clean)
print(f"Registros apos limpeza: {len(dados_clean)} ({n_removidas} removidos)")

X_train = dados_clean[colunas_disponiveis]

# Codificar o target (SEXO)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(dados_clean['SEXO'])

print(f"\nClasses codificadas:")
for i, classe in enumerate(label_encoder.classes_):
    n_amostras = (y_train == i).sum()
    print(f"  {classe} -> {i} ({n_amostras} amostras)")

# Calcular scale_pos_weight para lidar com desbalanceamento
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"\nDesbalanceamento detectado: scale_pos_weight = {scale_pos_weight:.4f}")

# =============================================================================
# CONFIGURACAO DO MODELO
# =============================================================================
N_ROUNDS = 15
N_ITER_PER_ROUND = 20

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(2, 7),
    'learning_rate': uniform(0.005, 0.15),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 0.5),
    'min_child_weight': randint(1, 15),
    'reg_alpha': uniform(0, 1.0),
    'reg_lambda': uniform(0.5, 2.0)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =============================================================================
# TREINAMENTO
# =============================================================================
print(f"\n{'='*80}")
print(f"EXECUTANDO {N_ROUNDS} RODADAS DE RANDOMIZEDSEARCHCV")
print(f"{'='*80}\n")

best_models_per_round = []

for round_idx in range(N_ROUNDS):
    print(f"RODADA {round_idx + 1}/{N_ROUNDS}...", end=' ')

    xgb = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight  # Compensar desbalanceamento
    )

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=N_ITER_PER_ROUND,
        cv=cv,
        scoring='f1',
        n_jobs=1,
        random_state=42 + round_idx,
        return_train_score=True,
        verbose=0
    )

    random_search.fit(X_train, y_train)

    best_models_per_round.append({
        'round': round_idx + 1,
        'best_params': random_search.best_params_,
        'best_cv_score': random_search.best_score_,
        'best_estimator': random_search.best_estimator_
    })

    print(f"Melhor F1: {random_search.best_score_:.4f}")

# =============================================================================
# SELECIONAR MELHOR MODELO
# =============================================================================
overall_best = max(best_models_per_round, key=lambda x: x['best_cv_score'])

print(f"\n{'='*80}")
print("MELHOR CONFIGURACAO GERAL")
print(f"{'='*80}")
print(f"Da Rodada: {overall_best['round']}")
print(f"Melhor F1 CV: {overall_best['best_cv_score']:.4f}")
print(f"\nParametros:")
for k, v in overall_best['best_params'].items():
    print(f"  {k}: {v}")

best_xgb = overall_best['best_estimator']

# =============================================================================
# AVALIACAO NO CONJUNTO DE TREINO
# =============================================================================
y_train_pred = best_xgb.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
train_prec = precision_score(y_train, y_train_pred)
train_rec = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print(f"\n{'='*80}")
print("METRICAS DO CONJUNTO DE TREINO:")
print(f"Accuracy:  {train_acc:.4f}")
print(f"Precision: {train_prec:.4f}")
print(f"Recall:    {train_rec:.4f}")
print(f"F1-Score:  {train_f1:.4f}")

# Matriz de confusao no treino
cm_train = confusion_matrix(y_train, y_train_pred)
print(f"\nMatriz de Confusao (TREINO):")
print(f"             Predito")
print(f"             {label_encoder.classes_[0]:>8}  {label_encoder.classes_[1]:>8}")
print(f"Real {label_encoder.classes_[0]:>6}   {cm_train[0,0]:>8}  {cm_train[0,1]:>8}")
print(f"     {label_encoder.classes_[1]:>6}   {cm_train[1,0]:>8}  {cm_train[1,1]:>8}")

print(f"\nRelatorio de Classificacao (TREINO):")
print(classification_report(y_train, y_train_pred, target_names=label_encoder.classes_))

# =============================================================================
# IMPORTANCIA DAS FEATURES
# =============================================================================
print(f"\n{'='*80}")
print("IMPORTANCIA DAS FEATURES:")
print(f"{'='*80}")

feature_importance = best_xgb.feature_importances_
feature_names = X_train.columns
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: -x[1]):
    print(f"  {name}: {importance:.4f}")

# =============================================================================
# ACURACIA POR FAIXA DE IDADE (TREINO)
# =============================================================================
print(f"\n{'='*80}")
print("ACURACIA POR FAIXA DE IDADE (TREINO):")
print(f"{'='*80}")

train_data = X_train.copy()
train_data['SEXO_REAL'] = y_train
train_data['SEXO_PRED'] = y_train_pred
train_data['ACERTO'] = (train_data['SEXO_REAL'] == train_data['SEXO_PRED']).astype(int)

for idade in sorted(train_data['IDADE'].unique()):
    subset = train_data[train_data['IDADE'] == idade]
    if len(subset) > 0:
        acc_idade = subset['ACERTO'].mean()
        print(f"  Idade {int(idade):3d}: Accuracy = {acc_idade:.4f} (n={len(subset)})")

# =============================================================================
# GRAFICOS DO TREINO
# =============================================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Classificacao de Sexo - XGBoost (TREINO)', fontsize=16, fontweight='bold')

# Grafico 1: Matriz de Confusao (Treino)
plt.subplot(2, 2, 1)
plt.imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusao (Treino)')
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_)
plt.yticks(tick_marks, label_encoder.classes_)
thresh = cm_train.max() / 2.
for i in range(cm_train.shape[0]):
    for j in range(cm_train.shape[1]):
        plt.text(j, i, format(cm_train[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm_train[i, j] > thresh else "black")
plt.ylabel('Real')
plt.xlabel('Predito')

# Grafico 2: Importancia das Features
plt.subplot(2, 2, 2)
indices = np.argsort(feature_importance)
plt.barh(range(len(indices)), feature_importance[indices], color='orange', alpha=0.6)
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importancia')
plt.title('Importancia das Features')
plt.grid(True, alpha=0.3, axis='x')

# Grafico 3: Acuracia por Idade (Treino)
plt.subplot(2, 2, 3)
idades = sorted(train_data['IDADE'].unique())
accs = [train_data[train_data['IDADE'] == idade]['ACERTO'].mean() for idade in idades]
plt.bar(idades, accs, color='green', alpha=0.6)
plt.xlabel('Idade (dias)')
plt.ylabel('Accuracy')
plt.title('Acuracia por Idade (Treino)')
plt.ylim(0, 1.1)
plt.axhline(y=train_acc, color='r', linestyle='--', label=f'Media: {train_acc:.4f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Grafico 4: F1 por Rodada
plt.subplot(2, 2, 4)
rodadas = [m['round'] for m in best_models_per_round]
f1_scores = [m['best_cv_score'] for m in best_models_per_round]
plt.plot(rodadas, f1_scores, marker='o', color='purple', linewidth=2, markersize=8)
plt.axhline(y=overall_best['best_cv_score'], color='r', linestyle='--',
            label=f'Melhor: {overall_best["best_cv_score"]:.4f}')
plt.xlabel('Rodada')
plt.ylabel('Melhor F1 (CV)')
plt.title('Evolucao do F1 por Rodada')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'grafico_treino.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nGrafico salvo em: {FIGURES_DIR / 'grafico_treino.png'}")

# =============================================================================
# SALVAR MODELO E ENCODER
# =============================================================================
with open(MODEL_DIR / 'modelo_xgb.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open(MODEL_DIR / 'features.pkl', 'wb') as f:
    pickle.dump(colunas_disponiveis, f)

print(f"\n{'='*80}")
print("TREINAMENTO CONCLUIDO!")
print(f"Arquivos salvos em: {MODEL_DIR}")
print("  - modelo_xgb.pkl       (modelo treinado)")
print("  - label_encoder.pkl    (encoder das classes)")
print("  - features.pkl         (lista de features usadas)")
print(f"Grafico salvo em: {FIGURES_DIR / 'grafico_treino.png'}")
print(f"{'='*80}")
