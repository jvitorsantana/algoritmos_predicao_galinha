import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle

plt.rcParams['figure.figsize'] = [16, 10]

# =============================================================================
# CONFIGURACAO
# =============================================================================
print("="*80)
print("CLASSIFICACAO DE SEXO - TESTE")
print("Avaliando o modelo treinado no conjunto de teste")
print("="*80)

# Pasta para salvar resultados
os.makedirs('resultados_teste', exist_ok=True)

# =============================================================================
# CARREGAR MODELO E ARTEFATOS SALVOS
# =============================================================================
print("\nCarregando modelo e artefatos do treino...")

try:
    with open('resultados_treino/modelo_xgb.pkl', 'rb') as f:
        best_xgb = pickle.load(f)

    with open('resultados_treino/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('resultados_treino/features.pkl', 'rb') as f:
        colunas_disponiveis = pickle.load(f)

    print("Modelo carregado com sucesso!")
    print(f"Features utilizadas: {colunas_disponiveis}")

except FileNotFoundError as e:
    print(f"ERRO: {e}")
    print("Execute primeiro o script de TREINO (xgb_treino.py) para gerar o modelo.")
    exit(1)

# =============================================================================
# CARREGAR DATASET DE TESTE
# =============================================================================
dados_teste = pd.read_csv('teste.csv', sep=';', encoding='utf-8')

print(f"\nTotal de registros (teste): {len(dados_teste)}")
print(f"Animais unicos no teste: {dados_teste['ANIMAL'].nunique()}")

# Garantir que colunas numericas estao no tipo correto
for col in colunas_disponiveis:
    if col in dados_teste.columns:
        dados_teste[col] = pd.to_numeric(dados_teste[col], errors='coerce')

# Verificar coluna SEXO
if 'SEXO' not in dados_teste.columns:
    print("ERRO: Coluna 'SEXO' nao encontrada no dataset de teste!")
    exit(1)

print("\nDistribuicao de SEXO no dataset de teste:")
print(dados_teste['SEXO'].value_counts())

# =============================================================================
# PREPARACAO DOS DADOS DE TESTE
# =============================================================================
dados_clean = dados_teste.dropna(subset=colunas_disponiveis + ['SEXO'])
n_removidas = len(dados_teste) - len(dados_clean)
print(f"\nRegistros apos limpeza: {len(dados_clean)} ({n_removidas} removidos)")

X_test = dados_clean[colunas_disponiveis]
y_test = label_encoder.transform(dados_clean['SEXO'])

print(f"\nClasses no teste:")
for i, classe in enumerate(label_encoder.classes_):
    n_amostras = (y_test == i).sum()
    print(f"  {classe} -> {i} ({n_amostras} amostras)")

# =============================================================================
# PREDICAO
# =============================================================================
y_test_pred = best_xgb.predict(X_test)
y_test_proba = best_xgb.predict_proba(X_test)[:, 1]

# =============================================================================
# METRICAS DE TESTE
# =============================================================================
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\n{'='*80}")
print("METRICAS DO CONJUNTO DE TESTE:")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

# Matriz de confusao
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n{'='*80}")
print("MATRIZ DE CONFUSAO (TESTE):")
print(f"{'='*80}")
print(f"\n             Predito")
print(f"             {label_encoder.classes_[0]:>8}  {label_encoder.classes_[1]:>8}")
print(f"Real {label_encoder.classes_[0]:>6}   {cm[0,0]:>8}  {cm[0,1]:>8}")
print(f"     {label_encoder.classes_[1]:>6}   {cm[1,0]:>8}  {cm[1,1]:>8}")

# Relatorio de classificacao
print(f"\n{'='*80}")
print("RELATORIO DE CLASSIFICACAO:")
print(f"{'='*80}")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# =============================================================================
# ACURACIA POR FAIXA DE IDADE
# =============================================================================
print(f"\n{'='*80}")
print("ACURACIA POR FAIXA DE IDADE (TESTE):")
print(f"{'='*80}")

test_data = X_test.copy()
test_data['SEXO_REAL'] = y_test
test_data['SEXO_PRED'] = y_test_pred
test_data['ACERTO'] = (test_data['SEXO_REAL'] == test_data['SEXO_PRED']).astype(int)

for idade in sorted(test_data['IDADE'].unique()):
    subset = test_data[test_data['IDADE'] == idade]
    if len(subset) > 0:
        acc_idade = subset['ACERTO'].mean()
        print(f"  Idade {int(idade):3d}: Accuracy = {acc_idade:.4f} (n={len(subset)})")

# =============================================================================
# GRAFICOS DO TESTE
# =============================================================================
feature_importance = best_xgb.feature_importances_
feature_names = X_test.columns

fig = plt.figure(figsize=(16, 10))
fig.suptitle('Classificacao de Sexo - XGBoost (TESTE)', fontsize=16, fontweight='bold')

# Grafico 1: Matriz de Confusao
plt.subplot(2, 2, 1)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusao (Teste)')
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_)
plt.yticks(tick_marks, label_encoder.classes_)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
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

# Grafico 3: Acuracia por Idade
plt.subplot(2, 2, 3)
idades = sorted(test_data['IDADE'].unique())
accs = [test_data[test_data['IDADE'] == idade]['ACERTO'].mean() for idade in idades]
plt.bar(idades, accs, color='green', alpha=0.6)
plt.xlabel('Idade (dias)')
plt.ylabel('Accuracy')
plt.title('Acuracia por Idade (Teste)')
plt.ylim(0, 1.1)
plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Media: {test_acc:.4f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Grafico 4: Distribuicao das Probabilidades por Classe
plt.subplot(2, 2, 4)
plt.hist(y_test_proba[y_test == 0], bins=30, alpha=0.5, label=label_encoder.classes_[0], color='blue')
plt.hist(y_test_proba[y_test == 1], bins=30, alpha=0.5, label=label_encoder.classes_[1], color='red')
plt.xlabel('Probabilidade Predita (Classe 1)')
plt.ylabel('Frequencia')
plt.title('Distribuicao das Probabilidades por Classe')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados_teste/grafico_teste.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafico salvo em: resultados_teste/grafico_teste.png")

# =============================================================================
# SALVAR PREDICOES
# =============================================================================
resultado = dados_clean[['ANIMAL', 'SEXO', 'IDADE']].copy()
resultado['SEXO_PRED'] = label_encoder.inverse_transform(y_test_pred)
resultado['PROBABILIDADE'] = y_test_proba
resultado['ACERTO'] = (y_test == y_test_pred)
resultado.to_csv('resultados_teste/predicoes_teste.csv', sep=';', index=False)

print(f"\n{'='*80}")
print("TESTE CONCLUIDO!")
print("Arquivos salvos em: resultados_teste/")
print("  - grafico_teste.png       (graficos de avaliacao)")
print("  - predicoes_teste.csv     (predicoes detalhadas por animal e idade)")
print(f"{'='*80}")
