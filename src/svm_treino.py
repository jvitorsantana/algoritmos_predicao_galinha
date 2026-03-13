import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from scipy.stats import loguniform
import joblib

ROOT = Path(__file__).resolve().parent.parent

SVM_DATA_DIR = ROOT / 'data' / 'svm'
SVM_MODEL_DIR = ROOT / 'results' / 'models' / 'svm'

IDADES = sorted([
    int(f.replace('dataset_idade_', '').replace('.csv', ''))
    for f in os.listdir(SVM_DATA_DIR)
    if f.startswith('dataset_idade_') and f.endswith('.csv')
    and f.replace('dataset_idade_', '').replace('.csv', '').isdigit()
    and 0 <= int(f.replace('dataset_idade_', '').replace('.csv', '')) <= 115
])
print(f"Datasets encontrados: {IDADES}")

FEATURES = ['PESO', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA',
            'DORSO', 'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']

SVM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_ROUNDS = 15
N_ITER_PER_ROUND = 15

param_distributions = {
    'svm__C': loguniform(1e-3, 1e3),
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svm__gamma': loguniform(1e-5, 1e1),
    'svm__degree': [2, 3, 4, 5]
}

resultados = []

for idade in IDADES:
    arquivo = SVM_DATA_DIR / f'dataset_idade_{idade}.csv'

    if not arquivo.exists():
        print(f"\nArquivo {arquivo} nao encontrado, pulando...")
        continue

    print(f"\n{'='*80}")
    print(f"IDADE {idade}")
    print(f"{'='*80}")

    dados = pd.read_csv(arquivo, sep=';', encoding='utf-8')
    print(f"Registros: {len(dados)}")

    dados['SEXO_MAPPED'] = dados['SEXO'].map({'Macho': 1, 'Fêmea': 0})

    features_disponiveis = [col for col in FEATURES if col in dados.columns]
    print(f"Features: {features_disponiveis}")

    for col in features_disponiveis:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')

    dados_clean = dados.dropna(subset=features_disponiveis + ['SEXO_MAPPED'])
    n_removidas = len(dados) - len(dados_clean)
    if n_removidas > 0:
        print(f"ATENCAO: {n_removidas} linhas removidas por valores faltantes")
    print(f"Registros para treinamento: {len(dados_clean)}")

    if len(dados_clean) < 10:
        print("Poucos registros para treinar, pulando...")
        continue

    X = dados_clean[features_disponiveis]
    y = dados_clean['SEXO_MAPPED'].astype(int)

    dist = y.value_counts().to_dict()
    print(f"Distribuicao: Femea(0)={dist.get(0, 0)}, Macho(1)={dist.get(1, 0)}")

    if y.nunique() < 2:
        print("Apenas uma classe presente no dataset, pulando...")
        continue

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42, class_weight='balanced'))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Treinando ({N_ROUNDS} rodadas x {N_ITER_PER_ROUND} iteracoes)...")

    best_models_per_round = []

    for round_idx in range(N_ROUNDS):
        print(f"  Rodada {round_idx + 1}/{N_ROUNDS}...", end=' ')

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=N_ITER_PER_ROUND,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            random_state=42 + round_idx,
            return_train_score=True,
            verbose=0
        )

        random_search.fit(X, y)

        best_models_per_round.append({
            'round': round_idx + 1,
            'best_cv_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'cv_results': random_search.cv_results_
        })

        print(f"F1 CV = {random_search.best_score_:.4f}")

    overall_best = max(best_models_per_round, key=lambda x: x['best_cv_score'])
    best_pipeline = overall_best['best_estimator']

    print(f"\n  Melhor rodada: {overall_best['round']}")
    print(f"  Melhor F1 CV: {overall_best['best_cv_score']:.4f}")
    print(f"  Parametros: {overall_best['best_params']}")

    y_pred = best_pipeline.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"\n  Metricas de treino:")
    print(f"  Acuracia:  {acc:.4f}")
    print(f"  Precisao:  {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print(f"\n  Matriz de Confusao (treino):")
    cm = confusion_matrix(y, y_pred)
    print(f"  {cm}")

    print(f"\n  Relatorio de Classificacao (treino):")
    print(classification_report(y, y_pred, target_names=['Femea', 'Macho'],
                                 zero_division=0))

    modelo_path = SVM_MODEL_DIR / f'modelo_svm_idade_{idade}.joblib'
    joblib.dump({
        'model': best_pipeline,
        'features': features_disponiveis,
        'params': overall_best['best_params'],
        'cv_score': overall_best['best_cv_score'],
        'rounds': [{
            'round': r['round'],
            'best_cv_score': r['best_cv_score'],
            'cv_results': r['cv_results']
        } for r in best_models_per_round],
        'best_round': overall_best['round']
    }, modelo_path)
    print(f"  Modelo salvo em: {modelo_path}")

    resultados.append({
        'idade': idade,
        'n_registros': len(dados_clean),
        'f1_cv': overall_best['best_cv_score'],
        'f1_treino': f1,
        'acc_treino': acc,
        'kernel': overall_best['best_params']['svm__kernel'],
        'C': overall_best['best_params']['svm__C']
    })

print(f"\n{'='*80}")
print("RESUMO DO TREINAMENTO")
print(f"{'='*80}")
print(f"{'Idade':>6} | {'N':>6} | {'F1 CV':>8} | {'F1 Treino':>10} | {'Acc':>8} | {'Kernel':>8}")
print("-"*65)
for r in resultados:
    print(f"{r['idade']:6d} | {r['n_registros']:6d} | {r['f1_cv']:8.4f} | "
          f"{r['f1_treino']:10.4f} | {r['acc_treino']:8.4f} | {r['kernel']:>8}")

print(f"\n{'='*80}")
print("TREINAMENTO CONCLUIDO!")
print(f"Total de modelos treinados: {len(resultados)}")
print(f"Modelos salvos na pasta 'modelos_svm/'")
print(f"{'='*80}")
