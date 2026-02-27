import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import joblib

print("="*80)
print("TREINAMENTO - PREDICAO DE PESO POR IDADE (XGBoost)")
print("Um modelo por idade, lendo cada dataset separadamente")
print("="*80)

IDADES = [0, 7, 14, 21, 28, 52, 66, 80, 101, 115]

FEATURES_MORFOMETRICAS = ['CIRCFABDOM', 'DORSO', 'CANELA', 'ASA', 'COXA',
                          'BICO', 'CIRCFCABECA', 'PESCOCO', 'SOBRECOXA']

os.makedirs('modelos', exist_ok=True)

N_ROUNDS = 10
N_ITER_PER_ROUND = 10

param_distributions = {
    'n_estimators': randint(30, 100),
    'max_depth': randint(2, 5),
    'learning_rate': uniform(0.01, 0.1),
    'subsample': uniform(0.6, 0.3),
    'colsample_bytree': uniform(0.6, 0.3),
    'gamma': uniform(0.5, 2.0),
    'min_child_weight': randint(5, 15),
    'reg_alpha': uniform(0.5, 2.0),
    'reg_lambda': uniform(1.0, 3.0)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

resultados = []

for idade in IDADES:
    arquivo = f'datasets/dataset_idade_{idade}.csv'

    if not os.path.exists(arquivo):
        print(f"\nArquivo {arquivo} nao encontrado, pulando...")
        continue

    print(f"\n{'='*80}")
    print(f"IDADE {idade}")
    print(f"{'='*80}")

    dados = pd.read_csv(arquivo, sep=';', encoding='utf-8')
    print(f"Registros: {len(dados)}")

    features = FEATURES_MORFOMETRICAS.copy()

    if idade > 0 and 'PESO_ANTERIOR' in dados.columns:
        dados['PESO_ANTERIOR'] = pd.to_numeric(dados['PESO_ANTERIOR'], errors='coerce')
        if dados['PESO_ANTERIOR'].notna().sum() > 0:
            features = ['PESO_ANTERIOR'] + features

    features_disponiveis = [col for col in features if col in dados.columns]
    print(f"Features: {features_disponiveis}")

    for col in features_disponiveis + ['PESO']:
        dados[col] = pd.to_numeric(dados[col], errors='coerce')

    dados_clean = dados.dropna(subset=features_disponiveis + ['PESO'])
    n_removidas = len(dados) - len(dados_clean)
    if n_removidas > 0:
        print(f"ATENCAO: {n_removidas} linhas removidas por valores faltantes")
    print(f"Registros para treinamento: {len(dados_clean)}")

    if len(dados_clean) < 10:
        print("Poucos registros para treinar, pulando...")
        continue

    X = dados_clean[features_disponiveis]
    y = dados_clean['PESO']

    print(f"Estatisticas do PESO: media={y.mean():.1f}, min={y.min():.1f}, max={y.max():.1f}")

    print(f"Treinando ({N_ROUNDS} rodadas x {N_ITER_PER_ROUND} iteracoes)...")

    best_models_per_round = []

    for round_idx in range(N_ROUNDS):
        print(f"  Rodada {round_idx + 1}/{N_ROUNDS}...", end=' ')

        xgb = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_distributions,
            n_iter=N_ITER_PER_ROUND,
            cv=cv,
            scoring='r2',
            n_jobs=1,
            random_state=42 + round_idx,
            return_train_score=True,
            verbose=0
        )

        random_search.fit(X, y)

        best_models_per_round.append({
            'round': round_idx + 1,
            'best_cv_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_,
            'best_params': random_search.best_params_
        })

        print(f"R2 CV = {random_search.best_score_:.4f}")

    overall_best = max(best_models_per_round, key=lambda x: x['best_cv_score'])
    best_model = overall_best['best_estimator']

    print(f"\n  Melhor rodada: {overall_best['round']}")
    print(f"  Melhor R2 CV: {overall_best['best_cv_score']:.4f}")
    print(f"  Parametros: {overall_best['best_params']}")

    y_pred = best_model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print(f"\n  Metricas de treino:")
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")

    print(f"\n  Importancia das features:")
    fi = best_model.feature_importances_
    for name, imp in sorted(zip(features_disponiveis, fi), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.4f}")

    modelo_path = f'modelos/modelo_idade_{idade}.joblib'
    joblib.dump({
        'model': best_model,
        'features': features_disponiveis,
        'params': overall_best['best_params'],
        'cv_score': overall_best['best_cv_score']
    }, modelo_path)
    print(f"\n  Modelo salvo em: {modelo_path}")

    resultados.append({
        'idade': idade,
        'n_registros': len(dados_clean),
        'r2_cv': overall_best['best_cv_score'],
        'r2_treino': r2,
        'rmse': rmse,
        'mae': mae
    })

print(f"\n{'='*80}")
print("RESUMO DO TREINAMENTO")
print(f"{'='*80}")
print(f"{'Idade':>6} | {'N':>6} | {'R2 CV':>8} | {'R2 Treino':>10} | {'RMSE':>8} | {'MAE':>8}")
print("-"*60)
for r in resultados:
    print(f"{r['idade']:6d} | {r['n_registros']:6d} | {r['r2_cv']:8.4f} | {r['r2_treino']:10.4f} | {r['rmse']:8.2f} | {r['mae']:8.2f}")

print(f"\n{'='*80}")
print("TREINAMENTO CONCLUIDO!")
print(f"Total de modelos treinados: {len(resultados)}")
print(f"Modelos salvos na pasta 'modelos/'")
print(f"{'='*80}")
