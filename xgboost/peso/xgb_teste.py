import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

plt.rcParams['figure.figsize'] = [16, 10]

print("="*80)
print("TESTE - PREDICAO DE PESO POR IDADE (XGBoost)")
print("Lendo cada dataset separadamente e gerando CSV com predicoes")
print("="*80)

IDADES = [0, 7, 14, 21, 28, 52, 66, 80, 101, 115]

os.makedirs('predicoes', exist_ok=True)

todos_resultados = []
resumo = []
dados_plot = []
feature_importances_por_idade = []

for idade in IDADES:
    arquivo = f'datasets/dataset_idade_{idade}.csv'
    modelo_path = f'modelos/modelo_idade_{idade}.joblib'

    if not os.path.exists(arquivo):
        print(f"\nArquivo {arquivo} nao encontrado, pulando...")
        continue

    if not os.path.exists(modelo_path):
        print(f"\nModelo {modelo_path} nao encontrado, pulando...")
        print("Execute xgb_treino.py primeiro para gerar os modelos.")
        continue

    print(f"\n{'='*80}")
    print(f"IDADE {idade}")
    print(f"{'='*80}")

    modelo_data = joblib.load(modelo_path)
    model = modelo_data['model']
    features = modelo_data['features']
    cv_score = modelo_data['cv_score']

    print(f"Modelo carregado: R2 CV = {cv_score:.4f}")
    print(f"Features: {features}")

    dados = pd.read_csv(arquivo, sep=';', encoding='utf-8')
    print(f"Registros no dataset: {len(dados)}")

    for col in features:
        if col in dados.columns:
            dados[col] = pd.to_numeric(dados[col], errors='coerce')

    if 'PESO' in dados.columns:
        dados['PESO'] = pd.to_numeric(dados['PESO'], errors='coerce')

    dados_pred = dados.dropna(subset=features).copy()
    n_removidas = len(dados) - len(dados_pred)
    if n_removidas > 0:
        print(f"ATENCAO: {n_removidas} registros sem features completas, removidos")

    if len(dados_pred) == 0:
        print("Sem registros validos para predicao, pulando...")
        continue

    X = dados_pred[features]

    y_pred = model.predict(X)

    resultado_idade = pd.DataFrame({
        'ANIMAL': dados_pred['ANIMAL'].values,
        'IDADE': idade,
        'PESO_PREDITO': np.round(y_pred, 2)
    })

    tem_peso = 'PESO' in dados_pred.columns and dados_pred['PESO'].notna().sum() > 0
    if tem_peso:
        resultado_idade['PESO_REAL'] = dados_pred['PESO'].values

        mask_valido = resultado_idade['PESO_REAL'].notna()
        n_validos = mask_valido.sum()

        if n_validos > 0:
            y_real_valido = resultado_idade.loc[mask_valido, 'PESO_REAL'].values
            y_pred_valido = resultado_idade.loc[mask_valido, 'PESO_PREDITO'].values

            r2 = r2_score(y_real_valido, y_pred_valido)
            rmse = np.sqrt(mean_squared_error(y_real_valido, y_pred_valido))
            mae = mean_absolute_error(y_real_valido, y_pred_valido)

            print(f"\nMetricas de predicao ({n_validos} registros com peso real):")
            print(f"  R2:   {r2:.4f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")

            erros = np.abs(y_real_valido - y_pred_valido)
            print(f"  Erro medio por animal: {np.mean(erros):.2f}g")
            print(f"  Erro maximo: {np.max(erros):.2f}g")

            resumo.append({
                'idade': idade,
                'n_animais': len(dados_pred),
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            })
        else:
            resumo.append({
                'idade': idade,
                'n_animais': len(dados_pred),
                'r2': None,
                'rmse': None,
                'mae': None
            })
    else:
        print(f"\nPESO real nao disponivel, apenas predicoes geradas.")
        resumo.append({
            'idade': idade,
            'n_animais': len(dados_pred),
            'r2': None,
            'rmse': None,
            'mae': None
        })

    feature_importances_por_idade.append({
        'idade': idade,
        'features': features,
        'importances': model.feature_importances_
    })

    if tem_peso:
        mask_valido = resultado_idade['PESO_REAL'].notna()
        if mask_valido.sum() > 0:
            dados_plot.append({
                'idade': idade,
                'y_real': resultado_idade.loc[mask_valido, 'PESO_REAL'].values,
                'y_pred': resultado_idade.loc[mask_valido, 'PESO_PREDITO'].values
            })

    csv_path = f'predicoes/predicoes_idade_{idade}.csv'
    resultado_idade[['ANIMAL', 'IDADE', 'PESO_PREDITO']].to_csv(
        csv_path, sep=';', index=False, encoding='utf-8'
    )
    print(f"CSV salvo: {csv_path}")

    todos_resultados.append(resultado_idade)

if todos_resultados:
    resultado_final = pd.concat(todos_resultados, ignore_index=True)

    colunas_csv = ['ANIMAL', 'IDADE', 'PESO_PREDITO']
    resultado_final[colunas_csv].to_csv(
        'predicoes/predicoes_todas_idades.csv', sep=';', index=False, encoding='utf-8'
    )

    print(f"\n{'='*80}")
    print("RESUMO DAS PREDICOES")
    print(f"{'='*80}")
    print(f"Total de predicoes: {len(resultado_final)}")
    print(f"Animais unicos: {resultado_final['ANIMAL'].nunique()}")
    print(f"Idades processadas: {sorted(resultado_final['IDADE'].unique().tolist())}")

    print(f"\n{'Idade':>6} | {'N':>6} | {'R2':>8} | {'RMSE':>8} | {'MAE':>8}")
    print("-"*50)
    for r in resumo:
        if r['r2'] is not None:
            print(f"{r['idade']:6d} | {r['n_animais']:6d} | {r['r2']:8.4f} | {r['rmse']:8.2f} | {r['mae']:8.2f}")
        else:
            print(f"{r['idade']:6d} | {r['n_animais']:6d} |      N/A |      N/A |      N/A")

    print(f"\nArquivos gerados na pasta 'predicoes/':")
    for idade in IDADES:
        csv_path = f'predicoes/predicoes_idade_{idade}.csv'
        if os.path.exists(csv_path):
            print(f"  - {csv_path}")
    print(f"  - predicoes/predicoes_todas_idades.csv (consolidado)")

if dados_plot and resumo:
    all_y_real = np.concatenate([d['y_real'] for d in dados_plot])
    all_y_pred = np.concatenate([d['y_pred'] for d in dados_plot])
    all_idades = np.concatenate([np.full(len(d['y_real']), d['idade']) for d in dados_plot])
    all_residuos = all_y_real - all_y_pred

    resumo_valido = [r for r in resumo if r['r2'] is not None]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Predicao de Peso por Idade - XGBoost (Teste)', fontsize=16, fontweight='bold')

    plt.subplot(2, 2, 1)
    idades_r2 = [r['idade'] for r in resumo_valido]
    scores_r2 = [r['r2'] for r in resumo_valido]
    melhor_idx = np.argmax(scores_r2)
    cores = ['red' if i == melhor_idx else 'blue' for i in range(len(scores_r2))]
    plt.bar(range(len(idades_r2)), scores_r2, color=cores, tick_label=[str(i) for i in idades_r2])
    plt.xlabel('Idade (dias)')
    plt.ylabel('R2')
    plt.title(f'R2 por Idade (vermelho = melhor: {max(scores_r2):.4f})')
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(2, 2, 2)
    scatter = plt.scatter(all_y_real, all_y_pred, alpha=0.5, c=all_idades, cmap='viridis', s=15)
    plt.colorbar(scatter, label='Idade (dias)')
    min_val = min(all_y_real.min(), all_y_pred.min())
    max_val = max(all_y_real.max(), all_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    r2_global = r2_score(all_y_real, all_y_pred)
    rmse_global = np.sqrt(mean_squared_error(all_y_real, all_y_pred))
    plt.xlabel('Peso Real (g)')
    plt.ylabel('Peso Predito (g)')
    plt.title(f'Real vs Predito (R2 = {r2_global:.4f}, RMSE = {rmse_global:.1f}g)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    idades_mae = [r['idade'] for r in resumo_valido]
    scores_mae = [r['mae'] for r in resumo_valido]
    plt.bar(range(len(idades_mae)), scores_mae, color='red', alpha=0.6, tick_label=[str(i) for i in idades_mae])
    plt.xlabel('Idade (dias)')
    plt.ylabel('MAE (g)')
    plt.title('Erro Medio Absoluto por Idade')
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(2, 2, 4)
    plt.scatter(all_y_pred, all_residuos, alpha=0.5, c=all_idades, cmap='viridis', s=15)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Peso Predito (g)')
    plt.ylabel('Residuo (g)')
    plt.title('Analise de Residuos')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resultado_xgb_teste_geral.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGrafico geral salvo em: resultado_xgb_teste_geral.png")

    n_idades = len(feature_importances_por_idade)
    n_cols = min(3, n_idades)
    n_rows = (n_idades + n_cols - 1) // n_cols

    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    fig2.suptitle('Importancia das Features por Idade - XGBoost', fontsize=16, fontweight='bold')

    if n_idades == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, fi_data in enumerate(feature_importances_por_idade):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        importances = fi_data['importances']
        feat_names = fi_data['features']
        sorted_idx = np.argsort(importances)

        ax.barh(range(len(sorted_idx)), importances[sorted_idx], color='green', alpha=0.6)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feat_names[i] for i in sorted_idx])
        ax.set_xlabel('Importancia')
        ax.set_title(f'Idade {fi_data["idade"]}')
        ax.grid(True, alpha=0.3, axis='x')

    for idx in range(n_idades, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig('resultado_xgb_teste_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Grafico de features salvo em: resultado_xgb_teste_features.png")

print(f"\n{'='*80}")
print("TESTE CONCLUIDO!")
print(f"{'='*80}")
