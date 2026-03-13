import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
import joblib

ROOT = Path(__file__).resolve().parent.parent

SVM_DATA_DIR = ROOT / 'data' / 'svm'
SVM_MODEL_DIR = ROOT / 'results' / 'models' / 'svm'
SVM_PRED_DIR = ROOT / 'results' / 'predictions' / 'svm'
FIGURES_DIR = ROOT / 'results' / 'figures'

plt.rcParams['figure.figsize'] = [16, 10]

print("="*80)
print("TESTE - PREDICAO DE SEXO POR IDADE (SVM)")
print("Lendo cada dataset separadamente e gerando CSV com predicoes")
print("="*80)

IDADES = sorted([
    int(f.replace('dataset_idade_', '').replace('.csv', ''))
    for f in os.listdir(SVM_DATA_DIR)
    if f.startswith('dataset_idade_') and f.endswith('.csv')
    and f.replace('dataset_idade_', '').replace('.csv', '').isdigit()
    and 0 <= int(f.replace('dataset_idade_', '').replace('.csv', '')) <= 115
])
print(f"Datasets encontrados: {IDADES}")

SVM_PRED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

todos_resultados = []
resumo = []
dados_plot = []

for idade in IDADES:
    arquivo = SVM_DATA_DIR / f'dataset_idade_{idade}.csv'
    modelo_path = SVM_MODEL_DIR / f'modelo_svm_idade_{idade}.joblib'

    if not arquivo.exists():
        print(f"\nArquivo {arquivo} nao encontrado, pulando...")
        continue

    if not modelo_path.exists():
        print(f"\nModelo {modelo_path} nao encontrado, pulando...")
        print("Execute svm_treino.py primeiro para gerar os modelos.")
        continue

    print(f"\n{'='*80}")
    print(f"IDADE {idade}")
    print(f"{'='*80}")

    modelo_data = joblib.load(modelo_path)
    model = modelo_data['model']
    features = modelo_data['features']
    cv_score = modelo_data['cv_score']
    best_round = modelo_data['best_round']
    rounds = modelo_data['rounds']

    print(f"Modelo carregado: F1 CV = {cv_score:.4f} (melhor rodada: {best_round})")
    print(f"Features: {features}")
    print(f"Kernel: {modelo_data['params']['svm__kernel']} | "
          f"C: {modelo_data['params']['svm__C']:.4f}")

    dados = pd.read_csv(arquivo, sep=';', encoding='utf-8')
    print(f"Registros no dataset: {len(dados)}")

    for col in features:
        if col in dados.columns:
            dados[col] = pd.to_numeric(dados[col], errors='coerce')

    if 'SEXO' in dados.columns:
        dados['SEXO_MAPPED'] = dados['SEXO'].map({'Macho': 1, 'Fêmea': 0})

    dados_pred = dados.dropna(subset=features).copy()
    n_removidas = len(dados) - len(dados_pred)
    if n_removidas > 0:
        print(f"ATENCAO: {n_removidas} registros sem features completas, removidos")

    if len(dados_pred) == 0:
        print("Sem registros validos para predicao, pulando...")
        continue

    X = dados_pred[features]

    y_pred = model.predict(X)

    sexo_label = {1: 'Macho', 0: 'Femea'}
    y_pred_label = np.array([sexo_label[v] for v in y_pred])

    resultado_idade = pd.DataFrame({
        'ANIMAL': dados_pred['ANIMAL'].values,
        'IDADE': idade,
        'SEXO_PREDITO': y_pred_label,
        'SEXO_PREDITO_NUM': y_pred
    })

    tem_sexo = 'SEXO_MAPPED' in dados_pred.columns and dados_pred['SEXO_MAPPED'].notna().sum() > 0
    if tem_sexo:
        resultado_idade['SEXO_REAL'] = dados_pred['SEXO_MAPPED'].values
        mask_valido = resultado_idade['SEXO_REAL'].notna()
        n_validos = mask_valido.sum()

        if n_validos > 0:
            y_real_valido = resultado_idade.loc[mask_valido, 'SEXO_REAL'].values.astype(int)
            y_pred_valido = resultado_idade.loc[mask_valido, 'SEXO_PREDITO_NUM'].values.astype(int)

            acc = accuracy_score(y_real_valido, y_pred_valido)
            prec = precision_score(y_real_valido, y_pred_valido, zero_division=0)
            rec = recall_score(y_real_valido, y_pred_valido, zero_division=0)
            f1 = f1_score(y_real_valido, y_pred_valido, zero_division=0)
            cm = confusion_matrix(y_real_valido, y_pred_valido)

            print(f"\nMetricas de predicao ({n_validos} registros com sexo real):")
            print(f"  Acuracia:  {acc:.4f}")
            print(f"  Precisao:  {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

            print(f"\nMatriz de Confusao:")
            print(f"  {cm}")

            print(f"\nRelatorio de Classificacao:")
            print(classification_report(y_real_valido, y_pred_valido,
                                        target_names=['Femea', 'Macho'], zero_division=0))

            dados_plot.append({
                'idade': idade,
                'y_real': y_real_valido,
                'y_pred': y_pred_valido,
                'cm': cm,
                'rounds': rounds,
                'cv_score': cv_score
            })

            resumo.append({
                'idade': idade,
                'n_animais': n_validos,
                'acc': acc,
                'prec': prec,
                'rec': rec,
                'f1': f1,
                'f1_cv': cv_score
            })
        else:
            resumo.append({
                'idade': idade,
                'n_animais': len(dados_pred),
                'acc': None, 'prec': None, 'rec': None, 'f1': None, 'f1_cv': cv_score
            })
    else:
        print(f"\nSEXO real nao disponivel, apenas predicoes geradas.")
        resumo.append({
            'idade': idade,
            'n_animais': len(dados_pred),
            'acc': None, 'prec': None, 'rec': None, 'f1': None, 'f1_cv': cv_score
        })

    csv_path = SVM_PRED_DIR / f'predicoes_svm_idade_{idade}.csv'
    resultado_idade[['ANIMAL', 'IDADE', 'SEXO_PREDITO']].to_csv(
        csv_path, sep=';', index=False, encoding='utf-8'
    )
    print(f"CSV salvo: {csv_path}")

    todos_resultados.append(resultado_idade)

if todos_resultados:
    resultado_final = pd.concat(todos_resultados, ignore_index=True)

    resultado_final[['ANIMAL', 'IDADE', 'SEXO_PREDITO']].to_csv(
        SVM_PRED_DIR / 'predicoes_svm_todas_idades.csv',
        sep=';', index=False, encoding='utf-8'
    )

    print(f"\n{'='*80}")
    print("RESUMO DAS PREDICOES")
    print(f"{'='*80}")
    print(f"Total de predicoes: {len(resultado_final)}")
    print(f"Animais unicos: {resultado_final['ANIMAL'].nunique()}")

    print(f"\n{'Idade':>6} | {'N':>6} | {'F1 CV':>8} | {'F1':>8} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8}")
    print("-"*65)
    for r in resumo:
        if r['f1'] is not None:
            print(f"{r['idade']:6d} | {r['n_animais']:6d} | {r['f1_cv']:8.4f} | "
                  f"{r['f1']:8.4f} | {r['acc']:8.4f} | {r['prec']:8.4f} | {r['rec']:8.4f}")
        else:
            print(f"{r['idade']:6d} | {r['n_animais']:6d} | {r['f1_cv']:8.4f} |"
                  f"     N/A |      N/A |      N/A |      N/A")

    print(f"\nArquivos gerados na pasta '{SVM_PRED_DIR}':")
    for idade in IDADES:
        path = SVM_PRED_DIR / f'predicoes_svm_idade_{idade}.csv'
        if path.exists():
            print(f"  - {path}")
    print(f"  - {SVM_PRED_DIR / 'predicoes_svm_todas_idades.csv'} (consolidado)")

if dados_plot and resumo:
    resumo_valido = [r for r in resumo if r['f1'] is not None]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Predicao de Sexo por Idade - SVM (Teste)', fontsize=16, fontweight='bold')

    plt.subplot(2, 2, 1)
    idades_vals = [r['idade'] for r in resumo_valido]
    f1_vals = [r['f1'] for r in resumo_valido]
    f1_cv_vals = [r['f1_cv'] for r in resumo_valido]
    melhor_idx = int(np.argmax(f1_vals))
    cores = ['red' if i == melhor_idx else 'blue' for i in range(len(f1_vals))]
    bars = plt.bar(range(len(idades_vals)), f1_vals, color=cores,
                   tick_label=[str(i) for i in idades_vals], label='F1 Predicao')
    plt.plot(range(len(idades_vals)), f1_cv_vals, 'g--o', linewidth=1.5,
             markersize=6, label='F1 CV (treino)')
    for i, (bar, val) in enumerate(zip(bars, f1_vals)):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    plt.xlabel('Idade (dias)')
    plt.ylabel('F1-Score')
    plt.title(f'F1 por Idade (vermelho = melhor: {max(f1_vals):.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(2, 2, 2)
    melhor_dado = dados_plot[melhor_idx]
    all_scores = []
    round_labels = []
    for r in melhor_dado['rounds']:
        cv_res = r['cv_results']
        all_scores.extend(cv_res['mean_test_score'])
        round_labels.extend([r['round']] * len(cv_res['mean_test_score']))
    scatter = plt.scatter(range(len(all_scores)), all_scores,
                          c=round_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Rodada')
    plt.xlabel('Indice da Configuracao')
    plt.ylabel('F1 CV')
    plt.title(f"Todas as Configuracoes - Idade {melhor_dado['idade']} (melhor)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    round_scores = [[s for s in r['cv_results']['mean_test_score']]
                    for r in melhor_dado['rounds']]
    labels_box = [f"R{r['round']}" if r['round'] % 5 == 0 or r['round'] == 1 else ''
                  for r in melhor_dado['rounds']]
    plt.boxplot(round_scores, labels=labels_box)
    plt.ylabel('F1 CV')
    plt.xlabel('Rodada')
    plt.title(f"Distribuicao de F1 por Rodada - Idade {melhor_dado['idade']}")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    acc_vals = [r['acc'] for r in resumo_valido]
    prec_vals = [r['prec'] for r in resumo_valido]
    rec_vals = [r['rec'] for r in resumo_valido]
    x = np.arange(len(idades_vals))
    width = 0.25
    plt.bar(x - width, acc_vals, width, label='Acuracia', color='blue', alpha=0.7)
    plt.bar(x, prec_vals, width, label='Precisao', color='green', alpha=0.7)
    plt.bar(x + width, rec_vals, width, label='Recall', color='orange', alpha=0.7)
    plt.xticks(x, [str(i) for i in idades_vals])
    plt.xlabel('Idade (dias)')
    plt.ylabel('Score')
    plt.title('Acuracia, Precisao e Recall por Idade')
    plt.legend()
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'resultado_svm_teste_geral.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGrafico geral salvo em: {FIGURES_DIR / 'resultado_svm_teste_geral.png'}")

    n_idades = len(dados_plot)
    n_cols = min(4, n_idades)
    n_rows = (n_idades + n_cols - 1) // n_cols

    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig2.suptitle('Matrizes de Confusao por Idade - SVM', fontsize=16, fontweight='bold')

    axes_flat = np.array(axes).flatten() if n_idades > 1 else [axes]

    for idx, dp in enumerate(dados_plot):
        ax = axes_flat[idx]
        disp = ConfusionMatrixDisplay(confusion_matrix=dp['cm'],
                                      display_labels=['Femea', 'Macho'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        f1_val = resumo_valido[idx]['f1']
        ax.set_title(f"Idade {dp['idade']} | F1={f1_val:.3f}")

    for idx in range(n_idades, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'resultado_svm_teste_confusao.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Grafico de matrizes de confusao salvo em: {FIGURES_DIR / 'resultado_svm_teste_confusao.png'}")

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle('Predicao de Sexo - Comparacao de Desempenho no Conjunto de Teste (SVM)',
                  fontsize=14, fontweight='bold')

    cores_idades = plt.cm.tab10(np.linspace(0, 1, len(resumo_valido)))

    metrics_info = [
        ('acc', 'Acuracia', axes3[0]),
        ('f1', 'F1-Score', axes3[1]),
        ('f1_cv', 'F1 CV (treino)', axes3[2])
    ]

    for key, label, ax in metrics_info:
        vals = [r[key] for r in resumo_valido]
        labels = [f"Idade {r['idade']}" for r in resumo_valido]
        ax.bar(labels, vals, color=cores_idades, alpha=0.8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'resultado_svm_teste_comparacao.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Grafico de comparacao salvo em: {FIGURES_DIR / 'resultado_svm_teste_comparacao.png'}")

print(f"\n{'='*80}")
print("TESTE CONCLUIDO!")
print(f"{'='*80}")
