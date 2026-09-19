"""
Experimento 8 - XGBoost.

Mesma pergunta do arquivo irmão experimento_8_logreg.py: treinando só com as
medidas do 1º dia, depois acrescentando a idade seguinte, e assim por diante -
a partir de quando dá para prever o sexo da ave pela morfometria?

Aqui ela é respondida com XGBoost. Os dois scripts usam o módulo preparacao.py,
então recebem exatamente as mesmas aves, as mesmas features e as mesmas
divisões treino/teste. Só o classificador muda, e é isso que torna a
comparação entre eles interpretável.

Por que XGBoost como segundo modelo:
  - É uma família diferente da regressão logística. Árvores capturam relações
    não lineares e interações entre medidas; a logística só combina as
    variáveis de forma aditiva. Comparar os dois responde a uma pergunta real:
    existe alguma não linearidade que valha a pena capturar aqui?
  - É o modelo usado nos experimentos 1, 2, 3, 5, 6 e 7 deste projeto, então
    manter ele aqui liga o experimento 8 a todos os anteriores.
  - É o padrão esperado na área, o que evita a pergunta óbvia de revisor.

Uma diferença em relação ao script da logística: aqui não há padronização.
Árvores decidem por cortes do tipo "PESO > 1.200", e um corte não muda de lugar
se a variável for reescalada. Padronizar não atrapalharia, mas seria uma etapa
sem efeito, então fica de fora para o código dizer a verdade sobre o modelo.

Saídas:
  results/experimento_8_xgboost.json
  results/figures/experimento_8_xgboost.png
  results/models/experimento_8/xgboost_importancias_<idade>d.csv
"""
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from preparacao import (RESULTS, FIGS, MODELS, N_REPETICOES, FRACAO_TESTE,
                        SEMENTE, avaliar_em_repeticoes, benjamini_hochberg,
                        imprimir_cabecalho, preparar_janelas,
                        sortear_amostra_balanceada, teste_de_permutacao)

# Mesmos valores do script da logística, para que os valores-p dos dois tenham
# a mesma resolução e possam ser comparados lado a lado.
N_PERMUTACOES = 200
N_REPETICOES_PERMUTACAO = 10


def criar_modelo():
    """Devolve um XGBoost NOVO, ainda não treinado.

    Os hiperparâmetros são fixos e escolhidos de antemão, pensando no tamanho
    da amostra - são ~92 aves de treino:

      max_depth=3        árvores rasas. Com poucas aves, árvores profundas
                         isolam animais individuais em vez de achar padrão.
      learning_rate=0.05 passos pequenos, compensados por mais árvores.
      n_estimators=200   número de árvores.
      subsample=0.8      cada árvore vê 80% das aves, o que injeta variação e
      colsample=0.8      cada árvore vê 80% das features - as duas juntas
                         reduzem a chance de decorar o treino.
      reg_lambda=1.0     penalidade L2 sobre os pesos das folhas.

    Não há scale_pos_weight porque a amostra já chega balanceada: o
    preparacao.py sorteia o mesmo número de machos e de fêmeas.

    n_jobs=1 de propósito: a base é pequena e o custo de coordenar várias
    threads seria maior do que o ganho.
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric='logloss',
        random_state=SEMENTE,
        n_jobs=1,
    )


def importancias_medias(X, y, n_repeticoes=N_REPETICOES, semente=SEMENTE):
    """Importância média de cada feature ao longo das repetições.

    Repete o sorteio balanceado + divisão 70/30 e guarda a importância por
    ganho de cada treino. A média entre repetições é bem mais confiável do que
    a de um ajuste único: com poucas aves, um treino isolado pode eleger uma
    feature por acaso.

    Atenção na leitura: importância não tem sinal. Ela diz o quanto a medida
    ajudou a separar as classes, não para que lado ela empurra. Para saber a
    direção, use a tabela de coeficientes do script da logística.
    """
    sorteio = np.random.default_rng(semente)
    X_array = X.to_numpy(dtype=float)
    todas = []

    for repeticao in range(n_repeticoes):
        selecionados = sortear_amostra_balanceada(y, sorteio)
        X_treino, _, y_treino, _ = train_test_split(
            X_array[selecionados], y[selecionados],
            test_size=FRACAO_TESTE, stratify=y[selecionados],
            random_state=semente + repeticao,
        )
        modelo = criar_modelo()
        modelo.fit(X_treino, y_treino)
        todas.append(modelo.feature_importances_)

    todas = np.array(todas)
    tabela = pd.DataFrame({
        'feature': X.columns,
        'importancia_media': todas.mean(axis=0),
        'desvio': todas.std(axis=0),
    })
    return tabela.sort_values('importancia_media', ascending=False)


# =============================================================================
# EXECUÇÃO
# =============================================================================
# Este bloco só roda quando o arquivo é executado direto, assim:
#     uv run python src/experimento_8/experimento_8_xgboost.py
# Quando outro script faz "import experimento_8_xgboost" - é o caso do
# graficos.py, que reaproveita as funções acima - o Python precisa entrar
# neste arquivo para ler as definições. A verificação lá embaixo
# ("if __name__ == '__main__'") impede que o experimento inteiro seja
# executado de novo nessa hora.
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    janelas, info = preparar_janelas()
    imprimir_cabecalho('EXPERIMENTO 8 - XGBoost (classificação de sexo)', info)

    resultados = []
    for janela in janelas:
        X, y = janela['X'], janela['y']

        # 1) Desempenho real: 30 sorteios balanceados de 70/30.
        metricas = avaliar_em_repeticoes(criar_modelo, X, y, guardar_confusao=True)

        # 2) O resultado é melhor do que se o sexo fosse sorteado?
        valor_p = teste_de_permutacao(
            criar_modelo, X, y,
            auc_observado=metricas['auc_media'],
            n_permutacoes=N_PERMUTACOES,
            n_repeticoes=N_REPETICOES_PERMUTACAO,
        )

        metricas['idade_max'] = janela['idade_max']
        metricas['idades'] = janela['idades']
        metricas['n_features'] = int(X.shape[1])
        metricas['p_permutacao'] = round(float(valor_p), 4)
        resultados.append(metricas)

        print(f"  janela até {janela['idade_max']:>3}d concluída "
              f"(AUC {metricas['auc_media']:.3f}, p {valor_p:.3f})")

    # 3) Corrige os valores-p pelo número de janelas testadas.
    qs, significativas = benjamini_hochberg([r['p_permutacao'] for r in resultados])
    for resultado, q, e_significativa in zip(resultados, qs, significativas):
        resultado['q_bh'] = round(float(q), 4)
        resultado['significativo'] = bool(e_significativa)

    # --- tabela ---
    print()
    print(f"{'janela':>7} {'feats':>6} {'AUC':>16} {'acurácia':>16} "
          f"{'sens.':>7} {'espec.':>7} {'p':>7} {'q(BH)':>7}")
    print('-' * 92)
    for r in resultados:
        marca = ' *' if r['significativo'] else ''
        print(f"{r['idade_max']:>6}d {r['n_features']:>6} "
              f"{r['auc_media']:>9.3f} ±{r['auc_dp']:<5.3f} "
              f"{r['acuracia_media']:>9.3f} ±{r['acuracia_dp']:<5.3f} "
              f"{r['sensibilidade_macho']:>7.3f} {r['especificidade_femea']:>7.3f} "
              f"{r['p_permutacao']:>7.3f} {r['q_bh']:>7.3f}{marca}")
    print('-' * 92)
    print(f"Média ± desvio sobre {N_REPETICOES} divisões 70/30 balanceadas. "
          f"Classes iguais, então acaso = 0,500 em AUC e acurácia.")
    print(f"sens. = acerto entre machos | espec. = acerto entre fêmeas")
    print(f"p = teste de permutação ({N_PERMUTACOES} embaralhamentos); "
          f"q = p corrigido por Benjamini-Hochberg. * = q < 0,05")

    # --- janela decisiva ---
    melhor = max(resultados, key=lambda r: r['auc_media'])
    janela_melhor = next(j for j in janelas if j['idade_max'] == melhor['idade_max'])

    print()
    print('=' * 78)
    print(f"Melhor janela: até {melhor['idade_max']} dias  |  "
          f"AUC {melhor['auc_media']:.3f} ± {melhor['auc_dp']:.3f}  "
          f"(pior sorteio {melhor['auc_min']:.3f}, melhor {melhor['auc_max']:.3f})")

    tn, fp = melhor['confusao_somada'][0]
    fn, tp = melhor['confusao_somada'][1]
    print(f"Matriz de confusão somada nas {N_REPETICOES} repetições:")
    print(f"                    previsto Fêmea   previsto Macho")
    print(f"  real Fêmea {tn:>16} {fp:>16}")
    print(f"  real Macho {fn:>16} {tp:>16}")

    tabela_imp = importancias_medias(janela_melhor['X'], janela_melhor['y'])
    print()
    print(f"Medidas mais usadas na janela de {melhor['idade_max']} dias "
          f"(importância por ganho; sem direção):")
    print(tabela_imp.head(10).to_string(index=False,
                                        float_format=lambda v: f'{v:7.4f}'))

    caminho_imp = MODELS / f"xgboost_importancias_{melhor['idade_max']}d.csv"
    tabela_imp.to_csv(caminho_imp, index=False, sep=';', decimal='.')

    janelas_com_sinal = [r['idade_max'] for r in resultados if r['significativo']]
    if janelas_com_sinal:
        veredito = (f"Janelas acima do acaso após correção: {janelas_com_sinal} dias. "
                    f"A predição de sexo passa a funcionar a partir de "
                    f"{min(janelas_com_sinal)} dias.")
    else:
        veredito = ("Nenhuma janela superou o acaso após a correção: a morfometria "
                    "não prevê o sexo em nenhuma das idades analisadas.")
    print()
    print(veredito)
    print('=' * 78)

    # --- figura ---
    idades = [r['idade_max'] for r in resultados]
    medias = np.array([r['auc_media'] for r in resultados])
    desvios = np.array([r['auc_dp'] for r in resultados])

    figura, eixo = plt.subplots(figsize=(8.5, 5.4))
    eixo.plot(idades, medias, 'o-', color='#d62728', lw=2, ms=6, label='XGBoost')
    eixo.fill_between(idades, medias - desvios, medias + desvios,
                      color='#d62728', alpha=0.18,
                      label=f'± 1 desvio ({N_REPETICOES} divisões)')
    eixo.axhline(0.5, color='gray', ls='--', lw=1.2)
    eixo.text(idades[-1], 0.508, 'acaso', color='gray', ha='right', fontsize=9)

    for r in resultados:
        if r['significativo']:
            eixo.plot(r['idade_max'], r['auc_media'], '*', color='black', ms=15,
                      zorder=5)
    eixo.plot([], [], '*', color='black', ms=11, ls='none', label='q(BH) < 0,05')

    eixo.set_xlabel('idade máxima incluída na janela (dias)')
    eixo.set_ylabel('AUC no conjunto de teste (30%)')
    eixo.set_title('Experimento 8 - XGBoost\n'
                   f"{info['n_apos_balanceamento']} aves balanceadas "
                   f"({info['n_treino']} treino / {info['n_teste']} teste)",
                   fontsize=11)
    eixo.set_ylim(0.30, 0.90)
    eixo.grid(alpha=0.3)
    eixo.legend(loc='upper left', fontsize=9)
    figura.tight_layout()
    figura.savefig(FIGS / 'experimento_8_xgboost.png', dpi=150)
    plt.close(figura)

    # --- arquivo de resultados ---
    saida = {
        'experimento': 'experimento_8_xgboost',
        'modelo': 'XGBoost (200 árvores, profundidade 3, lr 0.05)',
        'unidade_amostral': 'ave (uma linha por animal)',
        'balanceamento': 'subamostragem dos machos: mesmo nº de machos e fêmeas',
        'divisao': f'{int((1 - FRACAO_TESTE) * 100)}/{int(FRACAO_TESTE * 100)} '
                   f'estratificada, repetida {N_REPETICOES} vezes',
        'features': 'por medida: valor(t), delta desde o dia 1, inclinação na janela',
        'coorte': info,
        'janelas': [{k: v for k, v in r.items() if k != 'confusao_somada'}
                    for r in resultados],
        'melhor_janela': melhor['idade_max'],
        'importancias_melhor_janela': tabela_imp.head(15).to_dict(orient='records'),
        'veredito': veredito,
    }
    with open(RESULTS / 'experimento_8_xgboost.json', 'w', encoding='utf-8') as arquivo:
        json.dump(saida, arquivo, indent=2, ensure_ascii=False)

    print(f"Resultados ..: results/experimento_8_xgboost.json")
    print(f"Figura ......: results/figures/experimento_8_xgboost.png")
    print(f"Importâncias : results/models/experimento_8/{caminho_imp.name}")

if __name__ == '__main__':
    main()
