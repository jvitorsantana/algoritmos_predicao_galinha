"""
Experimento 8 - Regressão Logística com regularização L2.

Pergunta do experimento: treinando só com as medidas do 1º dia de vida, depois
acrescentando as medidas da idade seguinte, e assim por diante - a partir de
que idade (se de alguma) dá para prever o sexo da ave pela morfometria?

Este script responde a essa pergunta com UM modelo: regressão logística L2.
O arquivo irmão, experimento_8_xgboost.py, responde com o outro. Os dois usam
o módulo preparacao.py, então enxergam exatamente as mesmas aves e as mesmas
divisões treino/teste - é o que torna a comparação entre eles justa.

Por que regressão logística aqui:
  - Na janela de 80 dias são 39 features para ~92 aves de treino. Com tantas
    colunas quanto amostras, um modelo flexível decora os dados; a penalidade
    L2 encolhe os coeficientes e segura isso.
  - Os coeficientes dizem QUAIS medidas carregam o sinal, com sinal e
    magnitude. É a tabela que vai para a discussão do artigo.
  - A saída é probabilidade, não apenas um escore ordenável.

Protocolo (detalhado em preparacao.py):
  amostra balanceada (mesmo nº de machos e fêmeas) -> 70% treino / 30% teste
  -> repetido 30 vezes com sorteios diferentes -> média e desvio no teste.

Saídas:
  results/experimento_8_logreg.json
  results/figures/experimento_8_logreg.png
  results/models/experimento_8/logreg_coeficientes_80d.csv
"""
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preparacao import (RESULTS, FIGS, MODELS, N_REPETICOES, FRACAO_TESTE,
                        SEMENTE, avaliar_em_repeticoes, benjamini_hochberg,
                        imprimir_cabecalho, preparar_janelas,
                        sortear_amostra_balanceada, teste_de_permutacao)

# Quantos embaralhamentos de rótulo usar no teste de permutação. Quanto maior,
# mais fino o valor-p (com 200, o menor p possível é 1/201 ~ 0,005).
N_PERMUTACOES = 200

# Repetições usadas DENTRO de cada embaralhamento. Menor que as 30 do resultado
# real só por custo: são 200 x 10 = 2.000 ajustes por janela. A média de 10
# sorteios já é estável o suficiente para desenhar a distribuição nula.
N_REPETICOES_PERMUTACAO = 10


def criar_modelo():
    """Devolve um modelo NOVO, ainda não treinado.

    O StandardScaler vai DENTRO do pipeline de propósito. As medidas estão em
    escalas muito diferentes - o peso vai de ~30 g a ~3.000 g enquanto o bico
    fica entre 1 e 2 cm - e sem padronizar a regressão seria dominada pelo
    peso. Padronizar dentro do pipeline faz com que a média e o desvio sejam
    calculados apenas no treino de cada repetição; fazer isso antes da divisão
    deixaria informação do teste vazar para o treino.

    C=1.0 é a força padrão da penalidade L2, fixada de antemão. Ajustar esse
    valor testando várias opções nos mesmos dados que reportam o resultado
    inflaria o desempenho artificialmente.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty='l2', max_iter=5000,
                           random_state=SEMENTE),
    )


def coeficientes_medios(X, y, n_repeticoes=N_REPETICOES, semente=SEMENTE):
    """Coeficiente médio de cada feature ao longo das repetições.

    Repete o mesmo sorteio balanceado + divisão 70/30 e guarda os coeficientes
    de cada treino. A média entre repetições é bem mais estável do que os
    coeficientes de um ajuste único, e o desvio mostra quais features mantêm o
    sinal e quais oscilam de positivo para negativo conforme a amostra.

    Como as features são padronizadas, os coeficientes são comparáveis entre si:
    valor positivo puxa a previsão para Macho, negativo para Fêmea.
    """
    sorteio = np.random.default_rng(semente)
    X_array = X.to_numpy(dtype=float)
    todos = []

    for repeticao in range(n_repeticoes):
        selecionados = sortear_amostra_balanceada(y, sorteio)
        X_treino, _, y_treino, _ = train_test_split(
            X_array[selecionados], y[selecionados],
            test_size=FRACAO_TESTE, stratify=y[selecionados],
            random_state=semente + repeticao,
        )
        modelo = criar_modelo()
        modelo.fit(X_treino, y_treino)
        todos.append(modelo.named_steps['logisticregression'].coef_[0])

    todos = np.array(todos)
    tabela = pd.DataFrame({
        'feature': X.columns,
        'coeficiente_medio': todos.mean(axis=0),
        'desvio': todos.std(axis=0),
    })
    # Ordena pelo tamanho do efeito, ignorando o sinal.
    tabela['forca'] = tabela['coeficiente_medio'].abs()
    return tabela.sort_values('forca', ascending=False).drop(columns='forca')


# =============================================================================
# EXECUÇÃO
# =============================================================================
# Este bloco só roda quando o arquivo é executado direto, assim:
#     uv run python src/experimento_8/experimento_8_logreg.py
# Quando outro script faz "import experimento_8_logreg" - é o caso do
# graficos.py, que reaproveita as funções acima - o Python precisa entrar
# neste arquivo para ler as definições. A verificação lá embaixo
# ("if __name__ == '__main__'") impede que o experimento inteiro seja
# executado de novo nessa hora.
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    janelas, info = preparar_janelas()
    imprimir_cabecalho('EXPERIMENTO 8 - Regressão Logística L2 (classificação de sexo)',
                       info)

    resultados = []
    for janela in janelas:
        X, y = janela['X'], janela['y']

        # 1) Desempenho real: 30 sorteios balanceados de 70/30.
        metricas = avaliar_em_repeticoes(criar_modelo, X, y,
                                         guardar_confusao=True)

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

    # --- janela decisiva: qual é e o que a sustenta ---
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

    tabela_coef = coeficientes_medios(janela_melhor['X'], janela_melhor['y'])
    print()
    print(f"Medidas que mais pesam na janela de {melhor['idade_max']} dias "
          f"(coeficiente padronizado; + puxa para Macho):")
    print(tabela_coef.head(10).to_string(index=False,
                                         float_format=lambda v: f'{v:7.3f}'))

    caminho_coef = MODELS / f"logreg_coeficientes_{melhor['idade_max']}d.csv"
    tabela_coef.to_csv(caminho_coef, index=False, sep=';', decimal='.')

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
    eixo.plot(idades, medias, 'o-', color='#1f77b4', lw=2, ms=6,
              label='Regressão Logística L2')
    eixo.fill_between(idades, medias - desvios, medias + desvios,
                      color='#1f77b4', alpha=0.18,
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
    eixo.set_title('Experimento 8 - Regressão Logística L2\n'
                   f"{info['n_apos_balanceamento']} aves balanceadas "
                   f"({info['n_treino']} treino / {info['n_teste']} teste)",
                   fontsize=11)
    eixo.set_ylim(0.30, 0.90)
    eixo.grid(alpha=0.3)
    eixo.legend(loc='upper left', fontsize=9)
    figura.tight_layout()
    figura.savefig(FIGS / 'experimento_8_logreg.png', dpi=150)
    plt.close(figura)

    # --- arquivo de resultados ---
    saida = {
        'experimento': 'experimento_8_logreg',
        'modelo': 'Regressão Logística L2 (C=1.0) com padronização',
        'unidade_amostral': 'ave (uma linha por animal)',
        'balanceamento': 'subamostragem dos machos: mesmo nº de machos e fêmeas',
        'divisao': f'{int((1 - FRACAO_TESTE) * 100)}/{int(FRACAO_TESTE * 100)} '
                   f'estratificada, repetida {N_REPETICOES} vezes',
        'features': 'por medida: valor(t), delta desde o dia 1, inclinação na janela',
        'coorte': info,
        'janelas': [{k: v for k, v in r.items() if k != 'confusao_somada'}
                    for r in resultados],
        'melhor_janela': melhor['idade_max'],
        'coeficientes_melhor_janela': tabela_coef.head(15).to_dict(orient='records'),
        'veredito': veredito,
    }
    with open(RESULTS / 'experimento_8_logreg.json', 'w', encoding='utf-8') as arquivo:
        json.dump(saida, arquivo, indent=2, ensure_ascii=False)

    print(f"Resultados ..: results/experimento_8_logreg.json")
    print(f"Figura ......: results/figures/experimento_8_logreg.png")
    print(f"Coeficientes : results/models/experimento_8/{caminho_coef.name}")

if __name__ == '__main__':
    main()
