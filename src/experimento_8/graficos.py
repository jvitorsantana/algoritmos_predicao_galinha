"""
Experimento 8 - gráficos de leitura.

Gera quatro figuras, cada uma respondendo a uma pergunta diferente:

  1. acuracia.png         Quanto o modelo acerta em cada janela de idade?
  2. matriz_confusao.png  Na janela decisiva, ONDE ele erra - confunde macho
                          com fêmea, ou o contrário?
  3. curva_roc.png        O que significa o AUC, visualmente.
  4. features.png         Quais medidas o modelo usa para decidir.

Os dois classificadores são reavaliados aqui com o MESMO protocolo dos scripts
experimento_8_logreg.py e experimento_8_xgboost.py - mesma função, mesma
semente - então os números das figuras batem exatamente com os das tabelas.
O que não é recalculado são os valores-p (custam alguns minutos de permutação):
esses vêm dos arquivos JSON já gerados.

Rode os dois scripts de modelo antes deste.

Saídas em results/figures/ e, copiadas, em src/experimento_8/resultados/.
"""
import warnings
warnings.filterwarnings('ignore')

import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from sklearn.metrics import roc_curve, roc_auc_score

from preparacao import (RESULTS, FIGS, N_REPETICOES, avaliar_em_repeticoes,
                        preparar_janelas)
import experimento_8_logreg as mod_logreg
import experimento_8_xgboost as mod_xgboost


# =============================================================================
# APARÊNCIA
# =============================================================================
# Duas cores só, uma por modelo, escolhidas para continuarem distinguíveis por
# quem tem daltonismo (separação verificada: ΔE 24,7 em visão protanope).
COR_LOGREG = '#2a78d6'      # azul
COR_XGBOOST = '#eb6834'     # laranja
COR_ACERTO = '#2a78d6'
COR_ERRO = '#e34948'        # vermelho, só para valores negativos/erros

TINTA = '#0b0b0b'           # texto principal
TINTA_FRACA = '#52514e'     # rótulos e eixos
FUNDO = '#fcfcfb'
GRADE = '#e3e2de'

# Rampa azul clara -> escura, para a matriz de confusão (mais escuro = mais aves)
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    'azul', ['#eef4fd', '#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#184f95'])

plt.rcParams.update({
    'figure.facecolor': FUNDO,
    'axes.facecolor': FUNDO,
    'axes.edgecolor': GRADE,
    'axes.labelcolor': TINTA_FRACA,
    'axes.titlecolor': TINTA,
    'text.color': TINTA,
    'xtick.color': TINTA_FRACA,
    'ytick.color': TINTA_FRACA,
    'grid.color': GRADE,
    'font.size': 10,
})

SAIDA_LOCAL = Path(__file__).resolve().parent / 'resultados'


def limpar_eixo(eixo, grade_y=True):
    """Tira as bordas de cima e da direita e deixa a grade discreta."""
    for lado in ('top', 'right'):
        eixo.spines[lado].set_visible(False)
    if grade_y:
        eixo.set_axisbelow(True)
        eixo.grid(axis='y', color=GRADE, lw=0.8)


def rotulo_janela(idade, primeira):
    """'1º dia' para a primeira janela, '+7d' para as seguintes.

    O sinal de mais existe para lembrar que as janelas são CUMULATIVAS: a de
    66 dias contém também todas as medições anteriores, não só as daquele dia.
    """
    return '1º dia' if primeira else f'+{idade}d'


# =============================================================================
# COLETA - roda o protocolo e guarda tudo que as figuras precisam
# =============================================================================
print('Reavaliando os dois modelos (mesmo protocolo dos scripts)...')
janelas, info = preparar_janelas()

MODELOS = [
    ('Regressão Logística L2', mod_logreg.criar_modelo, COR_LOGREG, 'logreg'),
    ('XGBoost', mod_xgboost.criar_modelo, COR_XGBOOST, 'xgboost'),
]

coleta = {}
for nome, criar_modelo, cor, chave in MODELOS:
    por_janela = []
    for janela in janelas:
        metricas = avaliar_em_repeticoes(
            criar_modelo, janela['X'], janela['y'],
            guardar_confusao=True, guardar_escores=True)
        metricas['idade_max'] = janela['idade_max']
        por_janela.append(metricas)
    coleta[chave] = por_janela
    print(f'  {nome}: {len(por_janela)} janelas')

# Valores-p e q já calculados pelos scripts de modelo.
significancia = {}
for chave in ('logreg', 'xgboost'):
    with open(RESULTS / f'experimento_8_{chave}.json', encoding='utf-8') as arq:
        dados = json.load(arq)
    significancia[chave] = {j['idade_max']: j['significativo']
                            for j in dados['janelas']}

idades = [j['idade_max'] for j in janelas]
rotulos = [rotulo_janela(idade, i == 0) for i, idade in enumerate(idades)]
posicoes = np.arange(len(idades))

# Janela decisiva: a de melhor acurácia da logística, entre as significativas.
candidatas = [m for m in coleta['logreg'] if significancia['logreg'][m['idade_max']]]
janela_decisiva = max(candidatas or coleta['logreg'],
                      key=lambda m: m['acuracia_media'])['idade_max']
indice_decisiva = idades.index(janela_decisiva)
print(f'Janela decisiva: {janela_decisiva} dias\n')


# =============================================================================
# FIGURA 1 - acurácia por janela
# =============================================================================
figura, eixo = plt.subplots(figsize=(10.6, 5.6))

largura = 0.38
for deslocamento, (nome, _, cor, chave) in zip((-largura / 2, largura / 2), MODELOS):
    valores = np.array([m['acuracia_media'] for m in coleta[chave]])
    desvios = np.array([m['acuracia_dp'] for m in coleta[chave]])
    eixo.bar(posicoes + deslocamento, valores, largura * 0.94,
             yerr=desvios, capsize=3, label=nome, color=cor,
             edgecolor=FUNDO, linewidth=2,
             error_kw={'ecolor': TINTA_FRACA, 'lw': 1, 'alpha': 0.7})

eixo.axhline(0.5, color=TINTA_FRACA, ls='--', lw=1.2)
# O rótulo do acaso fica FORA da área de plotagem, à direita: dentro dela ele
# cairia em cima das barras de alguma janela.
eixo.annotate('acaso\n50%', xy=(1.008, 0.5), xycoords=('axes fraction', 'data'),
              ha='left', va='center', fontsize=9, color=TINTA_FRACA,
              annotation_clip=False, linespacing=1.3)

# Rótulo direto só nas janelas que sobreviveram ao teste estatístico -
# número em cima de toda barra vira ruído.
for indice, idade in enumerate(idades):
    for deslocamento, (_, _, _, chave) in zip((-largura / 2, largura / 2), MODELOS):
        if significancia[chave][idade]:
            metrica = coleta[chave][indice]
            altura = metrica['acuracia_media'] + metrica['acuracia_dp']
            eixo.annotate(f"{metrica['acuracia_media'] * 100:.0f}%",
                          xy=(indice + deslocamento, altura), xytext=(0, 4),
                          textcoords='offset points', ha='center',
                          fontsize=9, color=TINTA, fontweight='bold')

eixo.set_xticks(posicoes)
eixo.set_xticklabels(rotulos)
eixo.set_xlim(-0.62, len(idades) - 0.38)
eixo.set_ylim(0.40, 0.75)
eixo.set_yticks(np.arange(0.40, 0.76, 0.05))
eixo.set_yticklabels([f'{v:.0%}' for v in np.arange(0.40, 0.76, 0.05)])
eixo.set_xlabel('medidas disponíveis para treinar — cada barra é um modelo '
                'diferente, treinado só com as idades até ali')
eixo.set_ylabel('acurácia no conjunto de teste (30%)')
eixo.set_title('Quanto o modelo acerta o sexo, conforme mais idades são medidas\n'
               f"{info['n_apos_balanceamento']} aves balanceadas · "
               f"{info['n_treino']} treino / {info['n_teste']} teste · "
               f"média de {N_REPETICOES} divisões (barra = ± 1 desvio)",
               fontsize=11.5, loc='left')
eixo.legend(loc='upper left', frameon=False, fontsize=9.5)
limpar_eixo(eixo)
figura.tight_layout()
figura.savefig(FIGS / 'experimento_8_acuracia.png', dpi=150)
plt.close(figura)
print('  figura 1/4: acurácia por janela')


# =============================================================================
# FIGURA 2 - matriz de confusão na janela decisiva
# =============================================================================
figura, eixos = plt.subplots(1, 2, figsize=(11, 4.9))

for eixo, (nome, _, cor, chave) in zip(eixos, MODELOS):
    matriz = np.array(coleta[chave][indice_decisiva]['confusao_somada'])
    total_por_linha = matriz.sum(axis=1, keepdims=True)
    proporcao = matriz / total_por_linha

    eixo.imshow(proporcao, cmap=RAMPA_AZUL, vmin=0.25, vmax=0.75)
    # Fio da cor do fundo separando as células, para que os quatro quadrantes
    # se leiam como blocos distintos e não como uma mancha contínua.
    eixo.set_xticks([0.5], minor=True)
    eixo.set_yticks([0.5], minor=True)
    eixo.grid(which='minor', color=FUNDO, lw=3)
    eixo.tick_params(which='minor', length=0)

    for linha in range(2):
        for coluna in range(2):
            quantidade = matriz[linha, coluna]
            percentual = proporcao[linha, coluna]
            # Texto claro sobre célula escura, escuro sobre célula clara.
            tinta = '#ffffff' if percentual > 0.55 else TINTA
            eixo.text(coluna, linha - 0.09, f'{percentual:.0%}',
                      ha='center', va='center', fontsize=21,
                      fontweight='bold', color=tinta)
            eixo.text(coluna, linha + 0.20, f'{quantidade} aves',
                      ha='center', va='center', fontsize=10, color=tinta)

    eixo.set_xticks([0, 1])
    eixo.set_xticklabels(['previu Fêmea', 'previu Macho'])
    eixo.set_yticks([0, 1])
    eixo.set_yticklabels(['é Fêmea', 'é Macho'])
    eixo.tick_params(length=0)
    for lado in eixo.spines.values():
        lado.set_visible(False)

    acuracia = np.trace(matriz) / matriz.sum()
    eixo.set_title(f'{nome}\nacerta {acuracia:.0%} das aves',
                   fontsize=11, color=cor, fontweight='bold', pad=12)

figura.suptitle(f'Onde cada modelo erra — janela de {janela_decisiva} dias\n'
                f'percentual dentro de cada linha; somadas as '
                f'{N_REPETICOES} divisões de teste',
                fontsize=11.5, y=1.02, x=0.02, ha='left')
figura.tight_layout()
figura.savefig(FIGS / 'experimento_8_matriz_confusao.png', dpi=150,
               bbox_inches='tight')
plt.close(figura)
print('  figura 2/4: matriz de confusão')


# =============================================================================
# FIGURA 3 - curva ROC na janela decisiva
# =============================================================================
figura, eixo = plt.subplots(figsize=(7.6, 6.4))

eixo.plot([0, 1], [0, 1], ls='--', lw=1.2, color=TINTA_FRACA,
          label='acaso (AUC 0,50)')

for nome, _, cor, chave in MODELOS:
    metrica = coleta[chave][indice_decisiva]
    fpr, tpr, _ = roc_curve(metrica['y_real'], metrica['y_escore'])
    auc = roc_auc_score(metrica['y_real'], metrica['y_escore'])
    eixo.plot(fpr, tpr, lw=2, color=cor,
              label=f'{nome} (AUC {auc:.2f})'.replace('.', ','))

eixo.set_xlabel('fêmeas classificadas como macho (erro)')
eixo.set_ylabel('machos classificados corretamente (acerto)')
eixo.set_title(f'Curva ROC na janela de {janela_decisiva} dias\n'
               'quanto mais a curva sobe para o canto superior esquerdo, '
               'melhor o modelo',
               fontsize=11.5, loc='left')
eixo.set_xlim(0, 1)
eixo.set_ylim(0, 1)
eixo.set_aspect('equal')
eixo.legend(loc='lower right', frameon=False, fontsize=9.5)
limpar_eixo(eixo)
eixo.grid(color=GRADE, lw=0.8)
figura.tight_layout()
figura.savefig(FIGS / 'experimento_8_curva_roc.png', dpi=150)
plt.close(figura)
print('  figura 3/4: curva ROC')


# =============================================================================
# FIGURA 4 - o que cada modelo olha
# =============================================================================
QUANTAS = 12
janela_melhor = janelas[indice_decisiva]

coeficientes = mod_logreg.coeficientes_medios(
    janela_melhor['X'], janela_melhor['y']).head(QUANTAS).iloc[::-1]
importancias = mod_xgboost.importancias_medias(
    janela_melhor['X'], janela_melhor['y']).head(QUANTAS).iloc[::-1]

figura, (esquerda, direita) = plt.subplots(1, 2, figsize=(12.5, 6.0))

# Painel esquerdo: coeficientes têm SINAL, então a cor codifica a direção.
cores = [COR_ACERTO if v > 0 else COR_ERRO
         for v in coeficientes['coeficiente_medio']]
esquerda.barh(range(len(coeficientes)), coeficientes['coeficiente_medio'],
              color=cores, edgecolor=FUNDO, linewidth=1.5, height=0.72)
esquerda.axvline(0, color=TINTA_FRACA, lw=1)
esquerda.set_yticks(range(len(coeficientes)))
esquerda.set_yticklabels(coeficientes['feature'], fontsize=9)
esquerda.set_xlabel('coeficiente padronizado')
esquerda.set_title('Regressão Logística L2\n'
                   'azul empurra para Macho · vermelho para Fêmea',
                   fontsize=11, color=COR_LOGREG, fontweight='bold', loc='left')
for lado in ('top', 'right', 'left'):
    esquerda.spines[lado].set_visible(False)
esquerda.set_axisbelow(True)
esquerda.grid(axis='x', color=GRADE, lw=0.8)
esquerda.tick_params(length=0)

# Painel direito: importância não tem sinal, só magnitude -> uma cor só.
direita.barh(range(len(importancias)), importancias['importancia_media'],
             color=COR_XGBOOST, edgecolor=FUNDO, linewidth=1.5, height=0.72)
direita.set_yticks(range(len(importancias)))
direita.set_yticklabels(importancias['feature'], fontsize=9)
direita.set_xlabel('importância por ganho')
direita.set_title('XGBoost\nmede o quanto ajudou a separar, sem dizer a direção',
                  fontsize=11, color=COR_XGBOOST, fontweight='bold', loc='left')
for lado in ('top', 'right', 'left'):
    direita.spines[lado].set_visible(False)
direita.set_axisbelow(True)
direita.grid(axis='x', color=GRADE, lw=0.8)
direita.tick_params(length=0)
direita.set_xlim(0, importancias['importancia_media'].max() * 1.08)
esquerda.margins(x=0.08)

figura.suptitle(f'Quais medidas decidem o sexo — janela de {janela_decisiva} dias\n'
                'sufixos: _valor = quanto mede · _delta = quanto cresceu desde '
                'o 1º dia · _inclinacao = velocidade de crescimento',
                fontsize=11.5, y=1.04, x=0.02, ha='left')
figura.tight_layout()
figura.savefig(FIGS / 'experimento_8_features.png', dpi=150, bbox_inches='tight')
plt.close(figura)
print('  figura 4/4: medidas mais usadas')


# =============================================================================
# CÓPIA PARA A PASTA DO EXPERIMENTO
# =============================================================================
SAIDA_LOCAL.mkdir(parents=True, exist_ok=True)
nomes = ['experimento_8_acuracia.png', 'experimento_8_matriz_confusao.png',
         'experimento_8_curva_roc.png', 'experimento_8_features.png']
for nome_arquivo in nomes:
    shutil.copy(FIGS / nome_arquivo, SAIDA_LOCAL / nome_arquivo)

# Tabela de apoio, para quem quiser os números em vez da figura.
tabela = pd.DataFrame([
    {
        'janela': rotulos[i],
        'idade_max_dias': idades[i],
        'acuracia_logreg': coleta['logreg'][i]['acuracia_media'],
        'desvio_logreg': coleta['logreg'][i]['acuracia_dp'],
        'auc_logreg': coleta['logreg'][i]['auc_media'],
        'acuracia_xgboost': coleta['xgboost'][i]['acuracia_media'],
        'desvio_xgboost': coleta['xgboost'][i]['acuracia_dp'],
        'auc_xgboost': coleta['xgboost'][i]['auc_media'],
        'significativo_logreg': significancia['logreg'][idades[i]],
        'significativo_xgboost': significancia['xgboost'][idades[i]],
    }
    for i in range(len(idades))
])
tabela.to_csv(SAIDA_LOCAL / 'experimento_8_tabela.csv', index=False,
              sep=';', decimal='.')
tabela.to_csv(RESULTS / 'experimento_8_tabela.csv', index=False,
              sep=';', decimal='.')

print()
print(f'4 figuras + tabela em results/figures/ e em '
      f'src/experimento_8/resultados/')
