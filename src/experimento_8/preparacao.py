"""
Experimento 8 - preparação dos dados e protocolo de avaliação.

Este módulo é compartilhado pelos dois classificadores (LogReg-L2 e XGBoost).
Ele existe por um motivo específico: se cada script montasse os dados por conta
própria, bastaria uma diferença pequena entre eles - um filtro a mais, outra
forma de sortear a amostra - para que a comparação entre os dois modelos
deixasse de ser justa. Carregando tudo daqui, os dois enxergam exatamente as
mesmas aves, as mesmas features e as mesmas divisões treino/teste.

Os dois scripts que usam este módulo:
    experimento_8_logreg.py
    experimento_8_xgboost.py

O dataset em disco NÃO é alterado. Todas as correções acontecem em memória,
aqui dentro, e estão comentadas uma a uma na função carregar_dataset().
"""
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


# =============================================================================
# CAMINHOS E CONSTANTES
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / 'results'
FIGS = RESULTS / 'figures'
MODELS = RESULTS / 'models' / 'experimento_8'

# As 12 medidas morfométricas coletadas com paquímetro/fita.
MORFOMETRICAS = ['BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA', 'DORSO',
                 'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA',
                 'UNHAMAIOR']

# O peso entra junto com elas: são 13 medidas base por ave, por idade.
MEDIDAS = ['PESO'] + MORFOMETRICAS

# Idades usadas. A de 35 dias ficou de fora porque tem apenas 4 registros,
# e a de 101 dias porque tem 116 (as demais têm ~220): exigir presença nela
# derrubaria a coorte de 173 para 84 aves.
IDADES = [0, 7, 14, 21, 28, 38, 52, 66, 80]

# Quantas vezes o sorteio 70/30 é repetido. Uma divisão só seria uma loteria:
# com 40 aves no teste, trocar a semente move o AUC em quase 0,1. Repetir e
# tirar a média mede o modelo, não a sorte do sorteio.
N_REPETICOES = 30

# Proporção do teste, conforme o padrão adotado no projeto.
FRACAO_TESTE = 0.30

SEMENTE = 42


# =============================================================================
# PARTE 1 - DADOS
# =============================================================================
def carregar_dataset():
    """Lê o dataset e aplica as correções necessárias, sem gravar em disco.

    Devolve um DataFrame com uma linha por (ave, idade) e as colunas
    ANIMAL, IDADE, SEXO e as 13 medidas.
    """
    caminho = ROOT / 'data' / 'raw' / 'dataset.csv'
    df = pd.read_csv(caminho, sep=';', decimal='.', encoding='utf-8')

    # Correção 1: alguns IDs vêm marcados com asterisco ('*181'). O asterisco é
    # uma anotação de campo indicando re-medição, não uma ave diferente - a ave
    # 181 e a '*181' são a mesma. Sem remover, elas virariam dois animais.
    ids_sem_asterisco = df['ANIMAL'].astype(str).str.replace('*', '', regex=False)
    df['ANIMAL'] = pd.to_numeric(ids_sem_asterisco, errors='coerce')

    # Correção 2: colunas numéricas podem conter texto residual da digitação.
    for coluna in ['IDADE'] + MEDIDAS:
        df[coluna] = pd.to_numeric(df[coluna], errors='coerce')

    # Correção 3: registro com qualquer medida faltando não serve, porque a
    # coorte exige a ave medida por completo em todas as idades da análise.
    df = df.dropna(subset=['ANIMAL', 'SEXO'] + MEDIDAS)

    # Correção 4: a idade de 35 dias tem só 4 registros - cobertura insuficiente.
    df = df[df['IDADE'] != 35]

    # Correção 5: existem 7 pares (ANIMAL, IDADE) duplicados, ou seja, a mesma
    # ave medida duas vezes na mesma idade. Mantemos o PRIMEIRO registro e
    # descartamos os demais, que é o mesmo critério já adotado no experimento 7.
    # A ordem preservada é a do arquivo original, então "primeiro" é a leitura
    # anotada antes.
    df = df.drop_duplicates(subset=['ANIMAL', 'IDADE'], keep='first')

    return df.reset_index(drop=True)


def construir_coorte(df, idades=IDADES):
    """Lista das aves que têm medida completa em TODAS as idades da análise.

    Essa coorte é FIXA: as mesmas aves são usadas em todas as janelas. Se cada
    janela usasse quem tivesse dado disponível, o número de aves mudaria junto
    com a janela e a curva final misturaria dois efeitos - mais informação e
    mais (ou menos) aves. Ficando fixa, a única coisa que varia é a informação.

    O filtro olha apenas quais idades a ave tem medida; nunca olha o SEXO.
    Por isso não há risco de vazamento aqui.
    """
    idades_por_ave = df.groupby('ANIMAL')['IDADE'].apply(set)
    idades_exigidas = set(idades)
    tem_tudo = idades_por_ave.apply(lambda tem: idades_exigidas.issubset(tem))
    return sorted(idades_por_ave[tem_tudo].index)


def montar_features(df, animais, janela):
    """Monta a matriz de features de uma janela, com UMA LINHA POR AVE.

    'janela' é a lista de idades disponíveis até o momento, por exemplo
    [0], depois [0, 7], depois [0, 7, 14], e assim por diante.

    Para cada uma das 13 medidas, criamos um bloco de 3 colunas:

        <medida>_valor       quanto a ave mede na última idade da janela
        <medida>_delta       o quanto ela cresceu desde o 1º dia
        <medida>_inclinacao  a velocidade de crescimento na janela
                             (coeficiente angular da reta ajustada aos pontos)

    A razão de resumir em vez de empilhar todas as idades lado a lado é
    dimensional: empilhando, a janela de 80 dias teria 13 x 9 = 117 colunas
    para ~130 aves, mais colunas do que amostras. Com o bloco resumido são
    sempre 39 colunas, independente do tamanho da janela.

    Na primeira janela existe uma idade só, então delta e inclinação não fazem
    sentido - ali ficam apenas os 13 valores.
    """
    idades = sorted(janela)
    recorte = df[df['ANIMAL'].isin(animais) & df['IDADE'].isin(idades)]
    colunas = {}

    for medida in MEDIDAS:
        # Reorganiza para o formato "uma linha por ave, uma coluna por idade".
        tabela = recorte.pivot_table(index='ANIMAL', columns='IDADE', values=medida)
        # reindex garante a mesma ordem de aves e de idades em todas as medidas.
        tabela = tabela.reindex(index=animais, columns=idades)
        valores = tabela.to_numpy(dtype=float)

        colunas[f'{medida}_valor'] = valores[:, -1]

        if len(idades) > 1:
            colunas[f'{medida}_delta'] = valores[:, -1] - valores[:, 0]
            # Uma reta por ave: np.polyfit(x, y, 1)[0] é o coeficiente angular.
            colunas[f'{medida}_inclinacao'] = [
                np.polyfit(idades, linha, 1)[0] for linha in valores
            ]

    X = pd.DataFrame(colunas, index=animais)

    # Alvo: 1 = Macho, 0 = Fêmea. O sexo é constante por ave, então basta o
    # primeiro registro de cada uma.
    sexo = df.groupby('ANIMAL')['SEXO'].first().reindex(animais)
    y = (sexo == 'Macho').astype(int).to_numpy()

    return X, y


# =============================================================================
# PARTE 2 - PROTOCOLO DE AVALIAÇÃO
# =============================================================================
def sortear_amostra_balanceada(y, sorteio):
    """Sorteia índices com o MESMO número de machos e de fêmeas.

    A coorte tem 107 machos e 66 fêmeas. Sem balancear, um classificador que
    chutasse 'macho' para todas as aves acertaria 62% e pareceria razoável.
    Igualando as classes (66 e 66), o acerto de quem chuta cai para 50% e a
    acurácia passa a ser lida diretamente.

    O sorteio é refeito a cada repetição, com machos diferentes. Assim, ao
    longo das 30 repetições, todos os machos acabam sendo usados em alguma
    delas - não descartamos 41 aves de forma permanente.
    """
    indices_macho = np.flatnonzero(y == 1)
    indices_femea = np.flatnonzero(y == 0)
    n = min(len(indices_macho), len(indices_femea))

    escolhidos = np.concatenate([
        sorteio.choice(indices_macho, size=n, replace=False),
        sorteio.choice(indices_femea, size=n, replace=False),
    ])
    return np.sort(escolhidos)


def avaliar_em_repeticoes(criar_modelo, X, y, n_repeticoes=N_REPETICOES,
                          semente=SEMENTE, guardar_confusao=False,
                          guardar_escores=False):
    """Roda o protocolo completo e devolve as métricas médias no TESTE.

    A cada repetição:
      1. sorteia uma amostra balanceada (mesmo nº de machos e fêmeas);
      2. divide em 70% treino e 30% teste, mantendo o balanço nos dois lados;
      3. treina um modelo NOVO e mede no teste.

    Dois cuidados contra vazamento:
      - 'criar_modelo' devolve um Pipeline novo a cada chamada, então o
        StandardScaler é ajustado só no treino daquela repetição. Normalizar a
        matriz inteira antes de dividir passaria média e desvio do teste para
        o treino.
      - Cada ave aparece uma única vez em X, então é impossível a mesma ave
        cair nos dois lados da divisão.
    """
    sorteio = np.random.default_rng(semente)
    X_array = X.to_numpy(dtype=float)

    aucs, acuracias, sensibilidades, especificidades = [], [], [], []
    confusao_total = np.zeros((2, 2), dtype=int)
    # Guarda o sexo real e o escore previsto de cada ave de teste, juntando
    # todas as repetições. Serve para desenhar uma curva ROC única no script
    # de gráficos, em vez de 30 curvas sobrepostas.
    escores_reais, escores_previstos = [], []

    for repeticao in range(n_repeticoes):
        selecionados = sortear_amostra_balanceada(y, sorteio)
        X_bal, y_bal = X_array[selecionados], y[selecionados]

        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X_bal, y_bal,
            test_size=FRACAO_TESTE,
            stratify=y_bal,            # mantém 50/50 no treino e no teste
            random_state=semente + repeticao,
        )

        modelo = criar_modelo()
        modelo.fit(X_treino, y_treino)

        # Escore contínuo para o AUC: probabilidade quando existe, senão a
        # distância até a fronteira de decisão.
        if hasattr(modelo, 'predict_proba'):
            escore = modelo.predict_proba(X_teste)[:, 1]
        else:
            escore = modelo.decision_function(X_teste)

        previsto = modelo.predict(X_teste)
        tn, fp, fn, tp = confusion_matrix(y_teste, previsto, labels=[0, 1]).ravel()

        aucs.append(roc_auc_score(y_teste, escore))
        acuracias.append(accuracy_score(y_teste, previsto))
        sensibilidades.append(tp / (tp + fn) if (tp + fn) else np.nan)  # machos
        especificidades.append(tn / (tn + fp) if (tn + fp) else np.nan)  # fêmeas

        if guardar_confusao:
            confusao_total += np.array([[tn, fp], [fn, tp]])
        if guardar_escores:
            escores_reais.append(y_teste)
            escores_previstos.append(escore)

    resultado = {
        'auc_media': round(float(np.mean(aucs)), 4),
        'auc_dp': round(float(np.std(aucs)), 4),
        'auc_min': round(float(np.min(aucs)), 4),
        'auc_max': round(float(np.max(aucs)), 4),
        'acuracia_media': round(float(np.mean(acuracias)), 4),
        'acuracia_dp': round(float(np.std(acuracias)), 4),
        'sensibilidade_macho': round(float(np.nanmean(sensibilidades)), 4),
        'especificidade_femea': round(float(np.nanmean(especificidades)), 4),
        'n_repeticoes': n_repeticoes,
    }
    if guardar_confusao:
        resultado['confusao_somada'] = confusao_total.tolist()  # [[TN,FP],[FN,TP]]
    if guardar_escores:
        resultado['y_real'] = np.concatenate(escores_reais)
        resultado['y_escore'] = np.concatenate(escores_previstos)
    return resultado


def teste_de_permutacao(criar_modelo, X, y, auc_observado, n_permutacoes,
                        n_repeticoes, semente=SEMENTE):
    """Qual a chance de obter este AUC se o sexo não tivesse relação com as medidas?

    Embaralhamos o rótulo e rodamos o MESMO protocolo. Repetindo muitas vezes,
    montamos a distribuição do AUC "sem sinal nenhum" e vemos onde o resultado
    real cai. Isso é necessário porque testamos 9 janelas: com 9 tentativas,
    uma delas passar de 0,6 por puro acaso é comum, e sem esse controle a gente
    anunciaria um achado que não existe.

    Devolve o valor-p: proporção de embaralhamentos que alcançaram um AUC tão
    alto quanto o observado. O +1 no numerador e no denominador é a correção
    padrão que impede o p de dar exatamente zero.
    """
    sorteio = np.random.default_rng(semente + 12345)
    tao_bons_quanto = 0

    for _ in range(n_permutacoes):
        y_embaralhado = sorteio.permutation(y)
        resultado = avaliar_em_repeticoes(
            criar_modelo, X, y_embaralhado,
            n_repeticoes=n_repeticoes,
            semente=int(sorteio.integers(0, 10_000_000)),
        )
        if resultado['auc_media'] >= auc_observado:
            tao_bons_quanto += 1

    return (tao_bons_quanto + 1) / (n_permutacoes + 1)


def benjamini_hochberg(valores_p, alfa=0.05):
    """Corrige os valores-p pelo número de janelas testadas (controle de FDR).

    São 9 janelas avaliadas a 5%. Sem correção, esperaríamos cerca de meia
    janela "significativa" só por sorte - exatamente o pico espúrio que o
    experimento precisa evitar declarar como descoberta. O método ordena os p,
    afrouxa a exigência conforme a posição e devolve um q comparável ao alfa.
    """
    p = np.asarray(valores_p, dtype=float)
    total = len(p)
    ordem = np.argsort(p)
    q = np.empty(total, dtype=float)

    # Percorre do maior p para o menor, impondo que o q nunca aumente.
    menor_ate_agora = 1.0
    for posicao_de_tras, indice in enumerate(ordem[::-1]):
        posicao = total - posicao_de_tras          # posição 1-based na ordenação
        menor_ate_agora = min(menor_ate_agora, p[indice] * total / posicao)
        q[indice] = menor_ate_agora

    q = np.minimum(q, 1.0)
    return q, q < alfa


# =============================================================================
# PARTE 3 - APOIO PARA OS SCRIPTS
# =============================================================================
def preparar_janelas():
    """Monta X e y de todas as janelas de uma vez.

    Devolve (lista_de_janelas, info), em que cada item da lista é um dicionário
    com a idade máxima, a matriz X e o alvo y. As features são construídas uma
    única vez aqui; as repetições do 70/30 apenas sorteiam linhas dessa matriz.
    """
    df = carregar_dataset()
    animais = construir_coorte(df)

    sexo = df.groupby('ANIMAL')['SEXO'].first().reindex(animais)
    n_macho = int((sexo == 'Macho').sum())
    n_femea = int((sexo == 'Femea').sum())
    n_balanceado = 2 * min(n_macho, n_femea)

    janelas = []
    for quantidade_de_idades in range(1, len(IDADES) + 1):
        idades_da_janela = IDADES[:quantidade_de_idades]
        X, y = montar_features(df, animais, idades_da_janela)
        janelas.append({
            'idade_max': idades_da_janela[-1],
            'idades': idades_da_janela,
            'X': X,
            'y': y,
        })

    info = {
        'n_aves_coorte': len(animais),
        'n_macho': n_macho,
        'n_femea': n_femea,
        'n_apos_balanceamento': n_balanceado,
        'n_treino': int(round(n_balanceado * (1 - FRACAO_TESTE))),
        'n_teste': n_balanceado - int(round(n_balanceado * (1 - FRACAO_TESTE))),
    }
    return janelas, info


def imprimir_cabecalho(titulo, info):
    print('=' * 78)
    print(titulo)
    print('=' * 78)
    print(f"Coorte fixa .........: {info['n_aves_coorte']} aves "
          f"({info['n_macho']} machos, {info['n_femea']} fêmeas)")
    print(f"Após balanceamento ..: {info['n_apos_balanceamento']} aves "
          f"({info['n_apos_balanceamento'] // 2} de cada sexo)")
    print(f"Divisão .............: {int((1 - FRACAO_TESTE) * 100)}% treino "
          f"({info['n_treino']} aves) / {int(FRACAO_TESTE * 100)}% teste "
          f"({info['n_teste']} aves)")
    print(f"Repetições ..........: {N_REPETICOES} sorteios independentes")
    print(f"Idades ..............: {IDADES}")
    print()
