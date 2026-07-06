# Perguntas e Respostas — Predição Biométrica de Galinhas

> Registro de perguntas e respostas sobre o projeto (análise CRISP-DM: predição de
> peso e classificação de sexo a partir de medidas morfométricas).
>
> Atualizado em: 2026-06-28

---

## 1) Em qual software foram realizadas a Correlação de Spearman, a PCA e a Análise Discriminante?

Todas as análises foram feitas em Python 3.11 (ambiente gerenciado com `uv`), usando o ecossistema científico da linguagem; não foram usados R, SAS ou SPSS. A correlação de Spearman foi calculada com o pandas, e a PCA e a Análise Discriminante com o scikit-learn.

## 2) Quais bibliotecas/funções foram utilizadas?

A correlação de Spearman foi obtida com o pandas (`DataFrame.corr(method='spearman')`). A padronização usou o `StandardScaler`, a PCA o `PCA` e a Análise Discriminante o `LinearDiscriminantAnalysis`, todos do scikit-learn; a acurácia foi validada com `cross_val_score` e `GroupKFold`, também do scikit-learn. A manipulação dos dados foi feita com pandas e numpy, e as figuras com matplotlib e seaborn.

## 3) Os dados foram padronizados antes da PCA?

Sim. Aplicou-se padronização z-score (`StandardScaler`: média 0 e desvio-padrão 1) em todas as variáveis antes da PCA, porque o Peso está em gramas (escala 0–2000) e as demais medidas em centímetros (escala ~1–45); sem isso, o Peso dominaria a análise apenas por ter maior variância numérica. A mesma padronização foi aplicada antes da Análise Discriminante.

## 4) A PCA foi baseada na matriz de correlação ou de covariância?

Foi baseada na matriz de correlação. Como as variáveis foram padronizadas (z-score) antes da PCA, a decomposição equivale à da matriz de correlação das variáveis originais.

## 5) Quantos componentes principais foram retidos e qual critério foi utilizado?

Foram retidos 2 componentes (PC1 e PC2), pelo critério de visualização bidimensional da separação por sexo; não se aplicou um critério formal de retenção (como Kaiser/autovalor > 1 ou *scree plot*). A PC1 explicou 83,2% da variância e a PC2 3,7% (86,9% acumulados) — a PC1 sozinha já domina, refletindo a forte multicolinearidade entre as medidas, todas ligadas ao tamanho/idade.

## 6) A Análise Discriminante corresponde à LDA (Linear Discriminant Analysis) ou à Análise Discriminante Canônica?

São a mesma coisa — LDA e Análise Discriminante Canônica são dois nomes pra mesma técnica. Usei a `LinearDiscriminantAnalysis` do scikit-learn, e o que ela calcula já é a função discriminante canônica. Como aqui só tem 2 grupos (Macho e Fêmea), sai uma função só (a CAN1); por isso o gráfico mostra as duas distribuições nessa linha, em vez do gráfico de elipses, que só aparece com 3 grupos ou mais.

## 7) A correlação de Spearman foi calculada utilizando todas as observações ou separadamente por idade?

Das duas formas, em figuras distintas. Na versão agregada, com todas as observações juntas, a correlação média (fora da diagonal) ficou em torno de 0,90; na versão separada por idade (11 idades), a correlação média caiu para algo entre 0,07 e 0,44. A comparação mostra que a alta correlação agregada é, em boa parte, efeito comum da idade/tamanho, e não relação direta entre as medidas.

## 8) As figuras (correlação, PCA e análise discriminante) foram produzidas em Python? Se sim, quais bibliotecas foram utilizadas?

Sim, todas em Python, com matplotlib (backend Agg) e seaborn: os heatmaps de correlação com `seaborn.heatmap`, a dispersão da PCA com `matplotlib.scatter`, a densidade dos escores da Análise Discriminante com `seaborn.kdeplot` e os boxplots por sexo com `seaborn.boxplot`.

## 9) Na correlação de Spearman, foram considerados apenas os coeficientes (ρ) ou também foi avaliada a significância estatística (p < 0,05)?

Foram considerados apenas os coeficientes (ρ); o heatmap não inclui teste de significância. Vale notar que, com os tamanhos amostrais envolvidos (n ≈ 2.300 na versão agregada e n ≈ 116–232 por idade), praticamente todos os coeficientes — sobretudo os mais fortes — seriam significativos a p < 0,05, já que com n grande mesmo correlações fracas atingem significância. Testes de significância (Mann-Whitney U) foram usados em outra análise, a dos boxplots por sexo.

## 10) A figura da "Análise Discriminante Canônica" foi gerada pela classe `LinearDiscriminantAnalysis` do scikit-learn ou por uma Análise Discriminante Canônica (CDA/CVA)?

Foi gerada pela classe `LinearDiscriminantAnalysis` do scikit-learn (a projeção do `.transform()`), e não por uma rotina dedicada de CDA/CANDISC. Na matemática dá no mesmo: a LDA de Fisher e a Análise Discriminante Canônica resolvem o mesmo problema e produzem os mesmos eixos, então os escores são de fato as variáveis canônicas. Mas, como a ferramenta usada foi a LDA do scikit-learn, o mais correto e seguro para o artigo é chamar de "Análise Discriminante Linear (LDA)" e nomear o eixo como "LD1" (o nome "Can1", da Análise Discriminante Canônica, é a convenção de pacotes como o PROC CANDISC do SAS). Portanto, a alteração proposta está certa.

## 11) Sobre os registros duplicados encontrados (ex.: Animal 128 e Animal 4).

O `data/raw/dataset.csv` não tem coluna de data, então não dá para cruzar pelas datas citadas (elas estão na planilha original); aqui a checagem é por (ANIMAL, IDADE). Cruzando assim, existem 5 pares duplicados: (181, 14 dias), (128, 0), (4, 21), (56, 21) e (9, 21). Nos três casos de 21 dias, as duas linhas do mesmo animal ainda trazem sexos diferentes. Esses duplicados não foram removidos antes do treinamento — os scripts dos modelos não fazem deduplicação. Há ainda três animais (181, 208, 190) com registros marcados por asterisco na coluna ANIMAL (`*181`, `*208`, `*190`), provável anotação de coleta; contados como texto, eles inflam a contagem para 238, mas o número real de animais é 235 (IDs de 1 a 235, sem falhas). Além disso, há um problema maior de fundo: o sexo é inconsistente dentro do mesmo animal em 170 dos 234 animais com sexo (73%), somando 284 registros (12,4%) com o sexo "minoritário" (provável erro), concentrados nas idades 21 (118), 28 (71) e 0 (53). Como uma ave não muda de sexo, isso indica erro de registro (ou que o identificador ANIMAL não é estável entre sessões). Na prática, os modelos de sexo foram treinados com cerca de 12% de rótulos errados, o que é um confundidor relevante para o baixo desempenho da classificação de sexo.

## 12) Qual foi o número final de machos e fêmeas utilizado no treinamento dos modelos?

No classificador de sexo, depois da limpeza (idades com ≥10 amostras, remoção de registros sem sexo e de registros com alguma medida faltando), restaram 2.278 registros: 1.335 Macho e 943 Fêmea, de 234 animais. Importante distinguir os dois níveis: por medição/registro (o que o modelo treina) são 1.335 Macho e 943 Fêmea; por ave individual, resolvendo o sexo de cada animal pela maioria dos seus registros, são 136 Macho e 98 Fêmea (234 aves, sem empates). O rótulo de sexo, porém, tem ~12% de ruído (ver pergunta 11).

## 13) Houve exclusão de animais ou registros antes do treinamento dos modelos?

Sim, mas apenas dos registros incompletos — não dos duplicados. Foram removidos 21 registros no total (2.299 → 2.278): 4 por pertencerem a uma idade com menos de 10 amostras (a sessão de 35 dias), 2 por não terem sexo e 15 por terem alguma medida faltando. Com isso, o conjunto passou de 235 para 234 animais (o animal 123 saiu por inteiro, pois seu único registro estava sem sexo). Já os 5 pares duplicados (ANIMAL, IDADE) permaneceram no treinamento — havia 5 antes e 5 depois da limpeza, ou seja, não houve deduplicação —, assim como os ~284 registros de sexo inconsistente (12,4%) e os identificadores com asterisco. A divisão treino/teste foi feita por animal (80/20).

## 14) O animal 123 permaneceu sem informação de sexo ou foi corrigido?

Permaneceu sem sexo. O animal 123 tem um único registro (idade 0, peso 25) com o campo SEXO vazio (NaN); não foi corrigido e, por isso, foi excluído do modelo de sexo na etapa que remove registros sem sexo. (O animal 124 também tem um registro sem sexo, na idade 80, com o peso em branco.)

## 15) Foi utilizada exatamente a planilha do dataset ou uma versão tratada?

Os modelos usaram exatamente o `data/raw/dataset.csv`, como está (cru), aplicando apenas conversão numérica das colunas e os filtros de limpeza descritos acima, tudo em código — não houve versão externa tratada, e as correções de erro de digitação (vírgula) identificadas nesta análise não foram aplicadas aos modelos. A única exceção são as minhas figuras de PCA e Análise Discriminante desta sessão, que usaram as 12 correções aplicadas em memória (sem alterar o arquivo); os heatmaps de correlação usaram o dado cru.

---
