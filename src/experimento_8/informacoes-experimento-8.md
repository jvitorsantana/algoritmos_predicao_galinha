# Experimento 8: predição de sexo por janelas de idade

**Informações metodológicas**

---

**1. O que o experimento quer descobrir?**

A partir de que idade é possível acertar o sexo da ave olhando só as medidas do
corpo.

O treino começa usando apenas as medidas do 1º dia de vida. Depois entram as
medidas do 1º dia mais as da idade seguinte. Depois essas mais as da próxima, e
assim por diante. Cada conjunto desses é uma janela, e cada janela gera um
modelo separado. A comparação entre os modelos mostra a partir de quando a
previsão começa a funcionar.

**2. Em qual software foi feito?**

Python 3.11.15, com o ambiente gerenciado pelo *uv*.

**3. Quais bibliotecas foram usadas?**

*pandas* 3.0.1 e *numpy* 2.4.3 para tratar os dados. *scikit-learn* 1.8.0 para
a regressão logística, a padronização, a divisão treino/teste e as métricas
(`LogisticRegression`, `StandardScaler`, `Pipeline`, `train_test_split`,
`roc_auc_score`, `accuracy_score`, `confusion_matrix`, `roc_curve`). *xgboost*
3.2.0 para o XGBoost (`XGBClassifier`). *matplotlib* 3.10.8 para as figuras.

**4. Cada linha analisada é uma ave ou uma medição?**

Uma ave. Cada animal aparece uma vez só, e o sexo dele é a resposta que o
modelo tenta acertar.

Nos experimentos 3, 4, 6 e 7 era diferente: cada medição valia como uma linha,
então a mesma ave aparecia de 9 a 11 vezes, e a idade entrava como variável.
Aqui a idade não é mais variável, ela é o que define a janela.

**5. Como as medidas viram variáveis do modelo?**

Cada uma das 13 medidas (peso mais as 12 morfométricas) gera três números
dentro de uma janela:

- quanto a ave mede na última idade da janela;
- quanto ela cresceu desde o 1º dia;
- com que velocidade ela cresceu no período.

A primeira janela tem uma idade só, então ali existe apenas o valor medido: são
13 variáveis. Todas as outras janelas têm 39.

O resumo foi adotado porque juntar todas as idades lado a lado daria 117
colunas para 92 aves de treino, ou seja, mais colunas do que animais. Nessa
situação qualquer modelo decora os dados em vez de aprender.

**6. Quais aves entraram na análise?**

As que têm medida completa em todas as idades usadas, num total de 173 animais
(107 machos e 66 fêmeas). São sempre as mesmas 173 aves em todas as janelas.

Isso é necessário para as janelas poderem ser comparadas entre si. Se cada
janela usasse quem tivesse dado disponível, o número de aves mudaria junto com
a quantidade de informação, e não seria possível saber qual dos dois causou a
diferença no resultado.

O critério olha apenas se a ave foi medida. Nunca olha o sexo dela.

**7. Por que só 132 aves, se os outros experimentos usaram todas?**

A redução acontece em dois passos.

**De 234 para 173 aves.** Nos outros experimentos, cada medição vale como uma
amostra, então uma ave medida em três idades já contribui com três linhas. Aqui
não funciona assim: para calcular o quanto a ave cresceu e com que velocidade,
é necessária a série completa dela. Saíram 61 aves, 26% do total, por duas
razões diferentes:

- **50 aves faltaram a pelo menos uma pesagem.** Destas, 40 faltaram a uma
  única das nove idades.
- **11 aves compareceram a todas as nove pesagens**, mas têm uma célula em
  branco em algum registro, e isso basta para a série ficar incompleta. Em 10
  delas é uma única medida que falta, quase sempre o peso ou uma medida isolada
  como pescoço, canela ou unha.

A ausência de aves não é concentrada em uma idade específica. A maior é aos 80
dias, com 21 aves ausentes, e a menor aos 7 dias, com 3.

**De 173 para 132 aves.** Sobram 107 machos e 66 fêmeas, e as classes são
igualadas em 66 de cada sexo.

Três pontos importantes sobre essa comparação:

1. **Nenhum macho é descartado de vez.** O sorteio é refeito nas 30 repetições,
   então todos os 107 machos entram em alguma delas. O que muda é quais deles
   compõem cada repetição.

2. **Os outros experimentos também tinham 234 aves, não 2.271.** Como o sexo da
   ave não muda com o tempo, aqueles 2.271 registros eram a mesma resposta
   repetida de 9 a 11 vezes por animal. Repetir a mesma ave não cria informação
   nova. Então a diferença real entre os experimentos é de 234 para 173 aves.

3. **Exigir a série completa é condição da pergunta.** O experimento compara
   janelas entre si, e isso só faz sentido se o grupo de animais for o mesmo em
   todas elas.

**8. O que foi corrigido nos dados?**

Tudo foi corrigido em código, sem alterar o arquivo original.

- **Identificadores com asterisco.** Cinco registros vêm marcados com asterisco
  (`*181`, `*190`, `*208`), que é uma anotação de campo indicando re-medição. O
  asterisco foi removido, porque `181` e `*181` são a mesma ave. Sem isso elas
  seriam contadas como dois animais.
- **Medidas faltando.** Registros com alguma medida em branco foram
  descartados, já que a análise exige a ave medida por completo. São poucos
  registros, mas eles custam 11 aves que compareceram a todas as pesagens.
- **Idade de 35 dias.** Ficou de fora, porque tem só 4 registros contra cerca
  de 220 das outras idades.
- **Medições repetidas.** Em 7 casos a mesma ave foi medida duas vezes na mesma
  idade. Foi mantido o primeiro registro e descartado o segundo. É o mesmo
  critério já usado no experimento 7, o que mantém os dois trabalhos
  comparáveis, e o primeiro registro é a leitura anotada antes, na ordem
  original da planilha.

**9. Quais idades foram usadas?**

0, 7, 14, 21, 28, 38, 52, 66 e 80 dias, sendo a idade 0 o primeiro dia de vida.

As idades de 101 e 115 dias ficaram fora. A de 101 dias tem 116 registros e a
de 115 dias tem 150, contra 211 a 230 das outras. Como as janelas são
cumulativas e o grupo de aves é fixo, exigir presença numa idade mal coberta
reduz o grupo em todas as janelas, inclusive nas primeiras: com os 101 dias o
grupo cai de 173 para 91 aves, e com as duas idades cai para 84. Uma análise
complementar até 115 dias, pulando os 101, é possível com 113 animais e está
pendente.

**10. As classes foram balanceadas?**

Sim, por sorteio do mesmo número de machos e de fêmeas: 66 de cada, 132 no
total.

Sem isso, um modelo que respondesse "macho" para todas as aves acertaria 62% e
pareceria razoável. Com as classes iguais, quem chuta acerta 50%, e aí o número
do modelo pode ser lido direto.

**11. Como foi a divisão entre treino e teste?**

70% para treino (92 aves) e 30% para teste (40 aves), mantendo metade de cada
sexo nos dois lados.

A divisão é repetida 30 vezes, com sorteios diferentes, e o resultado
apresentado é a média com o desvio. A repetição foi adotada porque, com 40 aves
no teste, uma divisão sozinha varia muito: trocar o sorteio mexe bastante no
resultado. A média de 30 divisões mede o modelo, não a sorte do sorteio.

**12. Os dados foram padronizados?**

Sim, para a regressão logística, usando z-score (`StandardScaler`, média 0 e
desvio 1). É preciso porque o peso vai de cerca de 30 g a 3.000 g enquanto o
bico fica entre 1 e 2 cm. Sem padronizar, o peso dominaria o modelo só por ser
um número maior.

A padronização acontece dentro de um `Pipeline`, então a média e o desvio são
calculados apenas sobre o treino de cada repetição. Padronizar tudo antes de
dividir passaria informação do teste para o treino.

No XGBoost não há padronização. Ele decide por cortes do tipo "peso maior que
1.200", e um corte desses não muda de lugar se a variável for reescalada.

**13. Quais modelos foram usados, e com quais configurações?**

**Regressão logística com penalidade L2:** `C=1.0`, `penalty='l2'`,
`max_iter=5000`.

**XGBoost:** `n_estimators=200`, `max_depth=3`, `learning_rate=0.05`,
`subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`.

As configurações foram definidas antes de rodar, pensando no tamanho da amostra
(92 aves de treino para até 39 variáveis). Não houve busca de melhores valores.
Testar várias opções e ficar com a melhor, usando os mesmos dados que reportam
o resultado, deixaria o desempenho melhor do que ele realmente é.

Não foi usado peso de classe, porque a amostra já chega equilibrada pelo
sorteio do item 10.

**14. Por que dois modelos, e justamente esses?**

Porque eles funcionam de formas diferentes. A regressão logística soma as
variáveis de maneira simples. O XGBoost monta árvores e consegue capturar
combinações mais complicadas entre as medidas. A comparação entre os dois
responde se vale a pena um modelo mais complexo aqui.

A regressão logística foi escolhida como modelo principal por três motivos: com
39 variáveis para 92 aves, ela é a mais adequada; seus coeficientes mostram
quais medidas pesam e para que lado; e ela devolve probabilidade, não só uma
nota. O XGBoost é o modelo usado nos experimentos 1, 2, 3, 5, 6 e 7, então
mantê-lo liga este experimento aos anteriores.

**15. Quais medidas de desempenho foram usadas?**

AUC, acurácia, acerto entre os machos e acerto entre as fêmeas, sempre medidos
no conjunto de teste e apresentados como média com desvio das 30 repetições.
Como as classes estão equilibradas, quem chuta fica em 0,500 tanto em AUC
quanto em acurácia.

A matriz de confusão soma as 30 repetições, num total de 600 fêmeas e 600
machos avaliados.

**16. Como foi evitado que o modelo "visse a resposta" antes da hora?**

1. **Cada ave aparece uma vez só.** Assim é impossível o mesmo animal estar no
   treino e no teste ao mesmo tempo.
2. **A padronização é calculada apenas no treino** de cada repetição, dentro do
   `Pipeline`.
3. **As variáveis vêm só da própria ave**, e só de idades que já estão na
   janela. Nada usa medida futura, nada usa média do grupo, nada é derivado do
   sexo.
4. **O grupo de aves foi montado pela disponibilidade de medida**, nunca pelo
   sexo.
5. **As configurações dos modelos não foram ajustadas** nos dados que reportam
   o resultado.
6. **O modelo do teste estatístico foi escolhido antes de ver os resultados**,
   para que o teste não fosse aplicado só ao que deu certo.

**17. Como foram obtidos os coeficientes e as importâncias?**

Pela média das 30 repetições, usando o mesmo sorteio e a mesma divisão. A média
é bem mais confiável que um treino isolado, porque com poucas aves um treino
sozinho pode eleger uma variável por acaso.

Os coeficientes da logística têm sinal, então dizem para que lado a medida
empurra. As importâncias do XGBoost não têm sinal: dizem o quanto a medida
ajudou a separar, mas não para qual lado.

Nos 66 dias, os dois modelos escolheram a mesma variável em primeiro lugar: a
velocidade de crescimento da unha maior. No XGBoost, as três primeiras posições
são todas dessa mesma medida. Depois vêm a coxa e a circunferência de cabeça.
São dois modelos bem diferentes chegando ao mesmo lugar.
