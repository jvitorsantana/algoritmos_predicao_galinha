# Experimento 5 — XGBoost para peso, base completa, apenas features mais impactantes

Treina um **único modelo XGBoost** para predizer o peso usando **toda a base** (sem separar
por faixa etária) e mantendo **apenas as variáveis mais impactantes**. A `IDADE` é
**excluída por padrão** (`INCLUDE_IDADE = False`): o modelo prevê o peso a partir **somente
das medidas do corpo**, sem saber a idade da ave.

## Abordagem (seleção de features data-driven)
1. Treina o XGBoost com as **12 variáveis morfométricas** (IDADE fora) e ranqueia por
   importância de *gain*.
2. Seleciona o menor conjunto cuja **importância acumulada ≥ 95%**.
3. **Retreina** o XGBoost apenas com as features selecionadas (busca de hiperparâmetros).
4. Compara *todas* vs *selecionadas* para confirmar a paridade de desempenho.

> Para reativar a idade como preditor e comparar, basta `INCLUDE_IDADE = True` no script.

Convenções do projeto: split **por animal** (80/20, `random_state=42`), `RandomizedSearchCV`
com `KFold` (5 folds), separador `;` e decimal `.`.

## Script
| Arquivo | Tarefa |
|---------|--------|
| `experimento_5_peso.py` | Regressão de peso (XGBoost) + seleção de features + figuras |

## Resultados-chave (sem IDADE)
- **7 features selecionadas** (somam ≥95% da importância): `CIRCFABDOM`, `CIRCFCABECA`,
  `SOBRECOXA`, `DORSO`, `ASA`, `UNHAMAIOR`, `VENTRE`.
- **5 descartadas:** `BICO`, `PESCOCO`, `TULIPA`, `COXA`, `CANELA`.
- **Paridade:** R²test 0,9388 (7 features) vs 0,9391 (12 features); MAE 73,7 g vs 73,8 g.
  Cortar ~42% das variáveis **não piora** o modelo → modelo mais simples e barato de medir.
- **CIRCFABDOM** sozinha responde por ~54% da importância; com `CIRCFCABECA` chega a ~75%.
- **Achado importante:** remover a `IDADE` quase não mexeu no R² global (0,9418 → 0,9388;
  MAE 69,8 → 73,7 g). As medidas do corpo já **codificam o tamanho/idade**, então a idade é
  praticamente redundante para a predição global — e o peso pode ser estimado sem conhecê-la.
- **R² global × intra-idade (o ponto central):** R² **global = 0,939**, mas a média do
  **R² por idade ≈ −0,73**. O gráfico `experimento_5_peso.png` rotula cada idade: negativo
  nas iniciais (0: −5,06; 7: −1,88; puxado por elas), com pico aos **38 (0,52) e 52 dias
  (0,51)**. Sem a IDADE, o erro intra-idade dos primeiros dias piora — o modelo perde a
  âncora etária. Confirma que o R² global mede a curva de crescimento, não a variação
  individual dentro de cada idade.

## Como rodar
```bash
uv run python src/experimento_5/experimento_5_peso.py
```

## Saídas
- `results/experimento_5_peso.json` — métricas, features selecionadas, importâncias, por idade.
- `results/figures/experimento_5_peso.png` — real vs predito, resíduos, R²/MAE por idade.
- `results/figures/experimento_5_features.png` — importância das variáveis + todas vs. selecionadas.
- `results/models/experimento_5/modelo_peso_exp5.json` — modelo XGBoost treinado.
