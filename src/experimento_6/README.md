# Experimento 6 — Importância das Variáveis + Curva ROC/AUC

Gera, com XGBoost e split por animal:
1. **Importância das variáveis para a regressão do peso** (XGBoost Regressor).
2. **Importância das variáveis para a classificação do sexo** (XGBoost Classifier).
3. **Curva ROC + AUC** da classificação do sexo.

## Script
| Arquivo | Tarefa |
|---------|--------|
| `experimento_6.py` | Treina os dois XGBoost, extrai a importância (gain) e gera ROC/AUC |

- **Features (peso):** `IDADE` + 12 morfométricas (alvo = `PESO`).
- **Features (sexo):** `PESO` + `IDADE` + 12 morfométricas (alvo = `SEXO`).
- Busca de hiperparâmetros: `RandomizedSearchCV` (`KFold` 5 p/ regressão, `StratifiedKFold` 5
  p/ classificação, otimizando `roc_auc`). Desbalanceamento via `scale_pos_weight`.

## Resultados-chave (o contraste é a mensagem)
- **Peso — sinal forte e concentrado:** `CIRCFABDOM` domina (~0,50 da importância), seguida
  por `IDADE` (~0,19), `CIRCFCABECA` (~0,10) e `DORSO` (~0,09). R² ≈ 0,94.
- **Sexo — sem sinal:** a importância fica **achatada** (todas as 14 variáveis entre ~0,064 e
  ~0,092, perto do uniforme 1/14 ≈ 0,071). Nenhuma variável se destaca.
- **ROC/AUC (sexo):** AUC Teste ≈ **0,60** (CV ≈ 0,57) — pouco acima do aleatório (0,50).
  A curva quase encosta na diagonal, confirmando a baixa capacidade discriminatória.

> Leitura: o **mesmo conjunto de medidas** carrega sinal forte para peso (uma variável
> explica metade) e praticamente nenhum sinal para sexo — reforço visual de que a
> galinha-d'angola é **monomórfica**. (Atenção: os eixos X das duas figuras de importância
> têm escalas diferentes — peso vai até ~0,5; sexo até ~0,09.)

## Como rodar
```bash
uv run python src/experimento_6/experimento_6.py
```

## Saídas
- `results/figures/experimento_6_importancia_peso.png`
- `results/figures/experimento_6_importancia_sexo.png`
- `results/figures/experimento_6_roc_sexo.png`
- `results/experimento_6.json` — importâncias, métricas e AUC.
