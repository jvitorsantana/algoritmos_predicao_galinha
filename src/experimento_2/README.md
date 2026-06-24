# Experimento 2 — Dataset completo, foco em XGBoost

Usa o **conjunto de dados completo** com separação manual de 20% para teste (balanceada entre
machos e fêmeas). Foco no XGBoost, com análise da **importância das variáveis**.

## Scripts
| Arquivo | Tarefa |
|---------|--------|
| `xgb_sexo_treino.py` | Classificação de **sexo** (XGBoost) — treino + importância das features |
| `xgb_sexo_teste.py`  | Classificação de **sexo** (XGBoost) — teste |

> A predição de **peso** deste experimento (modelo XGBoost único sobre todas as idades,
> R²=0,941; RMSE=113,4 g) está em [`notebooks/xgboost_peso.ipynb`](../../notebooks/xgboost_peso.ipynb).

## Resultados-chave (relatório, seção 4.2)
- **Sexo:** mesmo o melhor modelo apresentou forte viés — a maioria das fêmeas foi
  classificada como macho. Alto recall para Macho às custas da identificação de Fêmeas.
- **Importância das variáveis (peso):** `CIRCFABDOM` (circunferência abdominal) é, de longe,
  a mais relevante, seguida por `DORSO`, `IDADE` e `CIRCFCABECA`.

## Como rodar
```bash
uv run python src/experimento_2/xgb_sexo_treino.py
uv run python src/experimento_2/xgb_sexo_teste.py
```
> Requer os splits pré-gerados em `data/processed/` (`sexo_treino.csv` etc.).
