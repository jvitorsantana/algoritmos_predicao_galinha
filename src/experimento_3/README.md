# Experimento 3 — Comparação sistemática de modelos (split por animal)

Compara **múltiplos algoritmos** para as duas tarefas, usando o dataset completo com
**separação por animal** (`random_state=42`): nenhuma observação de um mesmo indivíduo
aparece em treino e teste ao mesmo tempo.

## Scripts
| Arquivo | Tarefa |
|---------|--------|
| `comparacao_peso.py` | **Peso** — 10 regressores (RF, Extra Trees, GB, XGBoost, LightGBM, KNN, Ridge, Lasso, ElasticNet, SVR) |
| `comparacao_sexo.py` | **Sexo** — 8 classificadores (GB, RF, Extra Trees, XGBoost, LightGBM, SVM, KNN, Reg. Logística) |

## Resultados-chave (relatório, Tabelas 4 e 5)
- **Peso:** melhor = **Random Forest** (R²test=0,9453; RMSE=107,1 g; MAE=68,6 g).
  Todos os modelos ficam entre 0,916 e 0,945. O R² global é **inflado pela IDADE**:
  intra-idade o R² colapsa (≈0 ou negativo aos 0–7 dias).
- **Sexo:** melhor = **Gradient Boosting** (F1test=0,6546), mas a acurácia (0,5784) fica
  **abaixo do baseline de maioria (0,5894)** — nenhum algoritmo supera o chute trivial.

## Como rodar
```bash
uv run python src/experimento_3/comparacao_peso.py
uv run python src/experimento_3/comparacao_sexo.py
```
Saídas em `results/` (`comparacao_peso.json`, `comparacao_sexo.json`, figuras).
