# Experimento 1 — Modelos individuais por faixa etária

Um modelo treinado **separadamente para cada idade** (0, 7, 14, 21, 28, 52, 66, 80, 101, 115 dias).
Busca de hiperparâmetros via `RandomizedSearchCV` com validação cruzada.

## Scripts
| Arquivo | Tarefa |
|---------|--------|
| `xgb_peso_treino.py` | Predição de **peso** (XGBoost Regressor) — treino |
| `xgb_peso_teste.py`  | Predição de **peso** — teste / geração de predições |
| `svm_treino.py`      | Classificação de **sexo** (SVM) — treino |
| `svm_teste.py`       | Classificação de **sexo** (SVM) — teste |

## Resultados-chave (relatório, Tabelas 2 e 3)
- **Peso (XGBoost):** melhor desempenho aos 66 dias (R²CV=0,82; MAE=37,4 g; %MAE=3,8%).
  Fase crítica aos 21–28 dias (R²CV ≈ 0 ou negativo). `PESO_ANTERIOR` é o preditor dominante a partir dos 14 dias.
- **Sexo (SVM):** F1CV ≈ 0,70–0,75 em todas as idades, próximo do baseline (~58% Machos).
  O modelo tende a prever tudo como Macho; único resultado equilibrado aos 101 dias (acc 74%).

## Como rodar
```bash
uv run python src/experimento_1/xgb_peso_treino.py
uv run python src/experimento_1/xgb_peso_teste.py
uv run python src/experimento_1/svm_treino.py
uv run python src/experimento_1/svm_teste.py
```
