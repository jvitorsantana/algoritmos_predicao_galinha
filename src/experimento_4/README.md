# Experimento 4 — Sexo a partir de features de crescimento

Ideia diferente dos anteriores: em vez das medidas do corpo, tenta classificar o sexo
olhando **como o peso evolui ao longo do tempo** em cada ave. As features usam apenas
dados passados do próprio animal (sem vazamento), com separação por animal e SMOTE.

## Script
| Arquivo | Tarefa |
|---------|--------|
| `experimento_4_sexo.py` | Classificação de **sexo** com 8 modelos sobre features de trajetória |

## Features de crescimento (relatório, Tabela 6)
`PESO_PRED_MA` (média móvel dos 3 últimos pesos), `PESO_RESID`, `GANHO_PESO`,
`TAXA_CRESC`, `TAXA_CRESC_MA`, `PESO_ACUM_MA`, `SLOPE_IA` (inclinação peso×idade).

## Resultados-chave (relatório, Tabela 7)
- Melhor = **Extra Trees** (F1test=0,564; acc=0,558).
- As curvas de crescimento de machos e fêmeas ficam **praticamente sobrepostas**.
- **Conclusão central:** a galinha-d'angola é **monomórfica** — nem medidas corporais
  nem features de crescimento bastam para distinguir o sexo de forma confiável.

## Como rodar
```bash
uv run python src/experimento_4/experimento_4_sexo.py
```
Saídas em `results/experimento_4_sexo.json` e `results/figures/experimento_4_sexo.png`.
