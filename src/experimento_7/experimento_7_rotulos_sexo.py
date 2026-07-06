"""
Experimento 7 - Efeito da limpeza de dados na classificação de sexo.

Pergunta: o baixo desempenho da classificação de sexo (AUC ~0,60) é causado por
RUÍDO DE RÓTULO/DADOS ou por AUSÊNCIA DE SINAL nas medidas morfométricas?

Compara duas versões dos MESMOS dados, com validação cruzada por animal
(GroupKFold, para não vazar registros de uma ave entre treino e validação):

  A) BRUTO  - dados originais: sexo inconsistente dentro da ave (~12% dos
              registros), pares (ANIMAL, IDADE) duplicados e erros de vírgula.
  B) LIMPO  - mesmas medidas, mas com: (i) 12 erros de vírgula corrigidos
              (valor ÷ 10), (ii) sexo por MAIORIA (uma ave = um sexo) e
              (iii) deduplicação dos pares (ANIMAL, IDADE) pela média das medidas.

Fonte: data/raw/dataset_original.csv (bruto pristino). Todas as correções são
aplicadas EM CÓDIGO aqui, então o experimento é auto-contido e reproduzível.

Leitura:
  - Se a limpeza NÃO melhorar o AUC  -> o sinal não existe nas medidas
    (o rótulo ruidoso não era o gargalo). Conclusão do projeto se mantém.
  - Se melhorar muito -> o rótulo era o problema e a conclusão deve ser revista.
"""
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

MORF = ['BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA', 'DORSO', 'VENTRE',
        'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']
FEAT = ['PESO', 'IDADE'] + MORF

# 12 correções de vírgula validadas: (ANIMAL, IDADE, coluna, valor_correto = ÷10)
CORRECOES_VIRGULA = [
    (179, 21, 'BICO', 1.14), (233, 28, 'BICO', 1.7), (10, 66, 'BICO', 2.0),
    (82, 66, 'BICO', 2.1), (203, 66, 'BICO', 2.0), (206, 66, 'BICO', 2.3),
    (49, 38, 'CIRCFCABECA', 10.5), (181, 115, 'CIRCFCABECA', 12.7),
    (162, 80, 'ASA', 11.7), (125, 21, 'DORSO', 21.0),
    (109, 66, 'VENTRE', 37.5), (121, 66, 'COXA', 11.1),
]


def carregar_bruto():
    """Carrega o dataset bruto pristino (prefere dataset_original.csv)."""
    raw = ROOT / 'data' / 'raw' / 'dataset_original.csv'
    if not raw.exists():
        raw = ROOT / 'data' / 'raw' / 'dataset.csv'
        print("AVISO: dataset_original.csv não encontrado; usando dataset.csv. "
              "Se ele já estiver limpo, as versões A e B ficarão parecidas.")
    df = pd.read_csv(raw, sep=';', decimal='.', encoding='utf-8')
    # ID limpo (remove asterisco de anotação: *181 == 181)
    df['ANIMAL'] = pd.to_numeric(df['ANIMAL'].astype(str).str.replace('*', '', regex=False),
                                 errors='coerce')
    for c in FEAT:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def limpeza_base(df):
    """Filtros comuns aos modelos: idade com >=10 amostras, sem sexo, sem medida."""
    vc = df['IDADE'].value_counts()
    df = df[df['IDADE'].isin(vc[vc >= 10].index)]
    return df.dropna(subset=['SEXO']).dropna(subset=FEAT + ['ANIMAL']).copy()


def avaliar(d, nome):
    """AUC e acurácia por validação cruzada agrupada por animal."""
    y = (d['SEXO'] == 'Macho').astype(int)
    X, g = d[FEAT], d['ANIMAL']
    cv = GroupKFold(n_splits=5)
    xgb = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, n_jobs=-1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    res = {
        'n_registros': int(len(d)),
        'n_aves': int(d['ANIMAL'].nunique()),
        'n_macho': int(y.sum()), 'n_femea': int((1 - y).sum()),
        'baseline_acc': round(float(max(y.mean(), 1 - y.mean())), 4),
        'xgb_auc': round(float(cross_val_score(xgb, X, y, groups=g, cv=cv, scoring='roc_auc').mean()), 4),
        'xgb_acc': round(float(cross_val_score(xgb, X, y, groups=g, cv=cv, scoring='accuracy').mean()), 4),
        'logreg_auc': round(float(cross_val_score(lr, X, y, groups=g, cv=cv, scoring='roc_auc').mean()), 4),
    }
    print(f"{nome}")
    print(f"   n={res['n_registros']} registros | {res['n_aves']} aves | "
          f"M/F={res['n_macho']}/{res['n_femea']}")
    print(f"   XGBoost: AUC={res['xgb_auc']:.3f}  Acc={res['xgb_acc']:.3f}  "
          f"(baseline acc={res['baseline_acc']:.3f})")
    print(f"   LogReg : AUC={res['logreg_auc']:.3f}")
    return res


# =============================================================================
# EXECUÇÃO
# =============================================================================
print("=" * 70)
print("EXPERIMENTO 7 - limpeza de dados x classificação de sexo (bruto vs limpo)")
print("=" * 70)

df = carregar_bruto()

# --- inconsistência de sexo (só para relatar o tamanho do problema) ---
inc = df.dropna(subset=['SEXO']).groupby('ANIMAL')['SEXO'].nunique()
n_inc = int((inc > 1).sum())
print(f"Aves com sexo inconsistente no bruto: {n_inc} de {df['ANIMAL'].nunique()} "
      f"({n_inc / df['ANIMAL'].nunique() * 100:.0f}%)\n")

# --- Versão A: BRUTO ---
A = limpeza_base(df)
res_A = avaliar(A, "A) BRUTO (rótulos ruidosos + duplicatas + erros de vírgula)")
print()

# --- Versão B: LIMPO (vírgula + sexo por maioria + dedup) ---
B = df.copy()
for animal, idade, col, val in CORRECOES_VIRGULA:          # (i) vírgula
    B.loc[(B['ANIMAL'] == animal) & (B['IDADE'] == idade), col] = val
B = limpeza_base(B)
maioria = B.groupby('ANIMAL')['SEXO'].agg(lambda s: s.value_counts().index[0])
B['SEXO'] = B['ANIMAL'].map(maioria)                        # (ii) sexo por maioria
B = B.groupby(['ANIMAL', 'IDADE'], as_index=False).agg(     # (iii) dedup por média
    {**{f: 'mean' for f in ['PESO'] + MORF}, 'SEXO': 'first'})
res_B = avaliar(B, "B) LIMPO (vírgula + sexo por maioria + deduplicado)")

# =============================================================================
# CONCLUSÃO
# =============================================================================
d_xgb = res_B['xgb_auc'] - res_A['xgb_auc']
d_lr = res_B['logreg_auc'] - res_A['logreg_auc']
print("=" * 70)
print(f"Delta AUC (LIMPO - BRUTO):  XGBoost {d_xgb:+.3f} | LogReg {d_lr:+.3f}")
veredito = ("Limpeza NÃO melhorou -> sinal ausente nas medidas (rótulo não era o gargalo)."
            if abs(d_xgb) < 0.03 else
            "Limpeza MUDOU o AUC -> o rótulo/dados afetavam o resultado; revisar conclusão.")
print(veredito)
print("=" * 70)

out = {
    'experimento': 'experimento_7',
    'descricao': 'classificacao de sexo: dados brutos vs limpos (virgula + sexo por maioria + dedup)',
    'validacao': 'GroupKFold(5) por ANIMAL',
    'aves_sexo_inconsistente_bruto': n_inc,
    'A_bruto': res_A,
    'B_limpo': res_B,
    'delta_auc_xgb': round(float(d_xgb), 4),
    'delta_auc_logreg': round(float(d_lr), 4),
    'veredito': veredito,
}
with open(RESULTS / 'experimento_7.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"Resultados salvos em: results/experimento_7.json")
