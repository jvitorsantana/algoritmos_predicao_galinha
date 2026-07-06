"""
Figura(s): Heatmap de correlação entre as medidas biométricas, UM POR IDADE.

Complementa o heatmap agregado (heatmap_correlacao.py). Ao calcular a correlação
DENTRO de cada idade, removemos o efeito do crescimento: como todos os animais
têm a mesma idade, a correlação que sobra é a relação real entre as medidas, não
o fato de tudo crescer junto.

Esperado: valores bem menores que na versão agregada (lá a média fora da
diagonal é ~0,90; controlando a idade cai para ~0,22), porque a maior parte da
"correlação" agregada era apenas o tamanho/idade em comum.

Método: Spearman (por postos), robusto aos erros de digitação do dataset.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES = ROOT / 'results' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_style('white')

BIOMETRIC = ['PESO', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA', 'DORSO',
             'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']
LABELS = {
    'PESO': 'Peso', 'BICO': 'Bico', 'CIRCFCABECA': 'Circ. Cabeça',
    'PESCOCO': 'Pescoço', 'ASA': 'Asa', 'TULIPA': 'Tulipa', 'DORSO': 'Dorso',
    'VENTRE': 'Ventre', 'CIRCFABDOM': 'Circ. Abdômen', 'SOBRECOXA': 'Sobrecoxa',
    'COXA': 'Coxa', 'CANELA': 'Canela', 'UNHAMAIOR': 'Unha Maior',
}
METHOD = 'spearman'
MIN_N = 30  # idades com menos amostras (ex.: 35d com n=4) sao puladas

# =============================================================================
# DADOS
# =============================================================================
df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')
for col in BIOMETRIC + ['IDADE']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

mask = np.triu(np.ones(len(BIOMETRIC), dtype=bool), k=1)
ages = sorted(a for a in df['IDADE'].dropna().unique())

print("=" * 60)
print("HEATMAP DE CORRELAÇÃO POR IDADE (Spearman, controla a idade)")
print("=" * 60)
print(f"{'idade':>6} | {'n':>4} | {'corr. média (fora diag.)':>24}")
print("-" * 45)

saved = []
for age in ages:
    sub = df[df['IDADE'] == age]
    n = len(sub)
    if n < MIN_N:
        print(f"{int(age):>5}d | {n:>4} | PULADA (n < {MIN_N})")
        continue

    corr = sub[BIOMETRIC].corr(method=METHOD)
    od_mean = corr.where(~np.eye(len(BIOMETRIC), dtype=bool)).stack().mean()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr.rename(index=LABELS, columns=LABELS), mask=mask, annot=True,
                fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Correlação de Spearman (ρ)'},
                annot_kws={'size': 8}, ax=ax)
    ax.set_title(f'Correlação entre as Medidas Biométricas\n'
                 f'Idade {int(age)} dias  (n = {n})',
                 fontsize=14, fontweight='bold', pad=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    out = FIGURES / f'heatmap_correlacao_idade_{int(age):03d}d.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(out.name)
    print(f"{int(age):>5}d | {n:>4} | {od_mean:>24.2f}")

print("-" * 45)
print(f"{len(saved)} imagens salvas em results/figures/ (heatmap_correlacao_idade_*d.png)")
print("=" * 60)
