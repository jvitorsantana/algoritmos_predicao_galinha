"""
Figura: Heatmap de correlação entre todas as medidas biométricas.

Mostra quais medidas estão correlacionadas. Variáveis = PESO + as 12 medidas
morfométricas (IDADE fica de fora por ser o eixo do desenho, não uma medida
corporal).

Método: correlação de SPEARMAN (por postos). Escolha deliberada para ESTE
dataset porque:
  - há erros de digitação documentados (vírgula decimal perdida -> valor ×10,
    ex.: Circ. Cabeça=127, Ventre=375, Bico=23). O Spearman usa postos, então
    esses extremos quase não distorcem; o Pearson seria puxado por eles.
  - medidas de crescimento têm relação monotônica mas não-linear e distribuição
    assimétrica (idades agregadas) — situação em que o Spearman é mais adequado.
A comparação Pearson×Spearman mostrou que corrigir os erros aproxima o Pearson
do Spearman; usando Spearman, não é preciso mexer nos dados brutos.

Leitura esperada: como todas as medidas crescem com o tamanho do corpo (e com a
idade), tendem a ser fortemente correlacionadas entre si.
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

# Medidas biométricas (ordem: peso + morfométricas) e rótulos curtos em PT.
BIOMETRIC = ['PESO', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA', 'DORSO',
             'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']
LABELS = {
    'PESO': 'Peso', 'BICO': 'Bico', 'CIRCFCABECA': 'Circ. Cabeça',
    'PESCOCO': 'Pescoço', 'ASA': 'Asa', 'TULIPA': 'Tulipa', 'DORSO': 'Dorso',
    'VENTRE': 'Ventre', 'CIRCFABDOM': 'Circ. Abdômen', 'SOBRECOXA': 'Sobrecoxa',
    'COXA': 'Coxa', 'CANELA': 'Canela', 'UNHAMAIOR': 'Unha Maior',
}
METHOD = 'spearman'

# =============================================================================
# DADOS + MATRIZ DE CORRELAÇÃO
# =============================================================================
df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')
for col in BIOMETRIC:
    df[col] = pd.to_numeric(df[col], errors='coerce')

corr = df[BIOMETRIC].corr(method=METHOD)
corr_disp = corr.rename(index=LABELS, columns=LABELS)

# =============================================================================
# FIGURA - heatmap (triângulo inferior, valores anotados)
# =============================================================================
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_disp, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlação de Spearman (ρ)'},
            annot_kws={'size': 8}, ax=ax)
ax.set_title('Correlação entre as Medidas Biométricas',
             fontsize=14, fontweight='bold', pad=14)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
out = FIGURES / 'heatmap_correlacao.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)

# =============================================================================
# RESUMO NO CONSOLE - pares mais e menos correlacionados
# =============================================================================
pairs = []
cols = corr.columns
for i in range(len(cols)):
    for j in range(i):
        pairs.append((LABELS[cols[i]], LABELS[cols[j]], corr.iloc[i, j]))
pairs.sort(key=lambda t: abs(t[2]), reverse=True)

print("=" * 60)
print(f"CORRELAÇÃO ENTRE MEDIDAS BIOMÉTRICAS (Spearman)")
print("=" * 60)
print("\nTop 10 pares MAIS correlacionados:")
for a, b, r in pairs[:10]:
    print(f"  {a:<14} x {b:<14}  rho = {r:+.3f}")
print("\n5 pares MENOS correlacionados:")
for a, b, r in pairs[-5:]:
    print(f"  {a:<14} x {b:<14}  rho = {r:+.3f}")
print("-" * 60)
print(f"Figura salva: results/figures/{out.name}")
print("=" * 60)
