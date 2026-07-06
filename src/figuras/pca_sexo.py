"""
Figura: PCA das medidas biométricas, colorida por sexo (Macho × Fêmea).

Objetivo: ver se os sexos se separam no espaço das medidas. Pela análise do
projeto (classificador de sexo com AUC ~0,60, boxplots sobrepostos), espera-se
que NÃO se separem — os pontos devem ficar misturados.

Detalhes:
  - Variáveis: PESO + 12 morfométricas, padronizadas (StandardScaler) antes do
    PCA — obrigatório, pois as escalas são muito diferentes (Peso em g, resto cm).
  - PCA é sensível a outliers, então (só aqui) aplicamos em memória as 12
    correções de erro de digitação já validadas (ver dataset-decimal-errors).
  - Espera-se PC1 ≈ tamanho/idade (tudo cresce junto); o sexo aparece misturado
    ao longo de todo o espaço.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES = ROOT / 'results' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

FEATURES = ['PESO', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA', 'DORSO',
            'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']
# 12 correções de vírgula validadas (animal, idade, coluna, valor_correto)
CORRECOES = [
    (181, 115, 'CIRCFCABECA', 12.7), (49, 38, 'CIRCFCABECA', 10.5),
    (109, 66, 'VENTRE', 37.5), (125, 21, 'DORSO', 21.0),
    (162, 80, 'ASA', 11.7), (121, 66, 'COXA', 11.1),
    (206, 66, 'BICO', 2.3), (82, 66, 'BICO', 2.1), (10, 66, 'BICO', 2.0),
    (203, 66, 'BICO', 2.0), (233, 28, 'BICO', 1.7), (179, 21, 'BICO', 1.14),
]
PALETTE = {'Macho': '#4C72B0', 'Femea': '#C44E52'}

# =============================================================================
# DADOS (+ correções em memória) e PCA
# =============================================================================
df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')
for col in FEATURES + ['IDADE', 'ANIMAL']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for a, i, col, v in CORRECOES:
    df.loc[(df['ANIMAL'] == a) & (df['IDADE'] == i), col] = v

data = df[df['SEXO'].isin(['Macho', 'Femea'])].dropna(subset=FEATURES + ['SEXO']).copy()

X = StandardScaler().fit_transform(data[FEATURES])
pca = PCA(n_components=2, random_state=42)
pc = pca.fit_transform(X)
# Orienta PC1 para "maior = mais pesado" (sinal do PCA é arbitrário)
if np.corrcoef(pc[:, 0], data['PESO'])[0, 1] < 0:
    pc[:, 0] *= -1
data['PC1'], data['PC2'] = pc[:, 0], pc[:, 1]
ev = pca.explained_variance_ratio_ * 100
corr_pc1_idade = np.corrcoef(data['PC1'], data['IDADE'])[0, 1]

# =============================================================================
# FIGURA - scatter PC1 x PC2 (● Macho cheio, ○ Fêmea vazado)
# =============================================================================
fig, ax = plt.subplots(figsize=(9.5, 7.5))
m = data[data['SEXO'] == 'Macho']
f = data[data['SEXO'] == 'Femea']
ax.scatter(m['PC1'], m['PC2'], s=20, c=PALETTE['Macho'], alpha=0.55,
           edgecolors='none', label='Macho')
ax.scatter(f['PC1'], f['PC2'], s=24, facecolors='none', edgecolors=PALETTE['Femea'],
           alpha=0.75, linewidths=0.8, label='Fêmea')

ax.set_xlabel(f'PC1 ({ev[0]:.1f}%)  —  ≈ tamanho/idade', fontsize=11)
ax.set_ylabel(f'PC2 ({ev[1]:.1f}%)', fontsize=11)
ax.set_title('PCA das Medidas Biométricas — Machos × Fêmeas',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
ax.axvline(0, color='gray', lw=0.6, alpha=0.5)

# Limita os eixos à massa de dados (alguns erros ainda não corrigidos -
# Tulipa/Sobrecoxa/Pescoço - geram pontos extremos que esticariam a escala).
def _lim(s):
    lo, hi = s.quantile(0.001), s.quantile(0.999)
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad
ax.set_xlim(*_lim(data['PC1']))
ax.set_ylim(*_lim(data['PC2']))

plt.tight_layout()
out = FIGURES / 'pca_sexo.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)

# =============================================================================
# CONSOLE
# =============================================================================
print("=" * 60)
print("PCA DAS MEDIDAS BIOMÉTRICAS (Macho x Femea)")
print("=" * 60)
print(f"  Amostras: {len(data)} (Macho={len(m)}, Femea={len(f)})")
print(f"  Variância explicada: PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  (soma={ev.sum():.1f}%)")
print(f"  Correlação PC1 x IDADE = {corr_pc1_idade:+.3f}  (confirma PC1 ~ tamanho/idade)")
print("-" * 60)
print(f"Figura salva: results/figures/{out.name}")
print("=" * 60)
