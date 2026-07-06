"""
Figura: Análise Discriminante Canônica (ADC) das medidas biométricas por sexo.

Diferente do PCA (não-supervisionado, maximiza variância), a ADC é SUPERVISIONADA:
procura a combinação linear das medidas que MAXIMIZA a separação entre Macho e
Fêmea. É o "melhor caso" para o sexo se separar — se nem ela separar, é a prova
mais forte de que o sexo não está codificado nas medidas (confirma o ML).

Como há 2 grupos, a ADC produz apenas 1 função canônica (CAN1); a figura mostra a
distribuição dos escores canônicos dos dois sexos (sobreposição = não separa).

Detalhes: PESO + 12 morfométricas, padronizadas. PCA/ADC são sensíveis a outliers,
então aplicamos em memória as 12 correções de digitação já validadas
(ver dataset-decimal-errors). Acurácia validada com split por ANIMAL (GroupKFold).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GroupKFold
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES = ROOT / 'results' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_style('whitegrid')

FEATURES = ['PESO', 'BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA', 'DORSO',
            'VENTRE', 'CIRCFABDOM', 'SOBRECOXA', 'COXA', 'CANELA', 'UNHAMAIOR']
CORRECOES = [
    (181, 115, 'CIRCFCABECA', 12.7), (49, 38, 'CIRCFCABECA', 10.5),
    (109, 66, 'VENTRE', 37.5), (125, 21, 'DORSO', 21.0),
    (162, 80, 'ASA', 11.7), (121, 66, 'COXA', 11.1),
    (206, 66, 'BICO', 2.3), (82, 66, 'BICO', 2.1), (10, 66, 'BICO', 2.0),
    (203, 66, 'BICO', 2.0), (233, 28, 'BICO', 1.7), (179, 21, 'BICO', 1.14),
]
PALETTE = {'Macho': '#4C72B0', 'Femea': '#C44E52'}

# =============================================================================
# DADOS (+ correções) e ADC
# =============================================================================
df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')
for col in FEATURES + ['IDADE', 'ANIMAL']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for a, i, col, v in CORRECOES:
    df.loc[(df['ANIMAL'] == a) & (df['IDADE'] == i), col] = v

data = df[df['SEXO'].isin(['Macho', 'Femea'])].dropna(subset=FEATURES + ['SEXO', 'ANIMAL']).copy()

X = StandardScaler().fit_transform(data[FEATURES])
y = data['SEXO'].values
lda = LinearDiscriminantAnalysis()
z = lda.fit(X, y).transform(X)[:, 0]
# Orienta para Macho à direita (sinal é arbitrário)
if z[y == 'Macho'].mean() < z[y == 'Femea'].mean():
    z = -z
data['CAN1'] = z

# Métricas de separação
mean_m, mean_f = z[y == 'Macho'].mean(), z[y == 'Femea'].mean()
grand = z.mean()
ss_between = (np.sum(y == 'Macho') * (mean_m - grand) ** 2 +
              np.sum(y == 'Femea') * (mean_f - grand) ** 2)
eta2 = ss_between / np.sum((z - grand) ** 2)  # 0 = sem separação, 1 = perfeita

pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
cv_acc = cross_val_score(pipe, data[FEATURES], y, groups=data['ANIMAL'],
                         cv=GroupKFold(5), scoring='accuracy').mean()
baseline = data['SEXO'].value_counts(normalize=True).max()

# =============================================================================
# FIGURA - densidade dos escores canônicos por sexo
# =============================================================================
fig, ax = plt.subplots(figsize=(9.5, 6))
for sx in ['Macho', 'Femea']:
    sns.kdeplot(x=z[y == sx], fill=True, alpha=0.45, linewidth=2,
                color=PALETTE[sx], label='Macho' if sx == 'Macho' else 'Fêmea', ax=ax)
ax.axvline(mean_m, color=PALETTE['Macho'], ls='--', lw=1.5, alpha=0.9)
ax.axvline(mean_f, color=PALETTE['Femea'], ls='--', lw=1.5, alpha=0.9)
ax.set_xlabel('Função discriminante canônica (CAN1)', fontsize=11)
ax.set_ylabel('Densidade', fontsize=11)
ax.set_title('Análise Discriminante Canônica — Machos × Fêmeas',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
# Foca na massa de dados (erros nao corrigidos geram escores extremos).
lo, hi = np.quantile(z, 0.003), np.quantile(z, 0.997)
pad = (hi - lo) * 0.10
ax.set_xlim(lo - pad, hi + pad)

plt.tight_layout()
out = FIGURES / 'adc_sexo.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)

# =============================================================================
# CONSOLE
# =============================================================================
print("=" * 62)
print("ANÁLISE DISCRIMINANTE CANÔNICA (Macho x Femea)")
print("=" * 62)
print(f"  Amostras: {len(data)} (Macho={int((y=='Macho').sum())}, Femea={int((y=='Femea').sum())})")
print(f"  Separação (eta² da CAN1): {eta2:.3f}  -> {eta2*100:.1f}% da variância do eixo é por sexo")
print(f"  Correlação canônica (sqrt eta²): {np.sqrt(eta2):.3f}")
print(f"  Acurácia ADC (split por animal): {cv_acc:.3f}  |  baseline (classe maior): {baseline:.3f}")
print(f"  Ganho sobre baseline: {(cv_acc-baseline)*100:+.1f} pontos percentuais")
print("-" * 62)
print(f"Figura salva: results/figures/{out.name}")
print("=" * 62)
