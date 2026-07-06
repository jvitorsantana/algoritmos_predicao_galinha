"""
Figura(s): Boxplots HORIZONTAIS por sexo (Macho × Fêmea), uma imagem por variável.

Apenas as principais variáveis para a comparação do sexo. O classificador de
sexo (experimento_3/comparacao_sexo e experimento_6) usa as 14 features, mas a
importância para sexo é praticamente uniforme (galinha-d'angola ~monomórfica);
entre as morfométricas, as de maior peso coincidem com as escolhidas aqui:
    - Peso
    - Canela
    - Circunferência Abdominal
    - Circunferência da Cabeça

Layout: cada variável vira uma imagem separada (para não ficar tudo numa figura
gigante). Em cada imagem, Macho (topo) e Fêmea (base) como caixas horizontais —
"crescendo verticalmente", uma sobre a outra, como na anatomia do boxplot.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES = ROOT / 'results' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_style('whitegrid')

# (coluna, título, rótulo do eixo de valor, slug do arquivo)
VARS = [
    ('PESO',        'Peso',                     'Peso (g)',                  'peso'),
    ('CANELA',      'Canela',                   'Canela',                    'canela'),
    ('CIRCFABDOM',  'Circunferência Abdominal', 'Circunferência Abdominal',  'circ_abdominal'),
    ('CIRCFCABECA', 'Circunferência da Cabeça', 'Circunferência da Cabeça',  'circ_cabeca'),
]
ORDER = ['Macho', 'Femea']            # Macho primeiro -> fica no topo após invert
LABELS = {'Macho': 'Macho', 'Femea': 'Fêmea'}
PALETTE = {'Macho': '#4C72B0', 'Femea': '#C44E52'}


def sig_marker(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


# =============================================================================
# DADOS
# =============================================================================
df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')
for col, *_ in VARS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df[df['SEXO'].isin(ORDER)]

print("=" * 72)
print("BOXPLOTS HORIZONTAIS POR SEXO - UMA IMAGEM POR VARIÁVEL (idades agregadas)")
print("=" * 72)
print(f"{'Variável':<26} | {'Med. Macho':>10} | {'Med. Fêmea':>10} | {'p (MW-U)':>9} | sig")
print("-" * 72)

# =============================================================================
# UMA IMAGEM POR VARIÁVEL
# =============================================================================
for col, title, xlabel, slug in VARS:
    sub = df.dropna(subset=[col])
    m = sub[sub['SEXO'] == 'Macho'][col]
    f = sub[sub['SEXO'] == 'Femea'][col]
    _, p = stats.mannwhitneyu(m, f, alternative='two-sided')

    fig, ax = plt.subplots(figsize=(9, 3.2))
    # order=['Macho','Femea'] -> Macho no topo (seaborn já põe o 1º no topo).
    sns.boxplot(data=sub, x=col, y='SEXO', hue='SEXO',
                order=ORDER, hue_order=ORDER, palette=PALETTE,
                legend=False, width=0.6, showfliers=False, ax=ax)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABELS[s] for s in ORDER], fontsize=11)
    ax.set_ylabel('')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    # Valor da mediana acima de cada caixa (Macho=pos 0 no topo, Fêmea=pos 1).
    for pos, val in [(0, m.median()), (1, f.median())]:
        ax.annotate(f'{val:.1f}', xy=(val, pos), xytext=(0, 16),
                    textcoords='offset points', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#222222',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))

    fig.suptitle(f'{title} — Macho × Fêmea', fontsize=14, fontweight='bold', y=1.02)
    ax.set_title(f'Medianas: Macho {m.median():.1f} · Fêmea {f.median():.1f}',
                 fontsize=10, color='dimgray')

    fig.tight_layout()
    out = FIGURES / f'boxplot_sexo_{slug}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"{title:<26} | {m.median():>10.2f} | {f.median():>10.2f} | "
          f"{p:>9.4f} | {sig_marker(p)}    -> results/figures/{out.name}")

print("-" * 72)
print(f"{len(VARS)} imagens salvas em results/figures/ (boxplot_sexo_*.png)")
print("=" * 72)
