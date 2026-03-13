"""
CRISP-DM Phase 2: Data Understanding
Exploratory Data Analysis for chicken biometric prediction.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / 'results' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_style('whitegrid')

# =============================================================================
# 1. LOAD & INSPECT
# =============================================================================
print("=" * 80)
print("CRISP-DM - PHASE 2: DATA UNDERSTANDING")
print("=" * 80)

df = pd.read_csv(ROOT / 'data' / 'raw' / 'dataset.csv', sep=';', decimal='.', encoding='utf-8')

MORPHOMETRIC = ['BICO', 'CIRCFCABECA', 'PESCOCO', 'ASA', 'TULIPA',
                'DORSO', 'VENTRE', 'CIRCFABDOM', 'SOBRECOXA',
                'COXA', 'CANELA', 'UNHAMAIOR']

# Ensure numeric types
for col in ['PESO', 'IDADE'] + MORPHOMETRIC:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\nShape: {df.shape}")
print(f"Animals: {df['ANIMAL'].nunique()}")
print(f"Ages: {sorted(df['IDADE'].unique())}")
print(f"Sex distribution:\n{df['SEXO'].value_counts()}")

# =============================================================================
# 2. MISSING VALUES
# =============================================================================
print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
missing_df = missing_df[missing_df['count'] > 0]
if len(missing_df) > 0:
    print(missing_df.sort_values('count', ascending=False))
else:
    print("No missing values found.")

# Missing by age
print("\nMissing values by age:")
for age in sorted(df['IDADE'].unique()):
    subset = df[df['IDADE'] == age]
    n_missing = subset[MORPHOMETRIC + ['PESO']].isnull().any(axis=1).sum()
    if n_missing > 0:
        print(f"  Age {int(age):>3}: {n_missing}/{len(subset)} rows with NaN")

# =============================================================================
# 3. BASIC STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

desc = df[['PESO'] + MORPHOMETRIC].describe().round(2)
print(desc.to_string())

# Stats by sex
print("\n--- Weight by Sex ---")
for sex in df['SEXO'].unique():
    subset = df[df['SEXO'] == sex]['PESO'].dropna()
    print(f"  {sex}: mean={subset.mean():.1f}g, std={subset.std():.1f}g, "
          f"min={subset.min():.0f}g, max={subset.max():.0f}g")

# =============================================================================
# 4. WEIGHT DISTRIBUTION BY AGE
# =============================================================================
print("\n" + "=" * 80)
print("WEIGHT BY AGE")
print("=" * 80)

print(f"{'Age':>5} | {'N':>4} | {'Mean':>8} | {'Std':>7} | {'CV%':>5} | {'Min':>6} | {'Max':>6}")
print("-" * 55)
for age in sorted(df['IDADE'].unique()):
    subset = df[df['IDADE'] == age]['PESO'].dropna()
    cv = (subset.std() / subset.mean() * 100) if subset.mean() > 0 else 0
    print(f"{int(age):>5} | {len(subset):>4} | {subset.mean():>8.1f} | "
          f"{subset.std():>7.1f} | {cv:>5.1f} | {subset.min():>6.0f} | {subset.max():>6.0f}")

# =============================================================================
# 5. SEX DIMORPHISM BY AGE
# =============================================================================
print("\n" + "=" * 80)
print("SEXUAL DIMORPHISM - WEIGHT DIFFERENCE BY AGE")
print("=" * 80)

print(f"{'Age':>5} | {'M_mean':>8} | {'F_mean':>8} | {'Diff%':>6} | {'p-value':>8} | {'Separable?':>10}")
print("-" * 60)
for age in sorted(df['IDADE'].unique()):
    sub = df[df['IDADE'] == age]
    m = sub[sub['SEXO'] == 'Macho']['PESO'].dropna()
    f = sub[sub['SEXO'] == 'Femea']['PESO'].dropna()
    if len(m) < 3 or len(f) < 3:
        continue
    diff_pct = (m.mean() - f.mean()) / f.mean() * 100 if f.mean() > 0 else 0
    _, p = stats.mannwhitneyu(m, f, alternative='two-sided')
    separable = "YES" if p < 0.05 else "no"
    print(f"{int(age):>5} | {m.mean():>8.1f} | {f.mean():>8.1f} | "
          f"{diff_pct:>+5.1f}% | {p:>8.4f} | {separable:>10}")

# =============================================================================
# 6. FEATURE CORRELATION WITH TARGETS
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE CORRELATION WITH PESO")
print("=" * 80)

numeric_cols = ['PESO', 'IDADE'] + MORPHOMETRIC
corr_peso = df[numeric_cols].corr()['PESO'].drop('PESO').sort_values(ascending=False)
print(corr_peso.round(3).to_string())

print("\n--- Point-biserial correlation with SEXO (Macho=1) ---")
df_temp = df.copy()
df_temp['SEXO_NUM'] = (df_temp['SEXO'] == 'Macho').astype(int)
for col in ['PESO', 'IDADE'] + MORPHOMETRIC:
    valid = df_temp[[col, 'SEXO_NUM']].dropna()
    if len(valid) > 10:
        r, p = stats.pointbiserialr(valid['SEXO_NUM'], valid[col])
        sig = "*" if p < 0.05 else ""
        print(f"  {col:<12}: r={r:>+.3f} p={p:.4f} {sig}")

# =============================================================================
# 7. WITHIN-AGE CORRELATION (critical for per-age models)
# =============================================================================
print("\n" + "=" * 80)
print("WITHIN-AGE FEATURE CORRELATION WITH PESO")
print("=" * 80)

print(f"{'Age':>5} | ", end="")
top_features = ['CIRCFABDOM', 'DORSO', 'VENTRE', 'SOBRECOXA', 'ASA', 'COXA']
for f in top_features:
    print(f"{f:>10}", end=" | ")
print()
print("-" * (8 + 13 * len(top_features)))

for age in sorted(df['IDADE'].unique()):
    sub = df[df['IDADE'] == age].dropna(subset=['PESO'] + top_features)
    if len(sub) < 10:
        continue
    print(f"{int(age):>5} | ", end="")
    for feat in top_features:
        r = sub['PESO'].corr(sub[feat])
        print(f"{r:>+10.3f}", end=" | ")
    print()

# =============================================================================
# 8. WITHIN-AGE CORRELATION WITH SEX
# =============================================================================
print("\n" + "=" * 80)
print("WITHIN-AGE FEATURE SEPARABILITY FOR SEX (Mann-Whitney U p-value)")
print("=" * 80)

sex_features = ['PESO', 'CIRCFABDOM', 'DORSO', 'VENTRE', 'SOBRECOXA', 'ASA', 'COXA']
print(f"{'Age':>5} | ", end="")
for f in sex_features:
    print(f"{f:>10}", end=" | ")
print()
print("-" * (8 + 13 * len(sex_features)))

for age in sorted(df['IDADE'].unique()):
    sub = df[df['IDADE'] == age]
    m = sub[sub['SEXO'] == 'Macho']
    f = sub[sub['SEXO'] == 'Femea']
    if len(m) < 5 or len(f) < 5:
        continue
    print(f"{int(age):>5} | ", end="")
    for feat in sex_features:
        m_vals = m[feat].dropna()
        f_vals = f[feat].dropna()
        if len(m_vals) > 2 and len(f_vals) > 2:
            _, p = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{p:>7.4f}{marker:>3}", end=" | ")
        else:
            print(f"{'N/A':>10}", end=" | ")
    print()

# =============================================================================
# 9. OUTLIER DETECTION
# =============================================================================
print("\n" + "=" * 80)
print("OUTLIER DETECTION (IQR method)")
print("=" * 80)

for col in ['PESO'] + MORPHOMETRIC:
    valid = df[col].dropna()
    Q1 = valid.quantile(0.25)
    Q3 = valid.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((valid < lower) | (valid > upper)).sum()
    if n_outliers > 0:
        pct = n_outliers / len(valid) * 100
        print(f"  {col:<12}: {n_outliers:>4} outliers ({pct:.1f}%)")

# =============================================================================
# 10. SAMPLE SIZE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("SAMPLE SIZE BY AGE AND SEX")
print("=" * 80)

pivot = df.groupby(['IDADE', 'SEXO']).size().unstack(fill_value=0)
print(pivot.to_string())

# =============================================================================
# PLOTS
# =============================================================================
print("\nGenerating plots...")

# Plot 1: Weight distribution by age and sex
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Chicken Biometric Data - Exploratory Analysis', fontsize=16, fontweight='bold')

ax = axes[0, 0]
for sex, color in [('Macho', 'blue'), ('Femea', 'red')]:
    sub = df[df['SEXO'] == sex]
    means = sub.groupby('IDADE')['PESO'].mean()
    stds = sub.groupby('IDADE')['PESO'].std()
    ax.errorbar(means.index, means.values, yerr=stds.values,
                label=sex, marker='o', capsize=3, color=color, alpha=0.8)
ax.set_xlabel('Age (days)')
ax.set_ylabel('Weight (g)')
ax.set_title('Weight by Age and Sex (mean ± std)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Feature correlation heatmap
ax = axes[0, 1]
corr_matrix = df[['PESO'] + MORPHOMETRIC].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix')

# Plot 3: CV% by age (predictability indicator)
ax = axes[1, 0]
cvs = []
ages = sorted(df['IDADE'].unique())
for age in ages:
    sub = df[df['IDADE'] == age]['PESO'].dropna()
    cv = sub.std() / sub.mean() * 100
    cvs.append(cv)
colors = ['green' if cv > 15 else 'orange' if cv > 10 else 'red' for cv in cvs]
bars = ax.bar([str(int(a)) for a in ages], cvs, color=colors, alpha=0.7)
ax.set_xlabel('Age (days)')
ax.set_ylabel('CV (%)')
ax.set_title('Weight CV% by Age (higher = easier to predict variation)')
ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='CV=10%')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Box plots of features by sex (last age with most dimorphism)
ax = axes[1, 1]
last_age = max(df['IDADE'].unique())
sub = df[df['IDADE'] == last_age][['SEXO'] + MORPHOMETRIC].melt(
    id_vars='SEXO', var_name='Feature', value_name='Value')
sns.boxplot(data=sub, x='Feature', y='Value', hue='SEXO', ax=ax)
ax.set_title(f'Feature Distribution by Sex (Age={int(last_age)})')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES / 'eda_overview.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {FIGURES / 'eda_overview.png'}")

# Plot 5: Within-age correlation heatmap for sex
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('Sex Separability by Age (Feature Distributions)', fontsize=14, fontweight='bold')

key_ages = [0, 14, 28, 52, 80, 115]
key_ages = [a for a in key_ages if a in df['IDADE'].unique()]

for idx, age in enumerate(key_ages[:6]):
    row, col = idx // 3, idx % 3
    ax = axes2[row, col]
    sub = df[df['IDADE'] == age]

    # Best two features for this age
    best_feats = []
    for feat in MORPHOMETRIC + ['PESO']:
        m_vals = sub[sub['SEXO'] == 'Macho'][feat].dropna()
        f_vals = sub[sub['SEXO'] == 'Femea'][feat].dropna()
        if len(m_vals) > 3 and len(f_vals) > 3:
            _, p = stats.mannwhitneyu(m_vals, f_vals, alternative='two-sided')
            best_feats.append((feat, p))

    best_feats.sort(key=lambda x: x[1])
    f1, f2 = best_feats[0][0], best_feats[1][0]

    for sex, color, marker in [('Macho', 'blue', 'o'), ('Femea', 'red', 's')]:
        s = sub[sub['SEXO'] == sex]
        ax.scatter(s[f1], s[f2], c=color, marker=marker, alpha=0.6, label=sex, s=20)

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(f'Age {int(age)} (best: {f1}, {f2})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / 'eda_sex_separability.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {FIGURES / 'eda_sex_separability.png'}")

# =============================================================================
# SUMMARY & RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY & MODEL RECOMMENDATIONS")
print("=" * 80)

print("""
WEIGHT PREDICTION:
  - High inter-age variance (weight grows 32g -> 4000g+), global R2 will be inflated
  - Within-age CV% is key: low CV = little variation to predict
  - CIRCFABDOM, DORSO, VENTRE have strongest within-age correlations
  - Recommendation: per-age models may not have enough variance;
    unified model with IDADE as feature + all morphometric features

SEX CLASSIFICATION:
  - Young ages (0-14 days): very low dimorphism, essentially random
  - From ~28+ days: significant differences emerge (especially PESO, DORSO, VENTRE)
  - From ~52+ days: strong dimorphism, high classification accuracy expected
  - Recommendation: unified model with all features including IDADE and PESO
  - StratifiedKFold is essential for balanced CV

MODELS TO TRY:
  Regression (weight): XGBoost, LightGBM, Random Forest, Extra Trees, Ridge, SVR
  Classification (sex): XGBoost, LightGBM, Random Forest, Extra Trees, SVM, LogReg
""")

print("=" * 80)
print("EDA COMPLETE")
print("=" * 80)
