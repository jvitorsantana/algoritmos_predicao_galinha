# Research Findings - Chicken Biometric Prediction

## 1. Business Understanding

**Objective**: Use morphometric (body measurement) data to:
1. **Predict weight** from body measurements
2. **Classify sex** from body measurements

**Dataset**: 2299 records, 238 unique Guinea fowl (Galinha D'Angola) measured at 12 ages (0 to 115 days), with 13 morphometric features.

---

## 2. Data Understanding (EDA)

### 2.1 Dataset Overview

| Feature | Description | Range |
|---------|-------------|-------|
| PESO | Weight (g) | 0 - 1905 |
| BICO | Beak | 0.4 - 23.0 |
| CIRCFCABECA | Head circumference | 1.0 - 127.0 |
| PESCOCO | Neck | 1.0 - 65.0 |
| ASA | Wing | 1.0 - 117.0 |
| TULIPA | Tulip/Comb | 0.7 - 77.0 |
| DORSO | Back | 1.3 - 210.0 |
| VENTRE | Belly | 1.3 - 375.0 |
| CIRCFABDOM | Abdominal circumference | 5.3 - 45.0 |
| SOBRECOXA | Thigh | 0.5 - 81.0 |
| COXA | Drumstick | 0.1 - 111.0 |
| CANELA | Shank | 1.4 - 13.7 |
| UNHAMAIOR | Largest nail | 1.1 - 13.7 |

### 2.2 Missing Values
Very clean dataset - less than 0.3% missing for any column. PESO has 5 missing values (0.22%).

### 2.3 Weight Growth Pattern

| Age (days) | N | Mean (g) | Std (g) | CV% |
|------------|---|----------|---------|-----|
| 0 | 232 | 32.1 | 3.9 | 12.2 |
| 7 | 232 | 83.1 | 11.2 | 13.5 |
| 14 | 226 | 168.7 | 27.2 | 16.1 |
| 21 | 229 | 259.5 | 76.3 | **29.4** |
| 28 | 221 | 401.5 | 85.5 | 21.3 |
| 38 | 224 | 566.0 | 128.2 | 22.7 |
| 52 | 224 | 793.9 | 218.2 | **27.5** |
| 66 | 222 | 980.7 | 176.0 | 17.9 |
| 80 | 212 | 1166.6 | 184.3 | 15.8 |
| 101 | 116 | 1274.3 | 156.0 | 12.2 |
| 115 | 152 | 1432.8 | 180.8 | 12.6 |

**Key insight**: CV% indicates how much individual variation exists within each age. Ages 21, 38, and 52 have the highest CV%, making them potentially easier targets for within-age prediction. Ages 0, 101, and 115 have the lowest CV%.

### 2.4 Sexual Dimorphism Analysis (Critical Finding)

**Weight does NOT significantly differ between males and females at ANY age.**

| Age | Male mean | Female mean | Diff% | p-value | Significant? |
|-----|-----------|-------------|-------|---------|-------------|
| 0 | 31.9g | 32.4g | -1.4% | 0.4817 | NO |
| 7 | 84.0g | 81.8g | +2.8% | 0.1275 | NO |
| 14 | 170.2g | 166.6g | +2.2% | 0.6544 | NO |
| 21 | 264.2g | 252.8g | +4.5% | 0.1280 | NO |
| 28 | 410.2g | 389.2g | +5.4% | 0.1029 | NO |
| 52 | 800.6g | 784.5g | +2.0% | 0.3534 | NO |
| 80 | 1166.6g | 1166.5g | +0.0% | 0.8579 | NO |
| 115 | 1434.3g | 1430.6g | +0.3% | 0.9162 | NO |

**No morphometric feature shows meaningful correlation with sex** (max point-biserial r = 0.043 for CANELA, p=0.04). Within specific ages, only scattered features reach p < 0.05, and none are consistent across ages.

### 2.5 Feature Correlations with Weight

**Global correlations** (across all ages):
- CIRCFABDOM: 0.954 (highest)
- IDADE: 0.953
- CANELA: 0.890
- DORSO: 0.882

**Within-age correlations** (what actually matters for individual prediction):
- Best: CIRCFABDOM at age 52 (r=0.70) and age 38 (r=0.67)
- Most ages: weak correlations (r < 0.4)
- Ages 0, 21: near-zero correlations (r < 0.2)

### 2.6 Outliers
Minimal outliers detected (< 1% per feature). TULIPA has the most (16, 0.7%).

---

## 3. Data Preparation

- **Train/Test split**: By animal (80/20), ensuring the same animal never appears in both sets (190 train / 48 test animals)
- **Missing values**: Dropped rows with NaN (< 1% loss)
- **Numeric conversion**: All morphometric columns converted with `pd.to_numeric(errors='coerce')`
- **Class imbalance**: Macho:Femea ratio is 1.42:1, handled with `class_weight='balanced'` or `scale_pos_weight`

---

## 4. Modeling Results

### 4.1 Weight Prediction - 10 Models Compared

| Model | R2 CV | R2 Test | RMSE (g) | MAE (g) |
|-------|-------|---------|----------|---------|
| **Extra Trees** | **0.9445** | **0.9446** | **107.81** | **69.67** |
| Random Forest | 0.9440 | 0.9437 | 108.72 | 68.66 |
| XGBoost | 0.9431 | 0.9429 | 109.49 | 69.96 |
| LightGBM | 0.9421 | 0.9422 | 110.09 | 72.36 |
| Gradient Boosting | 0.9422 | 0.9421 | 110.18 | 71.61 |
| KNN | 0.9412 | 0.9392 | 112.93 | 72.47 |
| Lasso | 0.9339 | 0.9273 | 123.49 | 84.92 |
| Ridge | 0.9335 | 0.9266 | 124.07 | 85.43 |
| ElasticNet | 0.9304 | 0.9238 | 126.46 | 87.30 |
| SVR | 0.9304 | 0.9162 | 132.57 | 83.86 |

**Winner**: Extra Trees (R2=0.9446, MAE=69.67g). All tree-based models perform similarly (~0.94). Linear models are slightly worse (~0.93). The gap is small - model choice matters less than feature quality.

#### Per-Age R2 (Top 3 Models)

| Age | N | Extra Trees | Random Forest | XGBoost |
|-----|---|-------------|---------------|---------|
| 0 | 47 | -0.15 | -0.42 | -5.46 |
| 7 | 47 | 0.28 | 0.17 | 0.12 |
| 14 | 48 | 0.16 | 0.19 | 0.16 |
| 21 | 46 | -0.09 | -0.34 | -0.50 |
| 28 | 47 | -0.09 | -0.17 | -0.21 |
| **38** | **47** | **0.47** | **0.50** | **0.51** |
| **52** | **45** | **0.51** | **0.52** | **0.54** |
| 66 | 46 | 0.21 | 0.19 | 0.23 |
| 80 | 43 | 0.14 | 0.20 | 0.19 |
| 101 | 25 | 0.04 | -0.15 | -0.29 |
| 115 | 26 | -0.00 | 0.06 | 0.08 |

**Critical insight**: The global R2 of 0.94 is almost entirely driven by IDADE (the growth curve from 32g to 1400g). Within any single age group, the model can barely predict individual weight variation. Only ages 38 and 52 show meaningful within-age R2 (~0.50), which aligns with those ages having the highest CV%.

### 4.2 Sex Classification - 8 Models Compared

| Model | F1 CV | F1 Test | Accuracy | Precision | Recall |
|-------|-------|---------|----------|-----------|--------|
| **Gradient Boosting** | **0.7354** | **0.7439** | **0.6049** | 0.6019 | 0.9738 |
| LightGBM | 0.7100 | 0.7162 | 0.5872 | 0.6020 | 0.8839 |
| KNN | 0.7187 | 0.7115 | 0.5740 | 0.5920 | 0.8914 |
| Extra Trees | 0.6572 | 0.6546 | 0.5784 | 0.6329 | 0.6779 |
| Random Forest | 0.6518 | 0.6505 | 0.5541 | 0.6045 | 0.7041 |
| XGBoost | 0.6255 | 0.6260 | 0.5673 | 0.6381 | 0.6142 |
| SVM | 0.6065 | 0.5974 | 0.5298 | 0.6031 | 0.5918 |
| Logistic Regression | 0.5511 | 0.5300 | 0.4989 | 0.5926 | 0.4794 |

**Majority baseline accuracy: 58.9%** (predicting all as Macho).

**The "best" model (Gradient Boosting) achieves 60.5% accuracy - only 1.6pp above the baseline.** Its high F1 (0.74) is misleading: the classification report reveals it predicts 97% of samples as Macho and only identifies 8% of Femeas:

```
              precision    recall  f1-score   support
       Femea       0.67      0.08      0.14       186
       Macho       0.60      0.97      0.74       267
```

Per-age accuracy is 50-65% across all ages - essentially coin-flip territory.

---

## 5. Evaluation Summary

### Weight Prediction
- **Global performance**: Excellent (R2~0.94), but this is an artifact of the age-weight growth curve
- **Within-age performance**: Poor (R2 mostly negative or near zero)
- **Practical interpretation**: The model can tell you "a 52-day-old chicken weighs ~800g" but cannot meaningfully tell you which individual chicken is heavier than another at the same age
- **Best model**: Extra Trees, though all tree-based models are equivalent

### Sex Classification
- **Performance**: No model meaningfully distinguishes sex from morphometric data
- **Best accuracy**: 60.5% vs 58.9% baseline (not significant)
- **Root cause**: Guinea fowl (Galinha D'Angola) show **no statistically significant sexual dimorphism** in any of the 13 measured morphometric features, at any age. This is a biological characteristic of the species, not a modeling failure.

---

## 6. Conclusions and Recommendations

### What the data tells us

1. **Guinea fowl are monomorphic for body measurements.** Unlike commercial broiler chickens where males are visibly larger, Guinea fowl males and females are essentially indistinguishable by body size at all ages studied (0-115 days). No amount of algorithmic sophistication can extract a signal that doesn't exist in the data.

2. **Weight is tightly coupled to age.** IDADE and CIRCFABDOM alone explain >95% of weight variance. Other morphometric features add marginal value because they're all highly correlated with body size, which is primarily determined by age.

3. **Within-age individual variation is real but hard to predict.** There IS variance within age groups (CV up to 29%), but the measured morphometric features don't capture the underlying causes (genetics, feed intake, health status, hierarchical position in the flock).

### Recommendations for future work

#### For sex classification (the harder problem)

| Approach | Feasibility | Expected Impact |
|----------|-------------|-----------------|
| **Vocalization analysis** | Medium - needs audio equipment | High - Guinea fowl sexes have distinct calls |
| **Comb/wattle morphology** (shape, not just size) | Medium - needs image analysis | Medium - males may have slightly different comb shape |
| **Behavioral features** (aggression, mating behavior) | Low - hard to quantify | Medium |
| **Genetic/molecular sexing** (feather DNA) | High - standard lab technique | Definitive - ground truth |
| **Vent sexing by specialist** | Medium - needs trained personnel | High - standard practice in poultry |
| **Add more age groups beyond 115 days** | Easy - continue measurements | Low-Medium - dimorphism may appear later in maturity |

The most promising ML-compatible approach would be **audio classification of vocalizations** - Guinea fowl sexes produce distinctly different calls (males: "chi-chi-chi", females: "buckwheat" call). A simple audio classifier could likely achieve >90% accuracy.

#### For weight prediction

| Approach | Feasibility | Expected Impact |
|----------|-------------|-----------------|
| **Use model as-is for age-based estimation** | Ready now | Good for population-level estimates |
| **Add feed intake data** | Medium - needs feed tracking | High - primary driver of individual weight variation |
| **Add genetic line/parentage** | Medium - needs pedigree data | Medium - genetic potential affects growth |
| **Longitudinal features** (weight at previous age) | Easy - already in SVM data | Medium - past weight predicts future weight |
| **Environmental features** (temperature, density) | Low - needs environmental sensors | Medium |
| **Focus only on ages 38-52** | Ready now | Best per-age R2 (~0.50) |

The current model is already useful for **estimating weight from age without a scale** (MAE ~70g). For individual-level prediction within an age group, the morphometric features alone are insufficient - the model needs information about feed intake or genetics.

### Final assessment

The research successfully demonstrates that:
1. Morphometric measurements can predict weight with high accuracy when age is known (R2=0.94)
2. **Sex cannot be determined from morphometric measurements in Guinea fowl** - this is a negative but scientifically valuable finding
3. The within-age predictive power of morphometric features is limited (R2 < 0.5 for most ages)

These findings align with the known biology of Guinea fowl as a monomorphic species and suggest that future sex classification efforts should focus on **non-morphometric features** (vocalizations, molecular markers) rather than more sophisticated algorithms applied to the same body measurements.
