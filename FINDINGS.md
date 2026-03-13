# Research Findings - Chicken Biometric Prediction

## 1. Business Understanding

**Objective**: Use morphometric (body measurement) data to:
1. **Predict weight** from body measurements
2. **Classify sex** from body measurements

**Dataset**: 2299 records, 238 unique animals measured at 12 different ages (0 to 115 days).

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

**Key insight**: CV% indicates how much individual variation exists within each age. Ages 21, 38, and 52 have the highest CV%, making them potentially easier targets for within-age prediction. Ages 0, 101, and 115 have the lowest CV% - less room for differentiation.

### 2.4 Sexual Dimorphism Analysis (Critical Finding)

**Result: Weight does NOT significantly differ between males and females at ANY age.**

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

**No morphometric feature shows strong correlation with sex** (max point-biserial r = 0.043 for CANELA, p=0.04). This is a fundamental limitation of the data - sex classification from these measurements alone may be biologically impossible for this breed.

Even within specific ages, feature separability is very weak:
- Only a handful of features reach p < 0.05 at specific ages (e.g., DORSO at age 52: p=0.008)
- No feature is consistently significant across ages

### 2.5 Feature Correlations with Weight

**Global correlations** (across all ages):
- CIRCFABDOM: 0.954 (highest)
- IDADE: 0.953
- CANELA: 0.890
- DORSO: 0.882

**Within-age correlations** (what matters for within-age prediction):
- Best: CIRCFABDOM at age 52 (r=0.70) and age 38 (r=0.67)
- Most ages: weak correlations (r < 0.4)
- Ages 0, 21: near-zero correlations (r < 0.2)

### 2.6 Outliers
Minimal outliers detected (< 1% per feature). TULIPA has the most (16, 0.7%).

---

## 3. Data Preparation

- **Train/Test split**: By animal (80/20), ensuring the same animal never appears in both sets
- **Missing values**: Dropped rows with NaN (< 1% loss)
- **Numeric conversion**: All morphometric columns converted with `pd.to_numeric(errors='coerce')`
- **Class imbalance**: Macho:Femea ratio is 1.42:1, handled with class_weight='balanced' or scale_pos_weight

---

## 4. Modeling

### 4.1 Weight Prediction (Regression)

**Models compared** (10 total):
- XGBoost, LightGBM, Random Forest, Extra Trees, Gradient Boosting
- SVR, KNN, Ridge, Lasso, ElasticNet

**Results**: _(to be filled after comparison runs)_

### 4.2 Sex Classification (Binary Classification)

**Models compared** (8 total):
- XGBoost, LightGBM, Random Forest, Extra Trees, Gradient Boosting
- SVM, KNN, Logistic Regression

**Expected outcome**: Given the EDA findings (no significant dimorphism), models will likely perform near the majority-class baseline (~58.6% for predicting all as Macho).

**Results**: _(to be filled after comparison runs)_

---

## 5. Evaluation

### Key Metrics
- **Weight**: R2, RMSE (g), MAE (g) - evaluated globally and per-age
- **Sex**: F1-score, Accuracy, Precision, Recall - compared against majority baseline

### Important Caveats
1. **Global R2 for weight is misleading** - the ~0.94 R2 is mostly driven by IDADE explaining the growth curve. Per-age R2 is much lower.
2. **Sex classification may be fundamentally limited** - the biological data simply may not carry enough signal to differentiate sex from morphometric measurements alone in this breed.

---

## 6. Conclusions

_(To be completed after model comparison results)_
