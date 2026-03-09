# Diabetes Risk Prediction — Project Conclusion

## Overview

This project built two binary classification models to predict diabetes risk
from the sklearn diabetes dataset (442 patients, 10 features). Two risk
thresholds were defined based on the continuous disease progression target Y:
**Scenario 1 (Y >= 150)** representing at-risk patients and **Scenario 2
(Y >= 250)** representing high-risk patients. Each scenario was treated as
an independent binary classification problem with its own model selection,
preprocessing and evaluation strategy.

---

## Exploratory Data Analysis

The EDA drove every downstream decision in the project. Key findings:

- **Feature reduction** — S1, S2, S4 and S6 were dropped based on combined
  evidence from VIF scores (S1: 576, S2: 244) and pairwise correlations
  (S1↔S2: 0.90, S3↔S4: -0.74). The retained feature set — AGE, SEX, BMI,
  BP, S3, S5 — balances predictive value against multicollinearity.

- **Strongest predictors** — BMI (r=0.59) and S5 log triglycerides (r=0.57)
  are the most predictive features for disease progression. S3 HDL cholesterol
  (r=-0.39) is also one  as higher HDL consistently associates with lower progression.

- **Residual multicollinearity** — S5 retains a VIF of 77.4 after manual feature
  reduction due to its overlap with the BMI/BP cluster. This was accepted
  given S5's strong predictive value, with tree-based models selected for
  Scenario 2 to handle this naturally.

- **Class balance** — Scenario 1 has a near-balanced 54/46 split. Scenario 2
  has a severe 85/15 imbalance, requiring SMOTE resampling before training.

---

## Scenario 1 — At Risk (Y >= 150)

**Models tested:** Logistic Regression (mean centered dataset, L2
regularization) and Random Forest (original dataset, max_depth=5).

**Results:**

- Logistic Regression achieved a cross-validated AUC of **0.851 (±0.032)**,
  correctly identifying 29 out of 41 at-risk patients (recall: 0.71).
- Random Forest achieved a cross-validated AUC of **0.831 (±0.037)**,
  correctly identifying 27 out of 41 at-risk patients (recall: 0.66).
- Logistic Regression is the stronger model — higher AUC, fewer missed
  at-risk patients and more stable cross-validation performance.

**Feature coefficients** directly confirmed the EDA findings. S5 (0.84),
BP (0.74) and BMI (0.61) are the strongest positive predictors. S3 (-0.35)
confirmed its protective role with a negative coefficient exactly as the
correlation analysis predicted. AGE (0.06) is negligible, consistent with
its weak EDA correlation.

---

## Scenario 2 — High Risk (Y >= 250)

**Model:** Random Forest with SMOTE (original dataset, max_depth=5,
class_weight='balanced').

SMOTE balanced the training set from 301/52 to 301/301 before training.
The test set was kept at its natural 85/15 distribution to reflect
real-world performance.

**Results:**

- AUC of **0.898** on the test set, cross-validated AUC of **0.940 (±0.013)**.
- Correctly identified 10 out of 13 high-risk patients (recall: 0.77),
  missing only 3.
- Precision of 0.43 for the high-risk class reflects the heavily imbalanced
  test set (13 positive cases) rather than a model weakness.
- Scenario 2 achieved the strongest overall AUC despite being the harder
  problem — SMOTE and class weighting successfully addressed the imbalance.

---

## Performance Ceiling & Limitations

Both models plateau around 70-85% accuracy depending on the scenario. This
is not a modeling failure but a reflection of the dataset's information
limits. Diabetes progression is heavily influenced by factors not captured
in the dataset based on medical sources mentioned in the EDA — genetics, lifestyle, diet, medication history and longitudinal trends. These unmeasured variables create an unexplained
variance floor. 

---

## Future Work

- **Enrich the feature set** with lifestyle variables (diet, physical
  activity), genetic markers and longitudinal measurements — this would
  be the most impactful improvement to push beyond the current performance
  ceiling.
- **Increase dataset size** — at 1500+ patients with an enriched feature
  set, XGBoost or gradient boosting would be justified. At 5000+ patients,
  neural network approaches become reasonable.
- **Threshold sensitivity analysis** — exploring intermediate thresholds
  between 150 and 250 could reveal a more clinically meaningful risk
  stratification than the two binary splits used here.
- **Exploring additional models** — XGBoost and LightGBM are natural next
  candidates as the dataset grows, offering stronger handling of
  multicollinearity and class imbalance than the current models.