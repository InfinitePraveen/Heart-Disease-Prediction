# ❤️ Heart Disease Prediction — Ensemble ML

Predicting heart disease risk from routine clinical measurements using feature selection,
cross-validated model comparison, and an XGBoost-driven soft-voting ensemble.

Everything lives in a single, self-contained Jupyter notebook — no separate `src/` package,
no hidden preprocessing scripts. Every transformation is visible in the notebook cell that
performs it.

## Dataset

`data/heart.csv` — the UCI Heart Disease (Cleveland) dataset, 303 patients × 13 clinical
features + target, as commonly distributed on Kaggle. No missing values.

| Column | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG result (0–2) |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression (exercise vs. rest) |
| slope | Slope of peak exercise ST segment |
| ca | Major vessels colored by fluoroscopy (0–4) |
| thal | Thalassemia result |
| target | 1 = heart disease present, 0 = absent |

## What's in the notebook

1. **EDA** — class balance, correlation matrix, per-feature distributions split by target.
2. **Light feature engineering** — heart-rate reserve, an age×cholesterol interaction, a
   hypertensive-range flag.
3. **Feature selection** — univariate ANOVA F-scores for a first read, then `RFECV`
   (recursive feature elimination with 5-fold stratified cross-validation, scored on
   ROC-AUC) to pick the final feature set.
4. **Baseline comparison** — Logistic Regression, SVM, Random Forest, and XGBoost, all
   scored with identical stratified 5-fold CV.
5. **Hyperparameter tuning** — `RandomizedSearchCV` over XGBoost's depth, learning rate,
   subsampling, and regularization terms.
6. **Ensemble** — a weighted soft-voting classifier combining the tuned XGBoost, a Random
   Forest, and a regularized Logistic Regression.
7. **Evaluation** — held-out test metrics, ROC curve, confusion matrix, classification
   report, XGBoost feature importances.
8. **Export** — the fitted ensemble is saved with `joblib` for the companion web app.

## Results (held-out test set)

| Metric | Score |
|---|---|
| ROC-AUC | 0.894 |
| Accuracy | 0.820 |
| Precision | 0.775 |
| Recall | 0.939 |
| F1 | 0.849 |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebook/Heart_Disease_Prediction.ipynb
```

## Author

**Praveen**
GitHub: [github.com/InfinitePraveen](https://github.com/InfinitePraveen)
LinkedIn: [linkedin.com/in/infinitepraveen](https://www.linkedin.com/in/infinitepraveen/)
