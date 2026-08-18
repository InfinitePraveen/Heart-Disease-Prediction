# ❤️ Heart Disease Prediction — ML Notebook + CardioSense Web App

Two parts, one project: an ensemble machine learning model that predicts heart disease risk
from clinical features, and a deployed, portfolio-ready web app that puts that model behind
a live, monitor-styled interface.

```
heart-disease-project/
├── ml-project/                     Jupyter-notebook-only ML project
│   ├── notebook/
│   │   └── Heart_Disease_Prediction.ipynb
│   ├── data/
│   │   └── heart.csv               UCI Heart Disease (Cleveland), 303 patients
│   ├── requirements.txt
│   └── README.md
│
└── webapp/                         CardioSense — Flask demo app
    ├── app.py
    ├── requirements.txt
    ├── model/
    │   ├── heart_disease_model.pkl     exported from the notebook
    │   └── model_metadata.json
    ├── templates/
    │   └── index.html
    ├── static/
    │   ├── css/style.css
    │   └── js/script.js
    └── README.md
```

## 1. The ML project — `ml-project/`

A single notebook covering the full pipeline: EDA → light feature engineering → feature
selection (`SelectKBest` + `RFECV`) → baseline model comparison under 5-fold stratified
cross-validation → XGBoost hyperparameter tuning (`RandomizedSearchCV`) → a weighted
soft-voting ensemble (XGBoost + Random Forest + Logistic Regression) → held-out test
evaluation → exported model.

**Test set:** ROC-AUC 0.894 · Accuracy 0.820 · Recall 0.939 · F1 0.849

See [`ml-project/README.md`](ml-project/README.md) for the full write-up.

## 2. The web app — `webapp/`

**CardioSense**, a Flask app that loads the notebook's exported model and serves it behind a
futuristic patient-monitor UI: a live sweeping ECG trace, a sectioned patient-chart intake
form, and a risk readout that reacts — in rhythm, color, and jitter — to the model's
prediction. Built to demonstrate taking a notebook model to a deployed, interactive product
for interviews and portfolio review.

See [`webapp/README.md`](webapp/README.md) for setup and design notes.

```bash
cd webapp
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

> ⚠️ Educational/portfolio demo only, trained on 303 historical records. Not a medical
> device and not intended for real diagnosis or clinical use.

## Author

**Praveen** — aspiring data scientist, CS50x
GitHub: [github.com/InfinitePraveen](https://github.com/InfinitePraveen)
LinkedIn: [linkedin.com/in/infinitepraveen](https://www.linkedin.com/in/infinitepraveen/)
