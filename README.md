# AnomalyX — Interactive Machine Learning Anomaly Detection Lab

AnomalyX is a Streamlit portfolio project for exploring Isolation Forest anomaly detection on tabular data. It is designed to make the modeling assumptions visible: users choose numeric signals, set an expected anomaly rate, inspect the score-ranked output, and download the complete result table.

The included dataset is a small, deterministic demo dataset created for this application. It is synthetic and is not evidence about real-world behavior, fraud, or security incidents.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Choose Built-in sample for the guided demo or upload a CSV with at least 8 rows and one numeric column.

## Test

```powershell
pytest -q
```

## How it works

1. The app validates the table and exposes numeric columns for feature selection.
2. Missing and infinite numeric values are replaced with the median through a scikit-learn pipeline, while source rows are preserved.
3. Isolation Forest is trained with a fixed random seed and the selected contamination value.
4. Each row receives an anomaly score, a Normal or Anomaly label, and a boolean flag.

Anomaly scores are relative to the uploaded dataset and chosen features. A higher score means the row is more isolated according to the fitted model; it is not a probability, confidence level, or proof that a record is fraudulent. Results should be reviewed with domain context and, where appropriate, a validated labeled dataset.

## CSV guidance

The app keeps all original columns in the downloadable results. Non-numeric columns can be used for context but are not model features. Missing numeric values are median-imputed. Invalid CSV syntax, empty files, duplicate column names, no numeric features, and datasets that are too small produce an actionable validation message.
