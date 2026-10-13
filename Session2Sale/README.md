# Session2Sale — E-commerce Conversion Intelligence

A deployable Streamlit ML application that estimates whether an online shopping session will end in a purchase. It combines a single-session prediction interface, CSV batch scoring, model evaluation, and dataset exploration.

## Why this project?
Instead of presenting only a notebook and an accuracy number, this project demonstrates a small end-to-end decision-support product:
- **Frontend:** multi-page Streamlit UI with session controls, metrics, charts, CSV upload and export.
- **ML backend:** preprocessing pipelines, categorical encoding, missing-value handling, Logistic Regression and class-weighted Random Forest.
- **Evaluation:** stratified held-out test set, PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix and precision–recall curve.
- **Operational workflow:** batch scoring and downloadable results.

## Dataset
Sakar, C. & Kastro, Y. (2018), **Online Shoppers Purchasing Intention Dataset**, UCI Machine Learning Repository.
- 12,330 sessions; 17 input features; target `Revenue`.
- Public dataset page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- DOI: https://doi.org/10.24432/C5F88Q
- The app retrieves the dataset using `ucimlrepo` on first run. Internet access is required for the first model load.
- The dataset is observational: predictions are not causal explanations and do not guarantee a purchase.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open https://share.streamlit.io/ and select the repository and `app.py`.
3. Deploy. The first startup downloads the public UCI dataset and trains the models; allow it a little time.
4. Test each page and the batch CSV workflow after deployment.

## How to use
1. **Predict a session:** enter the session signals and click Analyze session.
2. **Batch scoring:** download the CSV template, add rows with the required feature columns, upload and download predictions.
3. **Model evaluation:** compare model metrics and inspect the selected model's confusion matrix and precision–recall curve.
4. **Dataset explorer:** review class balance and sample records.

## Engineering choices
- Preprocessing is fitted inside scikit-learn pipelines, preventing test-set information from leaking into training transformations.
- The split is stratified because purchase sessions are a minority class.
- Class weighting helps models pay attention to the minority purchase class; it does not guarantee better business outcomes.
- A 0.50 probability threshold is used for the displayed classification. For a real product, tune the threshold on validation data based on business costs.
- The model is trained in memory and cached for the Streamlit session; no database or separate API server is required for this deadline-friendly MVP.

## Limitations / next improvements
- Public benchmark data may not represent a particular store or current customer behavior.
- Probability calibration and threshold selection deserve further validation.
- Add automated tests, model/version tracking, drift monitoring, privacy controls, and a real event-data ingestion pipeline before production use.

## Attribution
Dataset: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
