import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from ucimlrepo import fetch_ucirepo

st.set_page_config(page_title="Session2Sale | Conversion Intelligence", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.main {background-color:#f7f8fc;}
.hero {padding:1.25rem 1.5rem;border-radius:16px;background:linear-gradient(120deg,#101b3f,#245a77);color:white;margin-bottom:1rem;}
.hero h1 {color:white;margin-bottom:.25rem;}
.hero p {color:#e1eef5;margin:0;}
div[data-testid="stMetric"] {background:white;border:1px solid #e5e9f0;padding:14px;border-radius:12px;}
</style>
<div class="hero">
<h1>Session2Sale</h1>
<p>E-commerce conversion intelligence · Predict purchase intent, inspect session signals, and prioritize high-intent visits.</p>
</div>
""", unsafe_allow_html=True)

FEATURE_HELP = {
    "Administrative": "Administrative pages viewed during the session.",
    "Administrative_Duration": "Time spent on administrative pages (seconds).",
    "Informational": "Informational pages viewed.",
    "Informational_Duration": "Time spent on informational pages (seconds).",
    "ProductRelated": "Product-related pages viewed.",
    "ProductRelated_Duration": "Time spent on product pages (seconds).",
    "BounceRates": "Average bounce rate of pages visited.",
    "ExitRates": "Average exit rate of pages visited.",
    "PageValues": "Estimated value of pages visited before transaction.",
    "SpecialDay": "Closeness to a special shopping day (0 to 1).",
    "Month": "Month in which the session occurred.",
    "OperatingSystems": "Operating system category.",
    "Browser": "Browser category.",
    "Region": "Geographic region category.",
    "TrafficType": "Traffic source category.",
    "VisitorType": "Returning or new visitor.",
    "Weekend": "Whether the session happened on a weekend."
}

@st.cache_resource(show_spinner="Loading UCI dataset and training models (first run may take a minute)...")
def build_models():
    # UCI Online Shoppers Purchasing Intention Dataset (ID 468)
    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy().iloc[:, 0]
    # Normalize occasional source-column spelling variations for consistent app controls.
    rename = {
        "ProductRelated": "ProductRelated",
        "ProductRelated_Duration": "ProductRelated_Duration",
    }
    X = X.rename(columns=rename)
    y = y.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Random Forest": Pipeline([("prep", prep), ("model", RandomForestClassifier(
            n_estimators=220, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ))]),
        "Logistic Regression": Pipeline([("prep", prep), ("model", LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ))])
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        trained[name] = {
            "pipeline": model,
            "metrics": {
                "PR-AUC": float(average_precision_score(y_test, prob)),
                "ROC-AUC": float(roc_auc_score(y_test, prob)),
                "Precision": float(precision_score(y_test, pred, zero_division=0)),
                "Recall": float(recall_score(y_test, pred, zero_division=0)),
                "F1": float(f1_score(y_test, pred, zero_division=0)),
            },
            "y_test": y_test,
            "pred": pred,
            "prob": prob
        }
    return X, y, X_train, X_test, y_train, y_test, trained, cat_cols, num_cols

try:
    X, y, X_train, X_test, y_train, y_test, trained, cat_cols, num_cols = build_models()
except Exception as e:
    st.error("The app could not download the public UCI dataset. Check internet access or try again in a moment.")
    st.code(str(e))
    st.markdown("Dataset source: [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)")
    st.stop()

with st.sidebar:
    st.header("Workspace")
    page = st.radio("Choose a view", ["Predict a session", "Batch scoring", "Model evaluation", "Dataset explorer"])
    model_name = st.selectbox("Prediction model", list(trained.keys()), index=0)
    st.caption("Model outputs are estimates, not guarantees. The dataset is observational and does not establish causation.")
    st.divider()
    st.markdown("**Dataset**  \nUCI · 12,330 sessions  \nTarget: `Revenue`")

model = trained[model_name]["pipeline"]

if page == "Predict a session":
    st.subheader("Session risk assessment")
    st.write("Enter the session signals. The model estimates the probability that the visit ends in a purchase.")
    defaults = X.median(numeric_only=True).to_dict()
    cat_defaults = {}
    for c in cat_cols:
        modes = X[c].mode(dropna=True)
        cat_defaults[c] = modes.iloc[0] if len(modes) else ""
    left, right = st.columns(2)
    input_data = {}
    for i, col in enumerate(X.columns):
        parent = left if i % 2 == 0 else right
        with parent:
            if col in cat_cols:
                choices = sorted(X[col].dropna().astype(str).unique().tolist())
                default = str(cat_defaults.get(col, choices[0] if choices else ""))
                idx = choices.index(default) if default in choices else 0
                input_data[col] = st.selectbox(col, choices, index=idx, help=FEATURE_HELP.get(col, "Dataset feature"))
            elif pd.api.types.is_integer_dtype(X[col]) or col in ["Administrative", "Informational", "ProductRelated", "OperatingSystems", "Browser", "Region", "TrafficType"]:
                lo = int(max(0, np.floor(X[col].min())))
                hi = int(np.ceil(X[col].max()))
                val = int(np.clip(round(float(defaults.get(col, lo))), lo, hi))
                input_data[col] = st.number_input(col, min_value=lo, max_value=hi, value=val, step=1, help=FEATURE_HELP.get(col, "Dataset feature"))
            else:
                lo, hi = float(X[col].min()), float(X[col].max())
                val = float(defaults.get(col, (lo+hi)/2))
                input_data[col] = st.number_input(col, min_value=lo, max_value=max(hi, lo+0.000001), value=float(np.clip(val, lo, hi)), format="%.4f", help=FEATURE_HELP.get(col, "Dataset feature"))
    if st.button("Analyze session", type="primary", use_container_width=True):
        row = pd.DataFrame([{c: input_data[c] for c in X.columns}])
        probability = float(model.predict_proba(row)[0, 1])
        predicted = int(probability >= 0.5)
        c1, c2, c3 = st.columns(3)
        c1.metric("Purchase probability", f"{probability:.1%}")
        c2.metric("Model classification", "Likely to purchase" if predicted else "Unlikely to purchase")
        c3.metric("Model used", model_name)
        if probability >= 0.70:
            st.success("High predicted intent: consider a relevant, low-friction next step (e.g., clear delivery/return information).")
        elif probability >= 0.35:
            st.warning("Moderate predicted intent: inspect product-page engagement and checkout friction.")
        else:
            st.info("Low predicted intent: consider improving product discovery and page relevance.")
        st.caption("These are illustrative decision-support suggestions, not proven interventions. Thresholds are configurable product heuristics.")

elif page == "Batch scoring":
    st.subheader("Score multiple sessions")
    st.write("Upload a CSV with the same feature columns as the UCI dataset. Extra columns such as `Revenue` are ignored.")
    template = X.head(0).copy()
    st.download_button("Download input CSV template", template.to_csv(index=False).encode("utf-8"), "session2sale_template.csv", "text/csv")
    upload = st.file_uploader("Upload session CSV", type=["csv"])
    if upload:
        try:
            batch = pd.read_csv(upload)
            missing = [c for c in X.columns if c not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                features = batch[X.columns].copy()
                probabilities = model.predict_proba(features)[:, 1]
                result = batch.copy()
                result["purchase_probability"] = probabilities.round(4)
                result["predicted_purchase"] = np.where(probabilities >= 0.5, "Yes", "No")
                result["intent_band"] = pd.cut(probabilities, [-0.001, 0.35, 0.70, 1.0], labels=["Low", "Moderate", "High"])
                st.dataframe(result.head(100), use_container_width=True)
                st.metric("Sessions scored", len(result))
                st.download_button("Download scored results", result.to_csv(index=False).encode("utf-8"), "session2sale_scored.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not score this file: {e}")

elif page == "Model evaluation":
    st.subheader("Evaluation on held-out test set")
    st.caption("A stratified 80/20 train/test split is used. Metrics are computed on data not used to fit the models.")
    metric_rows = []
    for name, item in trained.items():
        metric_rows.append({"Model": name, **{k: round(v, 4) for k, v in item["metrics"].items()}})
    st.dataframe(pd.DataFrame(metric_rows).set_index("Model"), use_container_width=True)
    selected = trained[model_name]
    a, b = st.columns(2)
    with a:
        st.markdown("**Confusion matrix (threshold = 0.50)**")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(selected["y_test"], selected["pred"], display_labels=["No purchase", "Purchase"], ax=ax, colorbar=False)
        plt.xticks(rotation=15)
        st.pyplot(fig, clear_figure=True)
    with b:
        st.markdown("**Precision–recall trade-off**")
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(selected["y_test"], selected["prob"])
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall curve")
        ax.grid(alpha=.25)
        st.pyplot(fig, clear_figure=True)
    st.info("Because purchases are the minority class, accuracy alone can be misleading. Compare PR-AUC, precision, recall and F1, and choose a threshold based on the business cost of missed purchases versus unnecessary outreach.")

elif page == "Dataset explorer":
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Sessions", f"{len(X):,}")
    b.metric("Features", X.shape[1])
    c.metric("Purchase rate", f"{y.mean():.1%}")
    st.markdown("**Target distribution**")
    counts = y.map({0: "No purchase", 1: "Purchase"}).value_counts()
    st.bar_chart(counts)
    st.markdown("**Feature sample**")
    st.dataframe(X.head(15), use_container_width=True)
    st.markdown("**Numeric feature correlations**")
    corr = X.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=70, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    fig.colorbar(im, ax=ax, fraction=.025)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

st.divider()
st.caption("Dataset: Sakar & Kastro, UCI Machine Learning Repository, Online Shoppers Purchasing Intention Dataset (2018), DOI: 10.24432/C5F88Q. Dataset is observational; this demo is for educational and decision-support use.")
