from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CSV = r"C:\Users\vamsi\Downloads\Telegram Desktop\preprocessed_automobile_dataset.csv"
ARTIFACT_DIR = Path("artifacts")
MODEL_DIRS = {
    "purchase_sales": Path("models/purchase_sales_intelligence"),
    "retention_service": Path("models/retention_service_intelligence"),
    "value_financial": Path("models/value_financial_intelligence"),
    "engagement_marketing": Path("models/engagement_marketing_intelligence"),
}

NUMERIC_FEATURES = [
    "vehicle_year",
    "vehicle_mileage",
    "vehicle_tradein_amount",
    "trade_in_year",
    "finance_term_months",
    "finance_amount",
    "finance_apr",
    "last_transaction_amount",
    "internal_ro_amount",
    "warranty_ro_amount",
    "sms_optin",
    "email_optin",
    "cash_deal_flag",
    "ownership_current",
    "dms_cell_linkage_score",
    "is_verified_email",
    "is_verified_address",
    "is_verified_owner",
    "warranty_1_amount",
    "defection_alert",
    "customer_age",
    "days_since_sale",
    "days_since_last_transaction",
    "vehicle_age",
    "sale_amount",
    "total_ro_amount",
    "payment_amount",
    "finance_payment_amount",
    "lease_payment_amount",
    "bd_payment",
    "bd_equity",
    "bd_incentive",
    "bd_mileage",
    "bd_warranty",
    "bd_remaining_payments",
]

CAT_FEATURES = [
    "gender",
    "vehicle_make",
    "vehicle_model",
    "trade_in_make",
    "trade_in_model",
    "finance_institution",
    "last_transaction_type",
    "last_transaction_sales_category",
    "deal_type",
    "latest_appointment_type",
    "condition",
    "fuel_type",
    "state",
    "city",
    "store_id",
    "individual_business_flag",
]

SUMMARY_FEATURES = [
    "id",
    "full_name",
    "vehicle_make",
    "vehicle_model",
    "vehicle_year",
    "vehicle_mileage",
    "days_since_sale",
    "days_since_last_transaction",
    "defection_alert",
    "sale_amount",
    "total_ro_amount",
    "finance_amount",
]

TRAINING_COLUMNS = list(dict.fromkeys(NUMERIC_FEATURES + CAT_FEATURES + SUMMARY_FEATURES))


def load_sample(data_path: str, max_rows: int | None) -> pd.DataFrame:
    path = Path(data_path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=TRAINING_COLUMNS)
        return df.head(max_rows).copy() if max_rows is not None else df

    chunks: List[pd.DataFrame] = []
    remaining = max_rows
    chunk_size = 25000 if max_rows is None else min(25000, max_rows)
    for chunk in pd.read_csv(data_path, usecols=lambda c: c in TRAINING_COLUMNS, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
        if remaining is not None:
            remaining -= len(chunk)
        if remaining is not None and remaining <= 0:
            break
    if not chunks:
        raise ValueError("No rows were read from the CSV.")
    df = pd.concat(chunks, ignore_index=True)
    return df.head(max_rows).copy() if max_rows is not None else df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def series(name: str, default=0) -> pd.Series:
        return out[name] if name in out.columns else pd.Series(default, index=out.index)

    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    days_sale = series("days_since_sale", np.nan).fillna(9999)
    days_txn = series("days_since_last_transaction", np.nan).fillna(9999)
    mileage = series("vehicle_mileage").fillna(0)
    age = series("vehicle_age").fillna(0)
    equity = series("bd_equity").fillna(0)
    incentive = series("bd_incentive").fillna(0)
    defection = series("defection_alert").fillna(0)

    purchase_score = (
        (days_sale.between(730, 2600)).astype(int)
        + (mileage >= 60000).astype(int)
        + (age >= 5).astype(int)
        + (equity > 0).astype(int)
        + (incentive > 0).astype(int)
        - (defection > 0).astype(int)
    )
    out["target_purchase_sales_opportunity"] = (purchase_score >= 2).astype(int)

    out["target_loyal_retention"] = (
        (defection == 0) & (days_txn <= 540) & (series("ownership_current").fillna(0) >= 0)
    ).astype(int)

    financial_signal = (
        series("sale_amount").fillna(0).clip(lower=0)
        + series("total_ro_amount").fillna(0).clip(lower=0)
        + series("finance_amount").fillna(0).clip(lower=0) * 0.01
        + series("payment_amount").fillna(0).clip(lower=0)
        + series("vehicle_tradein_amount").fillna(0).clip(lower=0) * 0.02
    )
    threshold = float(financial_signal.quantile(0.70))
    out["target_high_value_financial"] = (financial_signal >= threshold).astype(int)
    out["financial_signal"] = financial_signal

    email_optin = series("email_optin").fillna(0).astype(float)
    sms_optin = series("sms_optin").fillna(0).astype(float)
    verified_email = series("is_verified_email").fillna(0).astype(float)
    verified_address = series("is_verified_address").fillna(0).astype(float)
    ccpa_opt_out = series("ccpa_opt_out").fillna(0).astype(float)
    cell_dnc = series("cell_phone_dnc").fillna(0).astype(float)
    email_contactable = (email_optin > 0) & (verified_email == 1) & (ccpa_opt_out == 0)
    sms_contactable = (sms_optin > 0) & (cell_dnc == 0) & (ccpa_opt_out == 0)
    address_quality = verified_address == 1
    out["target_engagement_marketing"] = ((email_contactable | sms_contactable) & address_quality).astype(int)
    return out


def available_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def make_pipeline(df: pd.DataFrame, model_name: str):
    num_cols = [c for c in available_columns(df, NUMERIC_FEATURES) if df[c].notna().any()]
    cat_cols = available_columns(df, CAT_FEATURES)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    if model_name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=240, max_depth=18, n_jobs=-1, class_weight="balanced", random_state=42)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=240, learning_rate=0.055, max_leaf_nodes=31, l2_regularization=0.05, random_state=42)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)]), num_cols + cat_cols


def evaluate(model, x_test, y_test) -> Dict[str, float]:
    pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    try:
        proba = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def train(data_path: str, max_rows: int | None, model_choice: str) -> Dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading training data from {data_path}", flush=True)
    df = add_targets(load_sample(data_path, max_rows))
    print(f"Loaded rows={len(df):,} columns={len(df.columns):,}", flush=True)
    targets = {
        "purchase_sales": "target_purchase_sales_opportunity",
        "retention_service": "target_loyal_retention",
        "value_financial": "target_high_value_financial",
        "engagement_marketing": "target_engagement_marketing",
    }
    trained = {}
    metrics = {"rows_used": int(len(df)), "model_family": model_choice, "targets": {}}
    features = None

    for use_case, target in targets.items():
        print(f"Training {use_case}...", flush=True)
        pipeline, features = make_pipeline(df, "hist_gradient_boosting")
        train_df = df[features].copy()
        y = df[target].astype(int)
        print(f"{use_case} target_counts={y.value_counts().to_dict()}", flush=True)
        stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2, random_state=42, stratify=stratify)
        pipeline.fit(x_train, y_train)
        trained[use_case] = pipeline
        metrics["targets"][use_case] = evaluate(pipeline, x_test, y_test)
        metrics["targets"][use_case]["positive_rate"] = float(y.mean())
        print(f"Finished {use_case}: {metrics['targets'][use_case]}", flush=True)
        MODEL_DIRS[use_case].mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_DIRS[use_case] / "model.joblib")
        (MODEL_DIRS[use_case] / "metadata.json").write_text(
            json.dumps({"use_case": use_case, "features": features, "metrics": metrics["targets"][use_case]}, indent=2),
            encoding="utf-8",
        )

    summary_cols = list(dict.fromkeys(["id", "full_name", "vehicle_make", "vehicle_model", "vehicle_year", "vehicle_mileage", "days_since_sale", "days_since_last_transaction", "defection_alert", "sale_amount", "total_ro_amount", "finance_amount", "financial_signal"] + (features or [])))
    summary_df = df[[c for c in summary_cols if c in df.columns]].copy()
    sample_records = summary_df.head(1500).replace({np.nan: None}).to_dict(orient="records")
    population_prediction_counts = {}
    ranked_records = {}
    for use_case, model in trained.items():
        priority = []
        opportunity_count = 0
        for start in range(0, len(df), 25000):
            batch = df.iloc[start : start + 25000][features]
            probabilities = model.predict_proba(batch)[:, 1]
            opportunity_count += int((probabilities >= 0.5).sum())
            keep = min(25, len(probabilities))
            top_indexes = np.argpartition(probabilities, -keep)[-keep:]
            priority.extend((float(probabilities[index]), start + int(index)) for index in top_indexes)
        top_population_indexes = [index for _, index in heapq.nlargest(25, priority)]
        population_prediction_counts[use_case] = opportunity_count
        ranked_records[use_case] = summary_df.iloc[top_population_indexes].replace({np.nan: None}).to_dict(orient="records")
    bundle = {
        "models": trained,
        "features": features,
        "metrics": metrics,
        "sample_records": sample_records,
        "population_prediction_counts": population_prediction_counts,
        "ranked_records": ranked_records,
        "dataset_path": data_path,
    }
    joblib.dump(bundle, ARTIFACT_DIR / "intelligence_models.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--data", default=None, help="CSV or Parquet file. Overrides --csv when provided.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional development limit. Omit to train on every record.")
    parser.add_argument("--model", choices=["sklearn", "ft_transformer"], default="sklearn")
    args = parser.parse_args()
    if args.model == "ft_transformer":
        print("FT-Transformer requested. This project includes ft_transformer.py; use it after PyTorch imports cleanly.")
        print("Falling back to the verified sklearn trainer in this environment.")
    metrics = train(args.data or args.csv, args.max_rows, args.model)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
