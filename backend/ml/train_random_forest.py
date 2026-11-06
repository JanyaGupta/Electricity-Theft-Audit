"""
Step 4 — Supervised detection with Random Forest.

Trains on behavioural features (output of feature_engineering.py) joined
with a `label` column (1 = previously confirmed abnormal, 0 = normal),
as available in the SGCC benchmark. Saves the fitted model + metrics.

Usage:
    python train_random_forest.py --features features_labeled.csv --model-out rf_model.joblib
"""

import argparse
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE  # handles class imbalance (Step 2)

FEATURE_COLUMNS = [
    "avg_daily_kwh", "max_kwh", "min_kwh", "variance_kwh",
    "night_day_ratio", "weekend_weekday_ratio", "sudden_change_score",
    "trend_slope", "deviation_from_history",
]


def columns_for_training(df):
    """Keep a feature only when this file actually has values for it.

    SGCC is one total per day, so night_day_ratio is absent. The synthetic
    generator still has it, and that column stays in the model when present.
    """
    cols = [col for col in FEATURE_COLUMNS if col in df.columns and df[col].notna().any()]
    skipped = [col for col in FEATURE_COLUMNS if col not in cols]
    if skipped:
        print("Not using features this file does not have:", ", ".join(skipped))
    if len(cols) < 2:
        raise ValueError(f"Need at least 2 feature columns, found {cols}")
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV with FEATURE_COLUMNS + label")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", default="rf_metrics.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    if "label" not in df.columns:
        raise ValueError("features file has no label column")
    feature_columns = columns_for_training(df)

    X = df[feature_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Class imbalance is expected: confirmed theft cases are a small minority.
    sm = SMOTE(random_state=args.random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    metrics = {
        "f1_score": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "report": classification_report(y_test, preds, output_dict=True),
        "feature_importance": dict(zip(feature_columns, model.feature_importances_.tolist())),
        "feature_columns": feature_columns,
    }

    joblib.dump({"model": model, "feature_columns": feature_columns}, args.model_out)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"F1: {metrics['f1_score']:.3f}  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Model saved to {args.model_out}, metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
