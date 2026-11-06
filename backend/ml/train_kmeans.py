"""
Step 5 — Unsupervised anomaly detection with K-means.

Clusters consumers by behavioural features (no labels required), then scores
each consumer by how far it sits from its own cluster centroid. This catches
theft-like patterns that never showed up in the labelled training data.

Usage:
    python train_kmeans.py --features features.csv --model-out kmeans_model.joblib
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURE_COLUMNS = [
    "avg_daily_kwh", "max_kwh", "min_kwh", "variance_kwh",
    "night_day_ratio", "weekend_weekday_ratio", "sudden_change_score",
    "trend_slope", "deviation_from_history",
]


def columns_for_training(df):
    cols = [col for col in FEATURE_COLUMNS if col in df.columns and df[col].notna().any()]
    skipped = [col for col in FEATURE_COLUMNS if col not in cols]
    if skipped:
        print("Not using features this file does not have:", ", ".join(skipped))
    if len(cols) < 2:
        raise ValueError(f"Need at least 2 feature columns, found {cols}")
    return cols


def _sample_rows(X_scaled: np.ndarray, cap: int, random_state: int) -> np.ndarray:
    """Silhouette is pairwise, so the full SGCC table (about 40k rows) does not fit."""
    if len(X_scaled) <= cap:
        return X_scaled
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_scaled), size=cap, replace=False)
    return X_scaled[idx]


def choose_k(X_scaled: np.ndarray, k_range=range(3, 9), random_state=42) -> int:
    """Pick k by best silhouette score (Step 5: no labels available)."""
    sample = _sample_rows(X_scaled, cap=4000, random_state=random_state)
    best_k, best_score = k_range[0], -1
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(sample)
        score = silhouette_score(sample, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def anomaly_scores(X_scaled: np.ndarray, model: KMeans) -> np.ndarray:
    """Distance from each point to its assigned centroid, min-max normalized to [0, 1]."""
    labels = model.predict(X_scaled)
    centroids = model.cluster_centers_[labels]
    dist = np.linalg.norm(X_scaled - centroids, axis=1)
    d_min, d_max = dist.min(), dist.max()
    if d_max - d_min < 1e-9:
        return np.zeros_like(dist)
    return (dist - d_min) / (d_max - d_min)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--scores-out", default="kmeans_scores.csv")
    parser.add_argument("--metrics-out", default="kmeans_metrics.json")
    parser.add_argument("--k", type=int, default=0, help="0 = auto-select via silhouette score")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    feature_columns = columns_for_training(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_columns])

    k = args.k or choose_k(X_scaled, random_state=args.random_state)
    model = KMeans(n_clusters=k, n_init=10, random_state=args.random_state).fit(X_scaled)

    df["cluster"] = model.predict(X_scaled)
    df["kmeans_anomaly_score"] = anomaly_scores(X_scaled, model)

    # Raw (pre-normalization) centroid distances, so inference-time scoring
    # can normalize new points against the SAME scale this training set used,
    # instead of a guessed constant.
    labels = model.labels_
    raw_dist = np.linalg.norm(X_scaled - model.cluster_centers_[labels], axis=1)
    max_training_distance = float(raw_dist.max())

    joblib.dump(
        {
            "model": model, "scaler": scaler, "feature_columns": feature_columns, "k": k,
            "max_training_distance": max_training_distance,
        },
        args.model_out,
    )
    df[["meter_id", "cluster", "kmeans_anomaly_score"]].to_csv(args.scores_out, index=False)

    with open(args.metrics_out, "w") as f:
        sil_sample = _sample_rows(X_scaled, cap=4000, random_state=args.random_state)
        sil_labels = model.predict(sil_sample)
        json.dump(
            {
                "k": k,
                "silhouette_score": float(silhouette_score(sil_sample, sil_labels)),
                "silhouette_sample_size": int(len(sil_sample)),
                "n_consumers": int(len(df)),
            },
            f,
            indent=2,
        )

    print(f"k={k}. Scores written to {args.scores_out}, model to {args.model_out}")


if __name__ == "__main__":
    main()
