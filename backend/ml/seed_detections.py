"""
Step 4-6, applied to a full feature file — replaces "seeding MySQL".

Loads the trained Random Forest + K-means models, scores every meter in
features.csv, blends the two outputs via risk_score.py, and writes the
result to data/detections.csv. This is what the API's
`/consumers/high-risk` endpoint reads.

Usage:
    python seed_detections.py --features ../data/features.csv \\
        --rf-model ../models/rf_model.joblib --kmeans-model ../models/kmeans_model.joblib \\
        --out ../data/detections.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from risk_score import compute_risk_score, risk_tier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--rf-model", required=True)
    parser.add_argument("--kmeans-model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    rf_bundle = joblib.load(args.rf_model)
    km_bundle = joblib.load(args.kmeans_model)

    rf_feats = df[rf_bundle["feature_columns"]]
    rf_probs = rf_bundle["model"].predict_proba(rf_feats)[:, 1]

    km_feats = df[km_bundle["feature_columns"]]
    scaled = km_bundle["scaler"].transform(km_feats)
    clusters = km_bundle["model"].predict(scaled)
    centroids = km_bundle["model"].cluster_centers_[clusters]
    dist = np.linalg.norm(scaled - centroids, axis=1)
    d_min, d_max = dist.min(), dist.max()
    km_scores = (dist - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(dist)

    out_rows = []
    for meter_id, rf_p, km_s in zip(df["meter_id"], rf_probs, km_scores):
        score = compute_risk_score(float(rf_p), float(km_s))
        out_rows.append({
            "meter_id": meter_id,
            "rf_probability": round(float(rf_p), 4),
            "kmeans_score": round(float(km_s), 4),
            "risk_score": score,
            "risk_tier": risk_tier(score),
        })

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    tiers = out_df["risk_tier"].value_counts().to_dict()
    print(f"Wrote {len(out_df)} detections to {args.out}. Tier breakdown: {tiers}")


if __name__ == "__main__":
    main()
