"""
Step 6 — Combine Random Forest probability + K-means anomaly score into a
single consumer-level risk score, and bucket into Low / Medium / High.

This module is imported by the FastAPI scoring endpoint; it is also
runnable standalone over a CSV that already has both model outputs.

Usage:
    python risk_score.py --scores rf_and_kmeans.csv --output risk_scored.csv
"""

import argparse
import pandas as pd

# Weighting is tuned experimentally (Step 6 of the methodology).
# RF gets slightly more weight because it is trained on confirmed labels;
# K-means weight covers behaviour that never appeared in labelled data.
RF_WEIGHT = 0.55
KMEANS_WEIGHT = 0.45

HIGH_THRESHOLD = 0.72
MEDIUM_THRESHOLD = 0.42


def compute_risk_score(rf_probability: float, kmeans_anomaly_score: float) -> float:
    score = RF_WEIGHT * rf_probability + KMEANS_WEIGHT * kmeans_anomaly_score
    return round(min(max(score, 0.0), 1.0), 4)


def risk_tier(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_score"] = df.apply(
        lambda r: compute_risk_score(r["rf_probability"], r["kmeans_anomaly_score"]), axis=1
    )
    df["risk_tier"] = df["risk_score"].apply(risk_tier)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="CSV with meter_id, rf_probability, kmeans_anomaly_score")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.scores)
    required = {"meter_id", "rf_probability", "kmeans_anomaly_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    scored = score_dataframe(df)
    scored.to_csv(args.output, index=False)

    tier_counts = scored["risk_tier"].value_counts().to_dict()
    print(f"Scored {len(scored)} meters. Tier breakdown: {tier_counts}")


if __name__ == "__main__":
    main()
