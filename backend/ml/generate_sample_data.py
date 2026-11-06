"""
Generates a synthetic raw-readings dataset for local testing, with a
deliberate subset of meters carrying theft-like patterns (sudden drops,
skewed night-time usage) so train_random_forest.py has real signal to learn.

Usage:
    python generate_sample_data.py --meters 150 --days 30 --out-dir sample_data
"""

import argparse
import os
import numpy as np
import pandas as pd


def generate(meters: int, days: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    n_anomalous = max(1, int(meters * 0.15))
    anomalous_ids = set(rng.choice(meters, size=n_anomalous, replace=False))

    rows = []
    labels = []
    start = pd.Timestamp("2026-07-01")

    for m in range(meters):
        meter_id = f"MTR-{400000 + m * 7:06d}"
        is_anomalous = m in anomalous_ids
        base = rng.uniform(0.15, 0.45)  # baseline hourly kWh — gives a realistic ~4-10 kWh/day household meter

        for d in range(days):
            day = start + pd.Timedelta(days=d)
            for h in range(24):
                ts = day + pd.Timedelta(hours=h)
                night = h >= 22 or h < 6
                usage = base * (0.4 if night else 1.0) * rng.uniform(0.85, 1.15)

                if is_anomalous and d > days * 0.6:
                    # simulate meter tampering: daytime usage suppressed,
                    # night usage relatively unaffected -> unusual night/day ratio
                    if not night:
                        usage *= rng.uniform(0.25, 0.5)

                rows.append((meter_id, ts.isoformat(), round(max(usage, 0), 3)))
        labels.append((meter_id, 1 if is_anomalous else 0))

    readings = pd.DataFrame(rows, columns=["meter_id", "timestamp", "consumption_kwh"])
    labels_df = pd.DataFrame(labels, columns=["meter_id", "label"])
    return readings, labels_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meters", type=int, default=150)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out-dir", default="sample_data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    readings, labels = generate(args.meters, args.days)

    readings_path = os.path.join(args.out_dir, "raw_readings.csv")
    labels_path = os.path.join(args.out_dir, "labels.csv")
    readings.to_csv(readings_path, index=False)
    labels.to_csv(labels_path, index=False)

    print(f"Wrote {len(readings)} readings across {args.meters} meters to {readings_path}")
    print(f"Wrote labels ({labels['label'].sum()} anomalous of {len(labels)}) to {labels_path}")


if __name__ == "__main__":
    main()
