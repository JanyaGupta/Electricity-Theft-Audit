"""
Feature engineering for the electricity-theft risk pipeline.

Input : raw interval readings per meter (a CSV with meter_id, timestamp,
        consumption_kwh — either your real dataset or generate_sample_data.py's
        output).
Output: one behavioural feature row per meter per window, written to
        data/features.csv — this is the file the API and seed_detections.py
        read model inputs from.

Usage:
    python feature_engineering.py --input raw_readings.csv --output features.csv
"""

import argparse
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = ["meter_id", "timestamp", "consumption_kwh"]


def load_readings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    return df


def clean_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: handle missing values, drop impossible readings, sort."""
    df = df.dropna(subset=["meter_id", "timestamp"]).copy()

    # Negative or absurdly large readings are almost certainly sensor faults;
    # clip instead of dropping so we don't lose the whole day for one meter.
    upper_bound = df["consumption_kwh"].quantile(0.999)
    df["consumption_kwh"] = df["consumption_kwh"].clip(lower=0, upper=upper_bound)

    # Fill short gaps per-meter using forward fill; longer gaps are left as
    # NaN and excluded from aggregates rather than fabricating usage.
    df = df.sort_values(["meter_id", "timestamp"])
    df["consumption_kwh"] = (
        df.groupby("meter_id")["consumption_kwh"]
        .apply(lambda s: s.ffill(limit=2))
        .reset_index(drop=True)
    )
    return df.dropna(subset=["consumption_kwh"])


def _is_night(hour: int) -> bool:
    return hour >= 22 or hour < 6


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: transform interval readings into per-meter behavioural features."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek  # 0=Mon
    df["is_night"] = df["hour"].apply(_is_night)
    df["is_weekend"] = df["dow"] >= 5
    df["date"] = df["timestamp"].dt.date

    rows = []
    for meter_id, g in df.groupby("meter_id"):
        daily = g.groupby("date")["consumption_kwh"].sum()

        avg_daily = daily.mean()
        max_kwh = daily.max()
        min_kwh = daily.min()
        variance = daily.var(ddof=0)

        # SGCC (and any other daily-total file) has one row per day, so the
        # clock hour is not a real night/day split. Leave the feature out
        # rather than scoring every midnight stamp as "night".
        one_reading_per_day = int(g.groupby("date").size().max()) == 1
        if one_reading_per_day:
            night_day_ratio = None
        else:
            night_sum = g.loc[g["is_night"], "consumption_kwh"].sum()
            day_sum = g.loc[~g["is_night"], "consumption_kwh"].sum()
            night_day_ratio = night_sum / day_sum if day_sum > 0 else 0.0

        weekend_avg = g.loc[g["is_weekend"]].groupby("date")["consumption_kwh"].sum().mean()
        weekday_avg = g.loc[~g["is_weekend"]].groupby("date")["consumption_kwh"].sum().mean()
        # Guard both sides: data that only spans weekdays (or only weekends —
        # common with a short manually-entered sample) would otherwise leave
        # one of these as NaN and silently poison every downstream model call.
        if weekday_avg and not np.isnan(weekday_avg) and not np.isnan(weekend_avg):
            weekend_weekday_ratio = weekend_avg / weekday_avg
        else:
            weekend_weekday_ratio = 1.0

        # Sudden change: largest single-day jump relative to the trailing 7-day mean
        rolling_mean = daily.rolling(7, min_periods=3).mean()
        pct_change = ((daily - rolling_mean).abs() / rolling_mean.replace(0, np.nan)).dropna()
        sudden_change_score = float(pct_change.max()) if len(pct_change) else 0.0

        # Long-term trend: slope of a simple linear fit over the daily series
        if len(daily) >= 2:
            x = np.arange(len(daily))
            trend_slope = float(np.polyfit(x, daily.values, 1)[0])
        else:
            trend_slope = 0.0

        # Deviation from this meter's own history (z-score of the last week vs. the rest)
        if len(daily) >= 14:
            recent = daily.iloc[-7:].mean()
            history = daily.iloc[:-7]
            deviation = abs(recent - history.mean()) / (history.std(ddof=0) + 1e-6)
        else:
            deviation = 0.0

        rows.append(
            {
                "meter_id": meter_id,
                "avg_daily_kwh": round(avg_daily, 3),
                "max_kwh": round(max_kwh, 3),
                "min_kwh": round(min_kwh, 3),
                "variance_kwh": round(float(variance), 4),
                "weekend_weekday_ratio": round(float(weekend_weekday_ratio), 4),
                "sudden_change_score": round(sudden_change_score, 4),
                "trend_slope": round(trend_slope, 4),
                "deviation_from_history": round(float(deviation), 4),
            }
        )
        if night_day_ratio is not None:
            rows[-1]["night_day_ratio"] = round(float(night_day_ratio), 4)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build behavioural features from raw meter readings.")
    parser.add_argument("--input", required=True, help="CSV with meter_id, timestamp, consumption_kwh")
    parser.add_argument("--output", required=True, help="Output CSV of per-meter features")
    args = parser.parse_args()

    raw = load_readings(args.input)
    clean = clean_readings(raw)
    features = build_features(clean)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
