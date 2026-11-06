"""
Turn the Kaggle SGCC file into the CSVs this project already trains on.

The file (data set.csv from bensalem14/sgcc-dataset) is wide: one row per
consumer, one column per day (2014-01-01 to 2016-10-30), plus CONS_NO and
FLAG. This script does not invent hourly readings. Night-versus-day ratio
is left out, because that split is not in the file. The other eight
behavioural features are computed from the daily totals.

Download (needs a free Kaggle account, one time):
    pip install kaggle
    kaggle datasets download -d bensalem14/sgcc-dataset -p data/sgcc_raw --unzip

Then, from backend/:
    python ml/prepare_sgcc.py --input "data/sgcc_raw/data set.csv" --out-dir data

That overwrites data/features.csv, data/labels.csv and a raw-readings
sample used by the dashboard charts. Train afterwards with the same
commands as the synthetic set (see the README).
"""

import argparse
import os

import numpy as np
import pandas as pd


ID_NAMES = ("CONS_NO", "CONSUMER_NO", "METER_ID")
FLAG_NAMES = ("FLAG", "LABEL")


def _find_column(columns, names):
    lookup = {str(c).strip().upper(): c for c in columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def load_wide(path):
    preview = pd.read_csv(path, nrows=2)
    id_col = _find_column(preview.columns, ID_NAMES)
    flag_col = _find_column(preview.columns, FLAG_NAMES)
    if id_col is None or flag_col is None:
        raise ValueError(
            f"Expected a consumer id ({', '.join(ID_NAMES)}) and FLAG. "
            f"Got columns: {list(preview.columns)[:8]}"
        )

    df = pd.read_csv(path, dtype={id_col: str}, low_memory=False)
    labels = pd.to_numeric(df[flag_col], errors="coerce")
    keep = labels.isin((0, 1))
    df = df.loc[keep].copy()
    labels = labels.loc[keep].astype(int)

    candidates = [c for c in df.columns if c not in (id_col, flag_col)]
    parsed = pd.to_datetime(candidates, errors="coerce", format="mixed")
    dated = [(c, ts) for c, ts in zip(candidates, parsed) if not pd.isna(ts)]
    if len(dated) < 7:
        raise ValueError(f"Only {len(dated)} date columns parsed. Is this the SGCC wide file?")
    dated.sort(key=lambda item: item[1])
    date_cols = [c for c, _ in dated]
    dates = pd.DatetimeIndex([ts for _, ts in dated])

    values = df[date_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    meter_ids = df[id_col].astype(str).to_numpy()
    return meter_ids, labels.to_numpy(), values, dates, date_cols


def _features_from_daily(values, dates):
    """Eight features from a (consumers x days) matrix. NaN means 'not recorded'."""
    n, t = values.shape
    observed = np.isfinite(values)

    with np.errstate(all="ignore"):
        avg = np.nanmean(values, axis=1)
        max_kwh = np.nanmax(values, axis=1)
        min_kwh = np.nanmin(values, axis=1)
        variance = np.nanvar(values, axis=1)

        weekend = np.asarray(dates.dayofweek >= 5)
        if weekend.any():
            weekend_avg = np.nanmean(values[:, weekend], axis=1)
        else:
            weekend_avg = np.full(n, np.nan)
        if (~weekend).any():
            weekday_avg = np.nanmean(values[:, ~weekend], axis=1)
        else:
            weekday_avg = np.full(n, np.nan)

    ratio = np.ones(n)
    ok = np.isfinite(weekend_avg) & np.isfinite(weekday_avg) & (weekday_avg > 0)
    ratio[ok] = weekend_avg[ok] / weekday_avg[ok]

    rolling = pd.DataFrame(values).T.rolling(7, min_periods=3).mean().T.to_numpy()
    denom = rolling.copy()
    denom[denom == 0] = np.nan
    with np.errstate(all="ignore"):
        pct = np.abs(values - rolling) / denom
        sudden = np.nanmax(pct, axis=1)
    sudden = np.where(np.isfinite(sudden), sudden, 0.0)

    x = np.arange(t, dtype=float)
    weight = observed.astype(float)
    filled = np.where(observed, values, 0.0)
    weight_sum = weight.sum(axis=1)
    x_mean = (weight * x).sum(axis=1) / np.maximum(weight_sum, 1.0)
    y_mean = (weight * filled).sum(axis=1) / np.maximum(weight_sum, 1.0)
    x_centered = x - x_mean[:, None]
    var_x = (weight * x_centered ** 2).sum(axis=1)
    cov = (weight * x_centered * (filled - y_mean[:, None])).sum(axis=1)
    slope = np.zeros(n)
    enough = (weight_sum >= 2) & (var_x > 0)
    slope[enough] = cov[enough] / var_x[enough]

    deviation = np.zeros(n)
    if t >= 14:
        with np.errstate(all="ignore"):
            recent = np.nanmean(values[:, -7:], axis=1)
            history = values[:, :-7]
            hist_mean = np.nanmean(history, axis=1)
            hist_std = np.nanstd(history, axis=1)
            dev = np.abs(recent - hist_mean) / (hist_std + 1e-6)
        deviation = np.where(np.isfinite(dev), dev, 0.0)

    return {
        "avg_daily_kwh": np.round(avg, 3),
        "max_kwh": np.round(max_kwh, 3),
        "min_kwh": np.round(min_kwh, 3),
        "variance_kwh": np.round(variance, 4),
        "weekend_weekday_ratio": np.round(ratio, 4),
        "sudden_change_score": np.round(sudden, 4),
        "trend_slope": np.round(slope, 4),
        "deviation_from_history": np.round(deviation, 4),
    }


def _stratified_idx(labels, n, rng):
    labels = np.asarray(labels)
    if n >= len(labels):
        return np.arange(len(labels))
    theft = np.flatnonzero(labels == 1)
    normal = np.flatnonzero(labels == 0)
    n_theft = int(round(n * (len(theft) / len(labels)))) if len(labels) else 0
    n_theft = min(len(theft), max(1, n_theft) if len(theft) else 0)
    n_normal = min(len(normal), n - n_theft)
    take_theft = rng.choice(theft, size=n_theft, replace=False) if n_theft else np.array([], dtype=int)
    take_normal = rng.choice(normal, size=n_normal, replace=False) if n_normal else np.array([], dtype=int)
    return np.sort(np.concatenate([take_theft, take_normal]))


def prepare(path, out_dir, raw_meters, max_consumers, min_days, seed):
    meter_ids, labels, values, dates, date_cols = load_wide(path)
    print(f"Read {len(meter_ids)} consumers x {values.shape[1]} days from {path}")

    frame = pd.DataFrame(values)
    # Fill only short interior gaps. A missing day is not the same thing as zero use.
    frame = frame.interpolate(axis=1, limit=3, limit_area="inside")
    values = frame.to_numpy(dtype=float)

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("No numeric consumption values in the date columns.")
    upper = float(np.quantile(finite, 0.999))
    values = np.clip(values, 0, upper)

    n_obs = np.isfinite(values).sum(axis=1)
    need = min_days if min_days > 0 else min(30, max(7, values.shape[1] // 2))
    keep = n_obs >= need
    print(f"Keeping {int(keep.sum())} consumers with at least {need} observed days (of {values.shape[1]}).")
    meter_ids, labels, values = meter_ids[keep], labels[keep], values[keep]

    rng = np.random.default_rng(seed)
    if max_consumers and len(meter_ids) > max_consumers:
        chosen = _stratified_idx(labels, max_consumers, rng)
        meter_ids, labels, values = meter_ids[chosen], labels[chosen], values[chosen]
        print(f"Subsampled to {len(meter_ids)} consumers (--max-consumers).")

    feats = _features_from_daily(values, dates)
    features = pd.DataFrame(feats)
    features.insert(0, "meter_id", meter_ids)
    label_df = pd.DataFrame({"meter_id": meter_ids, "label": labels})
    labeled = features.merge(label_df, on="meter_id")

    os.makedirs(out_dir, exist_ok=True)
    features_path = os.path.join(out_dir, "features.csv")
    labels_path = os.path.join(out_dir, "labels.csv")
    labeled_path = os.path.join(out_dir, "features_labeled.csv")
    features.to_csv(features_path, index=False)
    label_df.to_csv(labels_path, index=False)
    labeled.to_csv(labeled_path, index=False)

    raw_n = min(raw_meters, len(meter_ids)) if raw_meters else 0
    raw_path = os.path.join(out_dir, "raw_readings.csv")
    if raw_n:
        take = _stratified_idx(labels, raw_n, rng)
        sample = pd.DataFrame(values[take], columns=dates.strftime("%Y-%m-%d"))
        sample.insert(0, "meter_id", meter_ids[take])
        long = sample.melt(id_vars="meter_id", var_name="timestamp", value_name="consumption_kwh")
        long = long.dropna(subset=["consumption_kwh"])
        long.to_csv(raw_path, index=False)
        print(f"Wrote raw daily readings for {raw_n} consumers to {raw_path}")

    n_theft = int((labels == 1).sum())
    print(f"Wrote {len(features)} feature rows to {features_path}")
    print(f"Labels: {n_theft} theft, {len(labels) - n_theft} normal ({n_theft / len(labels):.2%} theft)")
    print("night_day_ratio omitted: SGCC stores one total per day, not an hourly split.")
    print(f"Date span: {date_cols[0]} -> {date_cols[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the SGCC wide CSV for Vidyut Audit.")
    parser.add_argument("--input", required=True, help="Path to data set.csv (or datasetsmall.csv)")
    parser.add_argument("--out-dir", default="data", help="Where features.csv, labels.csv and the raw sample are written")
    parser.add_argument("--raw-meters", type=int, default=200, help="How many consumers to expand into raw_readings.csv for the dashboard. 0 skips it.")
    parser.add_argument("--max-consumers", type=int, default=0, help="0 = use every consumer that has enough days")
    parser.add_argument("--min-days", type=int, default=0, help="0 = 30 days, or half the file if the file is shorter")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    prepare(args.input, args.out_dir, args.raw_meters, args.max_consumers, args.min_days, args.seed)


if __name__ == "__main__":
    main()
