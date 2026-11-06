"""
Flat-file datastore for Vidyut Audit — replaces MySQL + MongoDB.

Every function here does exactly what the SQL query or Mongo lookup it
replaces used to do, against plain CSV/JSON files under backend/data/.
This is the only module in the project that touches the filesystem for
persistence; app/main.py calls these functions instead of running queries.

Files:
    data/features.csv        one row per meter — replaces Mongo feature_snapshots
    data/detections.csv      one row per meter — replaces MySQL detection_result + meter join
    data/investigations.json list of case dicts — replaces investigation + theft_alert + inspector
    data/audit_chain.json    persisted hash chain — replaces audit_anchor (see blockchain/audit_chain.py)
"""

import json
import os
import threading
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
DETECTIONS_PATH = os.path.join(DATA_DIR, "detections.csv")
INVESTIGATIONS_PATH = os.path.join(DATA_DIR, "investigations.json")
RAW_READINGS_PATH = os.path.join(DATA_DIR, "raw_readings.csv")
RF_METRICS_PATH = os.path.join(DATA_DIR, "rf_metrics.json")
KMEANS_METRICS_PATH = os.path.join(DATA_DIR, "kmeans_metrics.json")

FEATURE_COLUMNS = [
    "avg_daily_kwh", "max_kwh", "min_kwh", "variance_kwh",
    "night_day_ratio", "weekend_weekday_ratio", "sudden_change_score",
    "trend_slope", "deviation_from_history",
]

_lock = threading.Lock()  # simple write-serialization; fine for a single-process demo API


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ------------------------------------------------------------- features ----
def get_meter_features(meter_id: str) -> dict | None:
    """Replaces: mongo_db['feature_snapshots'].find_one({'meter_id': meter_id}, ...)"""
    if not os.path.exists(FEATURES_PATH):
        return None
    df = pd.read_csv(FEATURES_PATH)
    row = df[df["meter_id"] == meter_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def read_features_all() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        return pd.DataFrame(columns=["meter_id"] + FEATURE_COLUMNS)
    return pd.read_csv(FEATURES_PATH)


def get_dataset_feature_averages() -> dict:
    """Column means across every meter — used as a comparison baseline so the
    dashboard can show 'this meter's night_day_ratio vs. the dataset average'."""
    df = read_features_all()
    if df.empty:
        return {}
    return {col: round(float(df[col].mean()), 4) for col in FEATURE_COLUMNS if col in df.columns}


def append_feature_row(row: dict):
    """Used by /consumers/register — adds a new meter's feature vector so it
    flows through the same features.csv the rest of the pipeline reads from."""
    _ensure_data_dir()
    with _lock:
        df = read_features_all()
        df = df[df["meter_id"] != row["meter_id"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(FEATURES_PATH, index=False)


# --------------------------------------------------------- raw readings ----
def has_raw_readings() -> bool:
    return os.path.exists(RAW_READINGS_PATH)


def grid_daily_totals(days: int = 30) -> list[dict]:
    """Aggregates data/raw_readings.csv across ALL meters, per day — a real
    (not fabricated) feeder-level load trend, if raw readings are present."""
    if not has_raw_readings():
        return []
    df = pd.read_csv(RAW_READINGS_PATH, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["consumption_kwh"].sum().reset_index()
    daily = daily.sort_values("date").tail(days)
    return [{"date": str(r["date"]), "total_kwh": round(float(r["consumption_kwh"]), 2)} for _, r in daily.iterrows()]


def get_meter_raw_readings(meter_id: str, limit: int = 200) -> list[dict]:
    """Raw (non-aggregated) interval readings for one meter — used only to
    populate the Analyze Meter example loader in the dashboard with real data."""
    if not has_raw_readings():
        return []
    df = pd.read_csv(RAW_READINGS_PATH)
    df = df[df["meter_id"] == meter_id].sort_values("timestamp").head(limit)
    return [{"timestamp": r["timestamp"], "consumption_kwh": round(float(r["consumption_kwh"]), 4)} for _, r in df.iterrows()]


def meter_daily_totals(meter_id: str, days: int = 30) -> list[dict]:
    """Same aggregation, filtered to one meter — backs the consumer detail trend chart."""
    if not has_raw_readings():
        return []
    df = pd.read_csv(RAW_READINGS_PATH, parse_dates=["timestamp"])
    df = df[df["meter_id"] == meter_id]
    if df.empty:
        return []
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["consumption_kwh"].sum().reset_index()
    daily = daily.sort_values("date").tail(days)
    return [{"date": str(r["date"]), "total_kwh": round(float(r["consumption_kwh"]), 2)} for _, r in daily.iterrows()]


# -------------------------------------------------------------- metrics ----
def read_json_metrics(path: str) -> dict | None:
    """Reads a metrics file written by train_random_forest.py / train_kmeans.py
    (feature importance, F1/ROC-AUC, k/silhouette). Returns None if not yet trained."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------- detections ----
def read_detections() -> pd.DataFrame:
    if not os.path.exists(DETECTIONS_PATH):
        return pd.DataFrame(columns=["meter_id", "rf_probability", "kmeans_score", "risk_score", "risk_tier"])
    return pd.read_csv(DETECTIONS_PATH)


def top_risk_detections(limit: int = 50) -> list[dict]:
    """Replaces the MySQL detection_result/meter JOIN + ORDER BY + LIMIT."""
    df = read_detections()
    if df.empty:
        return []
    df = df.sort_values("risk_score", ascending=False).head(limit)
    return df.to_dict(orient="records")


def get_detection(meter_id: str) -> dict | None:
    df = read_detections()
    row = df[df["meter_id"] == meter_id]
    return None if row.empty else row.iloc[0].to_dict()


def upsert_detection(record: dict):
    """Used by seed_detections.py after (re)running the models."""
    _ensure_data_dir()
    with _lock:
        df = read_detections()
        df = df[df["meter_id"] != record["meter_id"]]
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(DETECTIONS_PATH, index=False)


# -------------------------------------------------------- investigations ----
def _load_investigations() -> list[dict]:
    if not os.path.exists(INVESTIGATIONS_PATH):
        return []
    with open(INVESTIGATIONS_PATH) as f:
        return json.load(f)


def _save_investigations(cases: list[dict]):
    _ensure_data_dir()
    with open(INVESTIGATIONS_PATH, "w") as f:
        json.dump(cases, f, indent=2, default=str)


def list_investigations() -> list[dict]:
    return _load_investigations()


def get_investigation(investigation_id: int) -> dict | None:
    for case in _load_investigations():
        if case["investigation_id"] == investigation_id:
            return case
    return None


def create_investigation(meter_id: str, inspector: str, opened_at: str) -> dict:
    """Replaces INSERT INTO theft_alert / investigation."""
    with _lock:
        cases = _load_investigations()
        next_id = (max((c["investigation_id"] for c in cases), default=0)) + 1
        detection = get_detection(meter_id) or {}
        case = {
            "investigation_id": next_id,
            "alert_id": next_id,  # 1:1 in this simplified model — no separate alert table
            "meter_id": meter_id,
            "inspector": inspector,
            "opened_at": opened_at,
            "closed_at": None,
            "outcome": "pending",
            "notes": None,
            "risk_score": detection.get("risk_score"),
        }
        cases.append(case)
        _save_investigations(cases)
        return case


def update_investigation_outcome(investigation_id: int, outcome: str, notes: str | None) -> dict | None:
    """Replaces UPDATE investigation SET outcome = ..., notes = ... WHERE investigation_id = ..."""
    with _lock:
        cases = _load_investigations()
        for case in cases:
            if case["investigation_id"] == investigation_id:
                case["outcome"] = outcome
                case["notes"] = notes
                _save_investigations(cases)
                return case
        return None
