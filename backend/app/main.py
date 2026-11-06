"""
FastAPI backend for Vidyut Audit — simplified, no external databases.

Wires together:
  - CSV/JSON flat files (backend/data/)  -> via app/datastore.py
  - Trained RF + K-means models          -> via joblib, blended in ml/risk_score.py
  - SHA-256 tamper-evident audit trail   -> blockchain/audit_chain.py (see that
    file's docstring for why "blockchain" isn't the accurate term)

Run:
    uvicorn app.main:app --reload

Config is read from environment variables (see .env.example) — just file
paths now, no connection strings.
"""

import datetime as dt
import os
import secrets
from typing import Optional

import joblib
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import datastore
from ml.risk_score import compute_risk_score, risk_tier, RF_WEIGHT, KMEANS_WEIGHT, HIGH_THRESHOLD, MEDIUM_THRESHOLD
from ml.feature_engineering import clean_readings, build_features
from blockchain.audit_chain import AuditEvent, LocalChainClient
from blockchain.sepolia_anchor import SepoliaNotConfigured, anchor_hash, is_anchored

# ---------------------------------------------------------------- config ---
RF_MODEL_PATH = os.getenv("RF_MODEL_PATH", "models/rf_model.joblib")
KMEANS_MODEL_PATH = os.getenv("KMEANS_MODEL_PATH", "models/kmeans_model.joblib")

chain_client = LocalChainClient()  # loads/persists backend/data/audit_chain.json
DASHBOARD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dashboard.html"))

app = FastAPI(title="Vidyut Audit API", version="2.0.0 (no-database)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to the dashboard origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def allow_private_network_access(request: Request, call_next):
    """
    Lets a page served over https (e.g. the published dashboard preview,
    or any site other than a bare http:// page) call this API even though
    it runs on a private/loopback address like 127.0.0.1. Chrome's Private
    Network Access policy requires this header on the response — including
    the preflight CORSMiddleware above generates — or it silently blocks
    the request client-side, which looks identical to the backend being
    unreachable even though curl/direct requests work fine.
    """
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


_model_cache: dict = {}


def _load_cached(path: str):
    """joblib.load of the Random Forest is slow (tens of MB). Keep the loaded
    model in memory and only reload if the file on disk changes."""
    if not os.path.exists(path):
        return None
    stamp = os.stat(path).st_mtime_ns
    hit = _model_cache.get(path)
    if hit and hit[0] == stamp:
        return hit[1]
    bundle = joblib.load(path)
    _model_cache[path] = (stamp, bundle)
    return bundle


def load_models():
    """Lazy-load so the API can start even before models are trained/placed."""
    return _load_cached(RF_MODEL_PATH), _load_cached(KMEANS_MODEL_PATH)


def run_models_on_features(features: dict) -> dict:
    """Steps 4-6 of the methodology, given one meter's feature row. Shared by
    /meters/{id}/score, /meters/{id}/explain, and /consumers/register so
    there's exactly one place this logic lives."""
    rf_bundle, km_bundle = load_models()
    if rf_bundle is None or km_bundle is None:
        raise HTTPException(status_code=503, detail="Models not trained/loaded on this server yet")

    feats = pd.DataFrame([features])[rf_bundle["feature_columns"]]
    rf_prob = float(rf_bundle["model"].predict_proba(feats)[:, 1][0])

    km_feats = pd.DataFrame([features])[km_bundle["feature_columns"]]
    scaled = km_bundle["scaler"].transform(km_feats)
    cluster = int(km_bundle["model"].predict(scaled)[0])
    centroid = km_bundle["model"].cluster_centers_[cluster]
    dist = float(np.linalg.norm(scaled[0] - centroid))
    # Normalized against this model's own training-set distances (saved by
    # train_kmeans.py) using dist / (dist + max_dist) rather than a hard
    # clip — a point further than any training example still gets a
    # meaningfully higher score instead of every out-of-range point
    # collapsing to the same 1.0.
    max_dist = km_bundle.get("max_training_distance", 1.0) or 1.0
    km_score = dist / (dist + max_dist)

    final = compute_risk_score(rf_prob, km_score)
    return {
        "rf_probability": round(rf_prob, 4),
        "kmeans_anomaly_score": round(km_score, 4),
        "kmeans_cluster": cluster,
        "risk_score": final,
        "risk_tier": risk_tier(final),
    }


# ------------------------------------------------------------- schemas -----
class RiskScoreResponse(BaseModel):
    meter_id: str
    rf_probability: float
    kmeans_anomaly_score: float
    risk_score: float
    risk_tier: str


class SendToInvestigationIn(BaseModel):
    meter_id: str
    inspector: str


class InvestigationOutcomeIn(BaseModel):
    investigation_id: int
    outcome: str  # pending | confirmed | not_confirmed | requires_further_review
    notes: Optional[str] = None


class AuditVerifyResponse(BaseModel):
    investigation_id: int
    verified: bool
    payload_hash: str
    chain_tx_ref: str
    sepolia_tx: Optional[str] = None
    sepolia_anchored: Optional[bool] = None


# --------------------------------------------------------------- routes ----
@app.get("/")
def dashboard_page():
    """The console lives in this response so the browser is not stuck on a cached file:// copy."""
    return FileResponse(
        DASHBOARD_PATH,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
def health():
    return {"status": "ok", "time": dt.datetime.utcnow().isoformat(), "storage": "csv/json (no database)"}


@app.get("/consumers/high-risk", response_model=list[RiskScoreResponse])
def high_risk_consumers(limit: int = 50):
    """Reads data/detections.csv, highest risk_score first.
    Replaces the old MySQL detection_result/meter JOIN."""
    rows = datastore.top_risk_detections(limit)
    return [
        RiskScoreResponse(
            meter_id=r["meter_id"],
            rf_probability=float(r["rf_probability"]),
            kmeans_anomaly_score=float(r["kmeans_score"]),
            risk_score=float(r["risk_score"]),
            risk_tier=r["risk_tier"],
        )
        for r in rows
    ]


@app.get("/meters/{meter_id}/score", response_model=RiskScoreResponse)
def score_meter(meter_id: str):
    """
    On-demand scoring: pulls the feature row from data/features.csv,
    runs both models, and blends the result (Steps 4-6 of the methodology).
    """
    features = datastore.get_meter_features(meter_id)
    if not features:
        raise HTTPException(status_code=404, detail=f"No feature row found for {meter_id}")

    scores = run_models_on_features(features)
    result = RiskScoreResponse(
        meter_id=meter_id,
        rf_probability=scores["rf_probability"],
        kmeans_anomaly_score=scores["kmeans_anomaly_score"],
        risk_score=scores["risk_score"],
        risk_tier=scores["risk_tier"],
    )
    # Keep detections.csv in sync so this meter also shows up in /consumers/high-risk
    datastore.upsert_detection({
        "meter_id": meter_id, "rf_probability": result.rf_probability,
        "kmeans_score": result.kmeans_anomaly_score, "risk_score": result.risk_score,
        "risk_tier": result.risk_tier,
    })
    return result


def scoring_formula() -> dict:
    """The ACTUAL weights/thresholds from risk_score.py, exposed as data so
    the frontend displays the real formula instead of hardcoding its own
    copy that could silently drift out of sync with the backend."""
    return {
        "rf_weight": RF_WEIGHT,
        "kmeans_weight": KMEANS_WEIGHT,
        "high_threshold": HIGH_THRESHOLD,
        "medium_threshold": MEDIUM_THRESHOLD,
    }


def build_feature_breakdown(features: dict) -> list[dict]:
    """
    Shared by /meters/{id}/explain and /analyze: shows WHY a feature vector
    scored the way it did — its own values next to the dataset average, plus
    which features the Random Forest model relies on most globally. This is
    an approximate, simple explanation (value vs. dataset average, weighted
    by global feature importance) — not a per-instance SHAP explanation, but
    enough to answer "why is this flagged" without adding a new ML dependency.
    """
    averages = datastore.get_dataset_feature_averages()
    rf_metrics = datastore.read_json_metrics(datastore.RF_METRICS_PATH)
    importance = rf_metrics.get("feature_importance", {}) if rf_metrics else {}

    feature_rows = []
    for col in datastore.FEATURE_COLUMNS:
        if col not in features:
            continue
        value = float(features[col])
        avg = averages.get(col)
        feature_rows.append({
            "feature": col,
            "value": round(value, 4),
            "dataset_average": avg,
            "importance": round(float(importance.get(col, 0)), 4) if importance else None,
            "above_average": (value > avg) if avg is not None else None,
        })
    # Most model-relevant features first, if we have importance; else leave as-is
    if importance:
        feature_rows.sort(key=lambda r: r["importance"] or 0, reverse=True)
    return feature_rows


@app.get("/meters/{meter_id}/explain")
def explain_meter(meter_id: str):
    """Why a STORED meter (already in features.csv) scored the way it did."""
    features = datastore.get_meter_features(meter_id)
    if not features:
        raise HTTPException(status_code=404, detail=f"No feature row found for {meter_id}")

    scores = run_models_on_features(features)
    return {
        "meter_id": meter_id,
        **scores,
        "features": build_feature_breakdown(features),
        "note": "Risk score is a prioritisation aid for human investigation, not a theft verdict.",
    }


@app.get("/meters/{meter_id}/raw-readings")
def meter_raw_readings(meter_id: str, limit: int = 200):
    """Real raw interval readings for one meter, from data/raw_readings.csv —
    used by the dashboard's Analyze Meter 'load example' button so the demo
    uses actual sample data instead of fabricated numbers."""
    rows = datastore.get_meter_raw_readings(meter_id, limit=limit)
    if not rows:
        raise HTTPException(status_code=404, detail="No raw readings available for this meter")
    return {"meter_id": meter_id, "readings": rows}


@app.get("/meters/{meter_id}/timeseries")
def meter_timeseries(meter_id: str, days: int = 30):
    """Real daily consumption totals for one meter, from data/raw_readings.csv
    (if present) — backs the dashboard's consumer detail trend chart."""
    points = datastore.meter_daily_totals(meter_id, days=days)
    if not points:
        raise HTTPException(status_code=404, detail="No raw readings available for this meter")
    return {"meter_id": meter_id, "daily": points}


@app.get("/grid/daily-load")
def grid_daily_load(days: int = 30):
    """Real feeder-level daily load, aggregated across every meter in
    data/raw_readings.csv. 404s if no raw readings file exists yet."""
    points = datastore.grid_daily_totals(days=days)
    if not points:
        raise HTTPException(status_code=404, detail="No data/raw_readings.csv found — run generate_sample_data.py or supply your dataset")
    return {"daily": points}


def _confusion_from_report(report: dict | None) -> dict | None:
    """Rebuilds the confusion matrix from the saved classification report
    (precision / recall / support per class), so no retraining is needed."""
    try:
        n0, n1 = float(report["0"]["support"]), float(report["1"]["support"])
        precision1, recall1 = float(report["1"]["precision"]), float(report["1"]["recall"])
        tp = round(recall1 * n1)
        fn = int(n1 - tp)
        predicted_pos = tp / precision1 if precision1 else 0
        fp = max(int(round(predicted_pos - tp)), 0)
        tn = int(n0 - fp)
        return {"tn": tn, "fp": fp, "fn": fn, "tp": int(tp)}
    except Exception:
        return None


@app.get("/model/insights")
def model_insights():
    """RF feature importance + K-means k/silhouette, read from the metrics
    files train_random_forest.py / train_kmeans.py write with --metrics-out."""
    rf_metrics = datastore.read_json_metrics(datastore.RF_METRICS_PATH)
    km_metrics = datastore.read_json_metrics(datastore.KMEANS_METRICS_PATH)
    report = rf_metrics.get("report") if rf_metrics else None
    return {
        "random_forest": {
            "feature_importance": rf_metrics.get("feature_importance") if rf_metrics else None,
            "f1_score": rf_metrics.get("f1_score") if rf_metrics else None,
            "roc_auc": rf_metrics.get("roc_auc") if rf_metrics else None,
            "accuracy": report.get("accuracy") if report else None,
            "per_class": {
                "normal": report.get("0"), "abnormal": report.get("1"),
            } if report else None,
            "confusion": _confusion_from_report(report),
        } if rf_metrics else None,
        "kmeans": km_metrics,
        "formula": scoring_formula(),
        "note": None if (rf_metrics and km_metrics) else
            "Train with --metrics-out (see README) to populate this.",
    }


@app.get("/audit/chain")
def audit_chain_view():
    """The whole hash chain, block by block, joined to the investigations
    that own each block. The dashboard draws the Audit chain tab from this.

    A case only remembers its latest anchor, so earlier saves of the same case
    are reported as `superseded` (still real blocks, still part of the chain)
    instead of being dropped from the picture."""
    blocks = chain_client.blocks()
    cases = datastore.list_investigations()

    revisions = {}   # payload_hash -> (case, revision dict, is_current)
    for c in cases:
        anchor = c.get("audit_anchor")
        current_hash = anchor.get("payload_hash") if anchor else None
        history = list(c.get("audit_history") or [])
        if anchor and not any(h.get("payload_hash") == current_hash for h in history):
            payload = c.get("audit_payload") or {}
            history.append({
                "payload_hash": current_hash, "chain_tx_ref": anchor.get("chain_tx_ref"),
                "outcome": payload.get("investigation_outcome"), "risk_score": payload.get("risk_score"),
                "sepolia_tx": anchor.get("sepolia_tx"), "anchored_at": anchor.get("anchored_at"),
            })
        for h in history:
            if h.get("payload_hash"):
                revisions[h["payload_hash"]] = (c, h, h["payload_hash"] == current_hash)

    enriched = []
    for b in blocks:
        item = dict(b)
        if b["index"] == 0:
            item.update({"kind": "genesis", "investigation_id": None, "meter_id": None,
                         "outcome": None, "risk_score": None, "sepolia_tx": None})
        else:
            hit = revisions.get(b["event_hash"])
            if hit:
                case, rev, is_current = hit
                item.update({
                    "kind": "current" if is_current else "superseded",
                    "investigation_id": case["investigation_id"], "meter_id": case["meter_id"],
                    "outcome": rev.get("outcome"),
                    "risk_score": rev.get("risk_score", case.get("risk_score")),
                    "sepolia_tx": rev.get("sepolia_tx"),
                })
            else:
                # An earlier save made before revision history was kept — the block
                # is real and linked, but the case it belonged to wasn't recorded.
                item.update({"kind": "superseded", "investigation_id": None, "meter_id": None,
                             "outcome": None, "risk_score": None, "sepolia_tx": None})
        enriched.append(item)

    real = [b for b in enriched if b["index"] > 0]
    gaps = [real[i]["created_at"] - real[i - 1]["created_at"] for i in range(1, len(real))]
    outcome_counts = {}
    for b in real:
        if b["kind"] == "current" and b["outcome"]:
            outcome_counts[b["outcome"]] = outcome_counts.get(b["outcome"], 0) + 1
    return {
        "blocks": enriched,
        "summary": {
            "length": len(real),
            "intact": chain_client.is_chain_intact(),
            "first_broken_index": chain_client.first_broken_index(),
            "current": sum(1 for b in real if b["kind"] == "current"),
            "superseded": sum(1 for b in real if b["kind"] == "superseded"),
            "on_sepolia": sum(1 for b in real if b.get("sepolia_tx")),
            "outcome_counts": outcome_counts,
            "avg_gap_seconds": (sum(gaps) / len(gaps)) if gaps else None,
        },
    }


_analytics_cache: dict = {"sig": None, "value": None}


def _file_sig(*paths):
    sig = []
    for p in paths:
        try:
            st = os.stat(p)
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)
    return tuple(sig)


@app.get("/analytics/summary")
def analytics_summary():
    """Aggregates over EVERY scored meter (not just the high-risk page the
    dashboard lists), for the Overview / Consumers / Model charts: score
    histograms, a stratified RF-vs-K-means scatter sample, and each tier's
    typical behavioural profile. Cached until the CSV files change."""
    sig = _file_sig(datastore.DETECTIONS_PATH, datastore.FEATURES_PATH)
    if _analytics_cache["sig"] == sig and _analytics_cache["value"] is not None:
        return _analytics_cache["value"]

    det = datastore.read_detections()
    if det.empty:
        raise HTTPException(status_code=404, detail="No detections on file yet")

    edges = np.linspace(0, 1, 21)
    labels = [f"{edges[i]:.2f}" for i in range(20)]

    def hist(series):
        return np.histogram(series.clip(0, 1), bins=edges)[0].astype(int).tolist()

    tier_hist = {t: hist(det.loc[det["risk_tier"] == t, "risk_score"]) for t in ("low", "medium", "high")}

    high = det[det["risk_tier"] == "high"]
    rest = det[det["risk_tier"] != "high"]
    sample = pd.concat([
        high.sample(min(len(high), 350), random_state=7),
        rest.sample(min(len(rest), 650), random_state=7),
    ])
    scatter = [
        {"x": round(float(r.rf_probability), 4), "y": round(float(r.kmeans_score), 4), "tier": r.risk_tier}
        for r in sample.itertuples()
    ]

    tier_means = {}
    for t in ("low", "medium", "high"):
        sub = det[det["risk_tier"] == t]
        tier_means[t] = {
            "count": int(len(sub)),
            "rf": round(float(sub["rf_probability"].mean()), 4) if len(sub) else 0,
            "km": round(float(sub["kmeans_score"].mean()), 4) if len(sub) else 0,
            "risk": round(float(sub["risk_score"].mean()), 4) if len(sub) else 0,
        }

    profile = None
    feats = datastore.read_features_all()
    if not feats.empty:
        cols = [c for c in datastore.FEATURE_COLUMNS if c in feats.columns]
        merged = feats.merge(det[["meter_id", "risk_tier"]], on="meter_id", how="inner")
        if cols and not merged.empty:
            ranks = merged[cols].rank(pct=True) * 100
            ranks["risk_tier"] = merged["risk_tier"].values
            profile = {"features": cols}
            for t in ("low", "medium", "high"):
                sub = ranks[ranks["risk_tier"] == t]
                profile[t] = [round(float(sub[c].mean()), 1) for c in cols] if len(sub) else [50.0] * len(cols)

    result = {
        "total": int(len(det)),
        "bins": labels,
        "risk_hist": hist(det["risk_score"]),
        "rf_hist": hist(det["rf_probability"]),
        "km_hist": hist(det["kmeans_score"]),
        "tier_hist": tier_hist,
        "scatter": scatter,
        "tier_means": tier_means,
        "correlation_rf_km": round(float(det["rf_probability"].corr(det["kmeans_score"])), 4),
        "tier_profile": profile,
    }
    _analytics_cache["sig"], _analytics_cache["value"] = sig, result
    return result


@app.get("/stats")
def stats():
    """Aggregate counts for the Overview tab — computed live from
    data/detections.csv, data/investigations.json and the audit chain,
    so these numbers always match whatever dataset is actually loaded."""
    df = datastore.read_detections()
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    if not df.empty:
        risk_counts.update(df["risk_tier"].value_counts().to_dict())

    investigations = datastore.list_investigations()
    open_count = sum(1 for c in investigations if c.get("outcome") == "pending")

    return {
        "total_meters": int(len(df)),
        "risk_counts": risk_counts,
        "open_investigations": open_count,
        "total_investigations": len(investigations),
        "chain_length": max(len(chain_client._chain) - 1, 0),  # exclude genesis block
        "chain_intact": chain_client.is_chain_intact(),
    }


class ReadingIn(BaseModel):
    timestamp: str
    consumption_kwh: float


class AnalyzeMeterIn(BaseModel):
    meter_id: Optional[str] = None
    category: Optional[str] = None  # display/context only — NOT a model feature, never fed to RF or K-means
    readings: list[ReadingIn]


@app.post("/analyze")
def analyze_meter(payload: AnalyzeMeterIn):
    """
    'Analyze Meter' — runs the ACTUAL existing pipeline (feature_engineering.py's
    clean_readings + build_features, then the same trained RF + K-means models
    everything else uses) on raw interval readings the user supplies, rather
    than the simplified 3-input proxy /consumers/register uses. Nothing here
    is persisted to data/features.csv or data/detections.csv — this is a
    one-off "what if" analysis, not meter registration.
    """
    if len(payload.readings) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 readings to compute any features")

    meter_id = payload.meter_id or "ANALYSIS"
    df = pd.DataFrame([
        {"meter_id": meter_id, "timestamp": r.timestamp, "consumption_kwh": r.consumption_kwh}
        for r in payload.readings
    ])
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse one or more timestamps")

    cleaned = clean_readings(df)  # the real Step 2 cleaning — same function feature_engineering.py's CLI uses
    feat_df = build_features(cleaned)  # the real Step 3 feature extraction
    if feat_df.empty:
        raise HTTPException(status_code=400, detail="Could not compute features from the given readings")

    features = feat_df.iloc[0].to_dict()
    scores = run_models_on_features(features)  # the real Steps 4-6 — same models every other endpoint uses

    warnings = []
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
    if span_days < 7:
        warnings.append(f"Only {span_days} day(s) of data — trend and deviation-from-history features are unreliable with this little history.")
    if len(df) < 24:
        warnings.append(f"Only {len(df)} reading(s) provided — results may be noisy. More daily readings (ideally several weeks) give a more reliable score.")

    return {
        "meter_id": meter_id,
        "category": payload.category,  # echoed for display only — the model never sees this
        **scores,
        "feature_values": {k: round(float(v), 4) for k, v in features.items() if k != "meter_id"},
        "features": build_feature_breakdown(features),
        "formula": scoring_formula(),
        "reading_count": len(payload.readings),
        "warnings": warnings,
        "note": "Risk score is a prioritisation aid for human investigation, not a theft verdict. This analysis is not saved to the dataset.",
    }


class RegisterConsumerIn(BaseModel):
    meter_id: Optional[str] = None
    avg_daily_kwh: float
    max_kwh: float
    min_kwh: float
    variance_kwh: float
    weekend_weekday_ratio: float
    sudden_change_score: float
    trend_slope: float
    deviation_from_history: float


@app.post("/consumers/register", response_model=RiskScoreResponse)
def register_consumer(payload: RegisterConsumerIn):
    """
    Registers a new meter through the real pipeline. The row uses the eight
    columns the SGCC model was trained on. Name, category, and night_day_ratio
    are not accepted: this file has none of them, and writing night_day_ratio
    would add a column the rest of features.csv does not have.
    """
    meter_id = (payload.meter_id or "").strip() or secrets.token_hex(17).upper()
    if datastore.get_meter_features(meter_id):
        raise HTTPException(status_code=409, detail=f"{meter_id} already exists")

    feature_row = {
        "meter_id": meter_id,
        "avg_daily_kwh": payload.avg_daily_kwh,
        "max_kwh": payload.max_kwh,
        "min_kwh": payload.min_kwh,
        "variance_kwh": payload.variance_kwh,
        "weekend_weekday_ratio": payload.weekend_weekday_ratio,
        "sudden_change_score": payload.sudden_change_score,
        "trend_slope": payload.trend_slope,
        "deviation_from_history": payload.deviation_from_history,
    }
    datastore.append_feature_row(feature_row)

    scores = run_models_on_features(feature_row)
    result = RiskScoreResponse(
        meter_id=meter_id,
        rf_probability=scores["rf_probability"],
        kmeans_anomaly_score=scores["kmeans_anomaly_score"],
        risk_score=scores["risk_score"],
        risk_tier=scores["risk_tier"],
    )
    datastore.upsert_detection({
        "meter_id": meter_id, "rf_probability": result.rf_probability,
        "kmeans_score": result.kmeans_anomaly_score, "risk_score": result.risk_score,
        "risk_tier": result.risk_tier,
    })
    return result


@app.get("/investigations")
def list_investigations():
    return datastore.list_investigations()


@app.post("/investigations/open")
def open_investigation(payload: SendToInvestigationIn):
    """Step 8: a high-risk meter is sent for human investigation.
    Replaces INSERT INTO theft_alert / investigation."""
    if not datastore.get_detection(payload.meter_id):
        raise HTTPException(status_code=404, detail=f"No detection on file for {payload.meter_id}")
    case = datastore.create_investigation(
        meter_id=payload.meter_id,
        inspector=payload.inspector,
        opened_at=dt.date.today().isoformat(),
    )
    return {"status": "opened", "investigation": case}


@app.post("/investigations/outcome")
def record_investigation_outcome(payload: InvestigationOutcomeIn):
    """
    Step 8: an inspector records the result of a field investigation.
    Step 9: the event is then hashed and appended to the audit trail.
    Replaces the old MySQL UPDATE + audit_anchor INSERT.
    """
    case = datastore.update_investigation_outcome(payload.investigation_id, payload.outcome, payload.notes)
    if not case:
        raise HTTPException(status_code=404, detail="Investigation not found")

    event = AuditEvent(
        alert_id=case["alert_id"],
        investigation_id=payload.investigation_id,
        risk_score=float(case["risk_score"]) if case["risk_score"] is not None else 0.0,
        model_version="rf-v2.3",
        investigation_outcome=payload.outcome,
        timestamp=dt.datetime.utcnow().isoformat(),
    )
    anchor = chain_client.submit(event)
    # Local chain is the record. Sepolia only receives the hash, and a missing
    # .env must not lose the investigation.
    try:
        anchor["sepolia_tx"] = anchor_hash(anchor["payload_hash"])
    except SepoliaNotConfigured:
        anchor["sepolia_tx"] = None
    except Exception as exc:
        anchor["sepolia_tx"] = None
        anchor["sepolia_error"] = str(exc)

    # Every save appends a NEW block, but a case only keeps its latest anchor.
    # Keep a revision list too, so earlier blocks can still be traced back to
    # this case (the Audit chain tab shows them as "superseded", not missing).
    history = list(case.get("audit_history") or [])
    old_anchor = case.get("audit_anchor")
    if old_anchor and not any(h.get("payload_hash") == old_anchor.get("payload_hash") for h in history):
        old_payload = case.get("audit_payload") or {}
        history.append({
            "payload_hash": old_anchor.get("payload_hash"),
            "chain_tx_ref": old_anchor.get("chain_tx_ref"),
            "outcome": old_payload.get("investigation_outcome"),
            "risk_score": old_payload.get("risk_score"),
            "sepolia_tx": old_anchor.get("sepolia_tx"),
            "anchored_at": old_anchor.get("anchored_at"),
        })
    history.append({
        "payload_hash": anchor["payload_hash"],
        "chain_tx_ref": anchor["chain_tx_ref"],
        "outcome": payload.outcome,
        "risk_score": event.risk_score,
        "sepolia_tx": anchor.get("sepolia_tx"),
        "anchored_at": anchor["anchored_at"],
    })
    case["audit_history"] = history

    # Store the anchor + the exact payload alongside the case, so /audit/verify
    # can recompute the same hash later without guessing what was hashed.
    case["audit_anchor"] = anchor
    case["audit_payload"] = {
        "alert_id": event.alert_id, "investigation_id": event.investigation_id,
        "risk_score": event.risk_score, "model_version": event.model_version,
        "investigation_outcome": event.investigation_outcome, "timestamp": event.timestamp,
    }
    cases = datastore.list_investigations()
    for i, c in enumerate(cases):
        if c["investigation_id"] == payload.investigation_id:
            cases[i] = case
    datastore._save_investigations(cases)

    return {"status": "recorded", "anchor": anchor}


@app.get("/audit/{investigation_id}/verify", response_model=AuditVerifyResponse)
def verify_audit_record(investigation_id: int):
    """Recomputes the hash of the stored event payload and compares it to
    what's anchored in the hash chain. Replaces the old MySQL audit_anchor JOIN."""
    case = datastore.get_investigation(investigation_id)
    if not case or "audit_anchor" not in case:
        raise HTTPException(status_code=404, detail="No anchored record for this investigation")

    event = AuditEvent(**case["audit_payload"])
    verified = chain_client.verify(event, case["audit_anchor"]["chain_tx_ref"])
    sepolia_tx = case["audit_anchor"].get("sepolia_tx")
    sepolia_anchored = None
    if sepolia_tx:
        try:
            sepolia_anchored = is_anchored(case["audit_anchor"]["payload_hash"])
        except Exception:
            sepolia_anchored = None
    return AuditVerifyResponse(
        investigation_id=investigation_id,
        verified=verified,
        payload_hash=case["audit_anchor"]["payload_hash"],
        chain_tx_ref=case["audit_anchor"]["chain_tx_ref"],
        sepolia_tx=sepolia_tx,
        sepolia_anchored=sepolia_anchored,
    )
