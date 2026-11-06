# Vidyut Audit — AI-Based Electricity Theft Investigation & Audit System

No databases, no Docker, no React. CSV/JSON files + FastAPI + a plain
HTML/CSS/JS dashboard.

## Architecture

```
Dataset/CSV → Feature Engineering → Random Forest + K-Means → Risk Scoring
    → CSV/JSON storage → FastAPI → HTML/CSS/JS Dashboard
```

The project ships with a pre-generated sample dataset and trained models
(`backend/data/`, `backend/models/`) so the API returns real numbers the
moment you start it — no training required first. Steps to regenerate
everything from scratch are below.

## If the dashboard shows "Demo data" / "Backend not connected" even though the API works

This usually isn't a bug in the dashboard — it's a browser security policy.
If you're viewing `dashboard.html` through a page served over **https**
(for example, a published preview link) rather than opening the file
directly, Chrome's **Private Network Access** policy blocks that https page
from fetching a private address like `127.0.0.1:8000` unless the server
explicitly allows it — even though `curl` or typing the URL directly in
your browser works fine, since neither of those goes through that policy.
`app/main.py` now sends the required `Access-Control-Allow-Private-Network`
header to handle this. If it's still blocked, the guaranteed-to-work setup
is: run the backend locally, then open `dashboard.html` directly from disk
(double-click it, or drag it into the browser) rather than through a
hosted link — a `file://` page isn't subject to this restriction at all.

## How the dashboard connects to the backend

`dashboard.html` tries `http://localhost:8000/health` as soon as it loads.

- **Backend reachable** → pulls real data from every endpoint below. A
  green **"● Live API"** pill appears top-right.
- **Backend not running** (e.g. you're just viewing the file, or the
  published link, with no server up) → falls back to a self-contained mock
  dataset so the UI still demos fully. An amber **"● Demo data"** pill
  appears instead.

Click the status pill to point the dashboard at a different address (e.g.
if you run the API on another port). This is stored in your browser only.

## What's in this project

```
dashboard.html              Standalone dashboard. Connects to the API if reachable,
                             otherwise runs on mock data (see above).

backend/
  app/
    main.py                  FastAPI app — every endpoint below lives here
    datastore.py              All CSV/JSON read/write logic
  ml/
    feature_engineering.py    Step 2-3: cleaning + behavioural features
    train_random_forest.py    Step 4: supervised classifier (writes rf_metrics.json)
    train_kmeans.py           Step 5: unsupervised clustering (writes kmeans_metrics.json)
    risk_score.py             Step 6: blends RF + K-means into one score/tier
    generate_sample_data.py   Synthetic household-scale dataset for testing
    seed_detections.py        Runs trained models over features.csv -> detections.csv
  blockchain/
    audit_chain.py            SHA-256 tamper-evident audit trail (not a blockchain — see below)
  data/                       features.csv, detections.csv, investigations.json,
                               audit_chain.json, raw_readings.csv, *_metrics.json
  models/                     rf_model.joblib, kmeans_model.joblib
  requirements.txt
  .env.example
```

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check the dashboard polls on load |
| `GET /consumers/high-risk?limit=` | all meters, highest risk first |
| `GET /meters/{meter_id}/score` | on-demand scoring from `features.csv` |
| `GET /meters/{meter_id}/explain` | feature values vs. dataset average + RF feature importance — the "why" behind a score |
| `GET /meters/{meter_id}/timeseries` | real daily consumption for one meter (from `raw_readings.csv`) |
| `GET /grid/daily-load` | real feeder-level daily load, aggregated across all meters |
| `GET /model/insights` | RF feature importance + K-means k/silhouette + the real scoring formula/thresholds |
| `GET /stats` | aggregate counts for the Overview tab |
| `POST /analyze` | **Analyze Meter** — runs the real `feature_engineering.py` (clean + extract) on raw readings you provide, then the real trained models. Not persisted. |
| `GET /meters/{meter_id}/raw-readings` | raw interval readings for one meter — powers the dashboard's "load example" button |
| `GET /investigations` | all investigation cases |
| `POST /investigations/open` | send a meter for human investigation |
| `POST /investigations/outcome` | record an outcome — also anchors it on the audit chain |
| `GET /audit/{investigation_id}/verify` | recompute + compare the anchored hash |
| `POST /consumers/register` | register a new meter through the **real** RF + K-means models |

## Running on Windows (from scratch)

```bat
cd theft-project\backend

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python ml\generate_sample_data.py --meters 120 --days 30 --out-dir data

python ml\feature_engineering.py --input data\raw_readings.csv --output data\features.csv

python -c "import pandas as pd; f=pd.read_csv('data/features.csv'); l=pd.read_csv('data/labels.csv'); f.merge(l,on='meter_id').to_csv('data/features_labeled.csv', index=False)"

python ml\train_random_forest.py --features data\features_labeled.csv --model-out models\rf_model.joblib --metrics-out data\rf_metrics.json

python ml\train_kmeans.py --features data\features.csv --model-out models\kmeans_model.joblib --metrics-out data\kmeans_metrics.json

python ml\seed_detections.py --features data\features.csv --rf-model models\rf_model.joblib --kmeans-model models\kmeans_model.joblib --out data\detections.csv

python -m uvicorn app.main:app --reload
```

Then open `dashboard.html` (double-click it, or drag it into a browser) —
it will detect the running API at `http://localhost:8000` automatically.
Swagger UI is at **http://localhost:8000/docs**.

## Running on macOS / Linux (from scratch)

```bash
cd theft-project/backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 ml/generate_sample_data.py --meters 120 --days 30 --out-dir data
python3 ml/feature_engineering.py --input data/raw_readings.csv --output data/features.csv

python3 -c "
import pandas as pd
f = pd.read_csv('data/features.csv')
l = pd.read_csv('data/labels.csv')
f.merge(l, on='meter_id').to_csv('data/features_labeled.csv', index=False)
"

python3 ml/train_random_forest.py --features data/features_labeled.csv --model-out models/rf_model.joblib --metrics-out data/rf_metrics.json
python3 ml/train_kmeans.py --features data/features.csv --model-out models/kmeans_model.joblib --metrics-out data/kmeans_metrics.json

python3 ml/seed_detections.py --features data/features.csv \
    --rf-model models/rf_model.joblib --kmeans-model models/kmeans_model.joblib \
    --out data/detections.csv

python3 -m uvicorn app.main:app --reload
```

## Already have a trained model and just want to run the API?

```bash
cd backend && python3 -m uvicorn app.main:app --reload   # macOS/Linux
cd backend && python -m uvicorn app.main:app --reload    # Windows
```

The shipped `data/` and `models/` files mean this works immediately with
no training step.

## How the pieces connect

1. Raw readings live in `data/raw_readings.csv`.
2. `feature_engineering.py` turns them into `data/features.csv`.
3. `train_random_forest.py` / `train_kmeans.py` produce model files **and**
   metrics files (`rf_metrics.json`, `kmeans_metrics.json`) — the metrics
   files are what power the "why this score" explanation and the Model
   Insights tab.
4. `seed_detections.py` runs both models over every meter and writes
   `data/detections.csv` — what `/consumers/high-risk` reads.
5. `/investigations/open` and `/investigations/outcome` read/write
   `data/investigations.json`.
6. Recording an outcome hashes the event (SHA-256) and appends it to
   `data/audit_chain.json` via `blockchain/audit_chain.py`.
   `/audit/{id}/verify` recomputes the hash from the stored record and
   compares it — a mismatch means the record was altered after the fact.
7. Registering a new meter from the dashboard (`POST /consumers/register`)
   appends a feature row to `data/features.csv`, runs it through the
   **same trained models**, and writes the result to `data/detections.csv`
   — the real pipeline, not a client-side approximation.

## Audit trail and Sepolia

`audit_chain.py` is the local SHA-256 hash chain. Each investigation outcome
links to the previous hash, so an edit is detectable. Raw readings stay in CSV.

If `backend/.env` has a Sepolia RPC URL, a testnet private key, and
`ANCHOR_CONTRACT_ADDRESS`, saving an outcome also sends that same hash to
the `AuditAnchor` contract. The dashboard links the transaction on
sepolia.etherscan.io. Without those three values, the local chain still works.

Deploy once, from `backend/`, after the URL and key are in `.env`:

```bat
python -m blockchain.deploy_anchor
```

## The exact ML reasoning (for reference)

- **Random Forest features** (9, defined once in `train_random_forest.py`'s
  `FEATURE_COLUMNS` and reused everywhere): `avg_daily_kwh`, `max_kwh`,
  `min_kwh`, `variance_kwh`, `night_day_ratio`, `weekend_weekday_ratio`,
  `sudden_change_score`, `trend_slope`, `deviation_from_history`.
- **K-means score**: distance from the point to its assigned cluster
  centroid (in standardized feature space), normalized as
  `dist / (dist + max_training_distance)` — calibrated per-model, not a
  magic constant.
- **Final risk score**: `0.55 × rf_probability + 0.45 × kmeans_anomaly_score`,
  clamped to `[0, 1]` (`backend/ml/risk_score.py`).
- **Thresholds**: `≥ 0.72` high, `≥ 0.42` medium, else low (same file).

## Analyze Meter

The dashboard's "Analyze meter" tab lets you enter a meter ID + category,
then paste or upload raw interval readings (`timestamp, consumption_kwh`) —
with an inline example, a downloadable CSV template, and a "load example
from dataset" button that pulls real sample readings. It runs them through
the **actual** `feature_engineering.py` functions (`clean_readings` +
`build_features` — the same code the batch pipeline uses, not a simplified
stand-in) and then the same trained Random Forest + K-means models
everything else uses.

The result shows: risk score, risk tier, RF probability, K-means score, all
9 computed behavioural features (each with a plain-English description),
the top 3 contributing factors (ranked by real RF feature importance when
available), and a "how was this calculated" panel that substitutes the
*actual numbers from this analysis* into the real formula —
`risk_score = rf_weight × RF_probability + kmeans_weight × Kmeans_score` —
with `rf_weight`, `kmeans_weight`, and both thresholds read live from the
server's `/model/insights` endpoint (backed by `risk_score.py`'s real
constants), not hardcoded a second time in JavaScript.

The category field and meter ID are metadata only — they are never fed to
either model. Nothing from this tab is saved to the dataset — it's a
one-off "what if" analysis. It only works when the backend is connected
(shown via the status pill); it deliberately shows a plain "connect the
backend" message rather than falling back to a client-side approximation,
since faking this would defeat the point.

While building this, feeding it a short, single-day sample surfaced a real
bug in `feature_engineering.py`: `weekend_weekday_ratio` divided by a
weekday/weekend average that could independently be `NaN` without both
sides being checked, so data that only spans one day-type produced a `NaN`
that crashed K-means scoring downstream. Fixed by guarding both sides of
the division — the shipped sample dataset was never affected (it spans 30
days, always has both), but anyone pasting a short manual sample would have
hit it.

## A note on the K-means anomaly score

Early versions of this project normalized the K-means distance with a fixed
constant, which caused almost every point to saturate at a score of 1.0.
It's now normalized against the model's own training-set distances
(`dist / (dist + max_training_distance)`, stored in the model file by
`train_kmeans.py`), which gives properly differentiated scores for both
real dataset meters and newly registered ones.

## Train on the real SGCC dataset

The file to use is the Kaggle set
[bensalem14/sgcc-dataset](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset),
`data set.csv`: 42,372 consumers, daily kWh from 2014-01-01 to 2016-10-30,
and a `FLAG` column (1 = theft). It is wide (one column per day), so it is
not the long `meter_id, timestamp, consumption_kwh` file the rest of the
pipeline reads. `prepare_sgcc.py` does that conversion.

SGCC has no hour of day. `night_day_ratio` is not computed. Filling it with
zero would look like "no night use", which the file does not say. The other
eight features (level, spread, weekend ratio, sudden change, trend,
deviation from the meter's own history) are computed from the daily totals.
Short gaps of up to 3 days are interpolated. Remaining missing days are
skipped, not treated as zero. Consumers with fewer than 30 observed days
are dropped.

```bat
pip install kaggle
kaggle datasets download -d bensalem14/sgcc-dataset -p data\sgcc_raw --unzip

cd theft-project\backend
python ml\prepare_sgcc.py --input "data\sgcc_raw\data set.csv" --out-dir data

python ml\train_random_forest.py --features data\features_labeled.csv --model-out models\rf_model.joblib --metrics-out data\rf_metrics.json
python ml\train_kmeans.py --features data\features.csv --model-out models\kmeans_model.joblib --metrics-out data\kmeans_metrics.json
python ml\seed_detections.py --features data\features.csv --rf-model models\rf_model.joblib --kmeans-model models\kmeans_model.joblib --out data\detections.csv
```

`kaggle` needs a free API token in `%USERPROFILE%\.kaggle\kaggle.json`
(Kaggle account → Settings → Create New Token). `prepare_sgcc.py` replaces
the synthetic `features.csv`, `labels.csv` and writes `raw_readings.csv`
for 200 consumers so the dashboard charts still have a series.
`--max-consumers 4000` keeps the theft ratio but trains on a subset if the
full table is too slow. `datasetsmall.csv` in the same download is only
January 2014; the same script accepts it.

The SGCC records are from the State Grid Corporation of China, not an
Indian utility. Any DISCOM-style labels in the dashboard are a display
layer, not a claim about where these consumers are.

## Notes on the synthetic sample

`generate_sample_data.py` is the fallback when the SGCC file is not
downloaded. It produces household-scale numbers (~3–9 kWh/day) with ~15%
of meters carrying an injected tampering pattern, and it does include
hourly readings, so `night_day_ratio` is a real feature on that file only.

## Human-in-the-loop, by design

The AI risk score is a prioritization aid, never an automatic verdict —
this is stated in the dashboard itself (a persistent banner, plus
footnotes throughout) and reflected in the data flow:
`detections.csv` (model output) → `investigations.json` (human case) →
human-recorded `outcome`. Nothing in this system auto-labels a consumer as
committing theft.

---

## Update: audit chain fix and analytics charts

**Audit chain tab.** Saving a case's outcome again appends a *new* block, but a case only
remembers its latest anchor, so the old tab drew one card per case and silently dropped
the earlier block (block#1 jumped to block#3, and the arrow's previous-hash pointed at a
block you couldn't see). The tab is now drawn from the real chain:

- `GET /audit/chain` returns every block with its own hash, whether its link holds, and
  which case it belongs to (`current`, `superseded`, or `genesis`).
- Saving an outcome now also records an `audit_history` list on the case, so earlier
  blocks stay attributed to it. Blocks written before this change are shown as
  "earlier save" and remain linked.
- The tab adds KPIs, a block-by-block chain with per-block hash patterns, link status on
  every arrow, a **Verify entire chain** button, a revision history, a ledger table, and
  four charts (growth, outcomes, risk at each save, time between blocks).

**More charts.** Overview (score histogram, outcomes, top-10 meters, tier profile),
Consumers (stacked score distribution, model score overlay), Investigations (KPIs,
outcomes, inspector workload, cases over time, risk per case), Model insights (confusion
matrix, precision/recall/F1, blend weights, tier radar, cumulative importance) and
Analyze meter (daily usage with 7-day average, score composition, weekday profile, features
vs dataset average).

**New endpoint:** `GET /analytics/summary` (aggregates over every scored meter, cached
until the CSVs change). `GET /model/insights` now also returns the confusion matrix and
per-class metrics rebuilt from `rf_metrics.json`.

**Other fixes.** Overview now shows the real meter and tier totals (it used to count only
the 2,000 listed meters). Investigations whose meter is outside the listed page are no
longer hidden. Models are cached in memory instead of reloaded on every `/analyze` call,
and the analyze request waits up to 60 s.
