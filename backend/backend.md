# Fraud Backend — Run & Deploy

## Required environment variables

These map 1:1 to `app.config`.

| Variable | Description |
|----------|-------------|
| `DB_CONN_STRING` | Connection string for DB (optional if `DB_MODE=sqlite`). For SQLite, path to `.db` file. For Db2, full Db2 connection string. |
| `DB_MODE` | `sqlite` (default) or `db2`. When `db2` **and** the `ibm_db` package is installed, the backend uses IBM Db2; otherwise it falls back to SQLite. |
| `MODEL_PATH` | Path to local `model.pkl` (joblib) for scoring in `app.pipeline.model_runner`. |
| `MODEL_URL` | Alternative to `MODEL_PATH`: URL of scoring microservice. Backend POSTs to `MODEL_URL + "/predict"` with JSON rows and expects `risk_scores` (and optional `top_features`). |
| `RISK_THRESHOLD` | Float in \[0,1\]. Only transactions with `risk_score >= RISK_THRESHOLD` appear in `/alerts`. Default `0.7`. |
| `WATSONX_URL` | Watsonx API base URL for HTTP fallback (optional when using the official SDK). If empty, a default IBM Cloud URL is used. |
| `WATSONX_APIKEY` | Watsonx API key (secret). Required for AI explanations. |
| `WATSONX_PROJECT_ID` | Watsonx project ID used by both SDK and HTTP client. |
| `WATSONX_MODEL_ID` | Watsonx model ID (default `granite-13b-instruct`). Also stored with each explanation row. |
| `LOG_LEVEL` | Log level for both pipeline and API. Typical values: `INFO` (default) or `DEBUG`. |
| `DATASET_SOURCE` | Inference pipeline dataset source: `"ibm"` (default, uses IBM/Kaggle dataset) or a filesystem path to a CSV file. |
| `OUTPUT_DIR` | Directory for pipeline outputs (including `predictions.parquet`). Default is project `outputs/`. |
| `CACHE_TTL_SECONDS` | Reserved for future TTL-based cache eviction. **Currently unused** by the backend code. |

**Secrets:** Do not commit `WATSONX_APIKEY`, Db2 passwords, or `DB_CONN_STRING` values. Use environment variables, `.env` files that stay local, or IBM Code Engine secrets.

---

## Choose database: SQLite (default) or IBM Db2

By default the backend uses **SQLite** with a file under `outputs/fraud_backend.db`:

- `DB_MODE=sqlite` (or unset)
- `DB_CONN_STRING` unset or path to `.db` file

To use **IBM Db2** instead, install the Db2 Python client and set:

```bash
# Install backend deps including ibm_db
pip install -r backend/requirements.txt

# Example: local Db2 instance
set DB_MODE=db2
set DB_CONN_STRING=DATABASE=MYDB;HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID=db2inst1;PWD=yourPassword;
```

For Db2 on IBM Cloud or another host, change `HOSTNAME`, `PORT`, and credentials accordingly. In production, store the `DB_CONN_STRING` in a secret (for example, IBM Code Engine secret) and expose it as an environment variable.

When `DB_MODE="db2"` but `ibm_db` is not installed, the code automatically falls back to SQLite using `DEFAULT_SQLITE_PATH`.

---

## Run locally

All commands below assume you are in the **repo root** (the directory that contains `backend/` and `src/`).

1. **Install deps (from repo root):**

   ```bash
   pip install -r backend/requirements.txt
   ```

2. **(Optional) Train and save the model for `MODEL_PATH`:**

   The training pipeline lives in `src.pipeline.runner` and writes `outputs/model.pkl`, which the backend uses via `MODEL_PATH`.

   ```bash
   # From repo root
   python -m src.pipeline.runner
   ```

   After this finishes, you can point the backend to the trained model with:

   ```bash
   set MODEL_PATH=outputs\model.pkl
   ```

   Alternatively, you can leave `MODEL_PATH` empty and configure `MODEL_URL` to call a separate scoring microservice instead.

3. **Populate DB and parquet (inference pipeline):**

   This uses `backend/run_pipeline.py` which calls `app.pipeline.run.run_pipeline` to:
   - load the dataset (from IBM/Kaggle or a CSV file),
   - preprocess it,
   - score it using `MODEL_PATH` or `MODEL_URL`,
   - write predictions to the DB and `predictions.parquet`.

   ```bash
   # Windows
   set DB_MODE=sqlite
   set RISK_THRESHOLD=0.7
   rem Optional: override data source (otherwise DATASET_SOURCE or "ibm" is used)
   rem set DATASET_SOURCE=ibm

   # Use local model if you trained it in step 2
   set MODEL_PATH=outputs\model.pkl

   python backend/run_pipeline.py --source ibm
   ```

   Notes:
   - The `--source` argument overrides `DATASET_SOURCE`. Use `--source ibm` to pull the IBM/Kaggle dataset (requires the `src` package and its data helpers).
   - When using `MODEL_URL` instead of `MODEL_PATH`, make sure the remote service implements the `/predict` contract expected by `app.pipeline.model_runner._score_remote`.

4. **Start the FastAPI backend (from repo root):**

   `app.main` defines the FastAPI app and mounts the routers for `/alerts`, `/account`, `/graph-data`, `/explain`, `/health`, and `/metrics`.

   ```bash
   # Windows: ensure both backend and project root are on PYTHONPATH so app.* and src.* import correctly
   set PYTHONPATH=%CD%\backend;%CD%
   set DB_MODE=sqlite
   set MODEL_PATH=outputs\model.pkl

   uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```

   Or, if you `cd backend` first:

   ```bash
   cd backend
   set PYTHONPATH=..
   uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```

---

## API (for frontend)

Base URL: `http://<host>:8080`

The API routes come directly from `app.main` and the `app.api.*` routers.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check: verifies DB connectivity and whether a model is configured (`MODEL_PATH` or `MODEL_URL`). Returns `{"status": "...", "db": "...", "model_configured": bool}`. |
| GET | `/alerts?limit=50` | List of high-risk alerts (`risk_score >= RISK_THRESHOLD`), ordered by `risk_score` desc and timestamp desc. |
| GET | `/account/{account_id}` | Recent transactions and a simple `trend` for an account (rising/falling/stable) based on recent risk scores. |
| GET | `/graph-data?account_id={id}` | Graph representation for visualization: `{ "nodes": [...], "edges": [...] }`, optionally filtered by `account_id`. |
| GET | `/explain/{transaction_id}?force=true` | Cached or new investigation summary powered by watsonx.ai. When `force=true`, bypasses the cache and regenerates. |
| GET | `/metrics` | Simple in-memory counters: `alerts_generated`, `watsonx_calls`, `watsonx_errors`. |

All responses include `X-Request-ID` (added by `RequestIDMiddleware`) when available. Use the `limit` query parameter on `/alerts` to keep payloads small.

---

## Deploy on IBM Code Engine

1. **Build and push image (from repo root):**

   The Dockerfile copies `backend/app`, `backend/run_pipeline.py`, and `src/`, then runs `uvicorn app.main:app` on port 8080.

   ```bash
   docker build -f backend/Dockerfile -t <registry>/fraud-backend:latest .
   docker push <registry>/fraud-backend:latest
   ```

2. **Create the Code Engine application (CLI example):**

   ```bash
   ibmcloud ce application create \
     --name fraud-backend \
     --image <registry>/fraud-backend:latest \
     --port 8080
   ```

3. **Configure environment variables and secrets (IBM Cloud console or CLI):**

   At minimum:
   - `DB_MODE`, `RISK_THRESHOLD`, `LOG_LEVEL`
   - `MODEL_PATH` **or** `MODEL_URL`
   - `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, and optionally `WATSONX_MODEL_ID`
   - `DB_CONN_STRING` when using Db2

   Use Code Engine secrets (or ConfigMaps) for:
   - `WATSONX_APIKEY`
   - `WATSONX_PROJECT_ID`
   - Any Db2 credentials embedded in `DB_CONN_STRING`

4. **Using the minimal manifest (`backend/code-engine.yaml`):**

   The repository includes a minimal manifest:

   - `backend/code-engine.yaml`

   Update:
   - `spec.image` → your registry image (for example, `<your-registry>/fraud-backend:latest`)
   - Add or adjust `env` entries or use `envFrom` secrets/ConfigMaps per your environment

   Then apply it if your cluster supports Code Engine-style manifests.

---

## Docker

You can also run the backend with plain Docker for local testing.

From repo root:

```bash
docker build -f backend/Dockerfile -t fraud-backend .
docker run \
  -p 8080:8080 \
  -e DB_MODE=sqlite \
  -e MODEL_PATH=/app/outputs/model.pkl \
  fraud-backend
```

Mount a volume if you want the SQLite DB and/or model file to persist outside the container, for example:

```bash
docker run \
  -p 8080:8080 \
  -v %CD%\outputs:/app/outputs \
  -e DB_MODE=sqlite \
  -e MODEL_PATH=/app/outputs/model.pkl \
  fraud-backend
```

Adjust the `-v` path syntax for your OS and shell as needed.
