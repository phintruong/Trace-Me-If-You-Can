# GenAI-Genesis Codebase Summary

Quick reference for the AML (Anti–Money Laundering) project.

---

## Entry points

| What | How |
|------|-----|
| **Run AML pipeline (GNN + Railtracks)** | `python run_pipeline.py` (needs saved GNN model + processed_data/) |
| **Download IBM data** | `python download_ibm_data.py` (needs `kagglehub` + Kaggle auth) |
| **Dashboard API** | `uvicorn src.api.dashboard:app --reload` then POST `/api/run-pipeline` |

---

## Main pipeline (GNN + Railtracks)

Flow: **run_pipeline.py** → **src.pipeline.run_aml_pipeline** → load data → build graph → GNN inference → Railtracks agents → API output.

1. **run_pipeline.py** — Calls `src.pipeline.run_aml_pipeline.run_pipeline()`.

2. **src/pipeline/run_aml_pipeline.py** — Load data, preprocess, build graph, detect patterns, load saved GraphSAGE, run inference, then 3 Railtracks agents (Pattern, Risk, Investigator). Output in `ctx.api_output`. See **docs/GNN_TO_RAILTRACKS.md**.

3. **src/data/ibm_loader.py** — `get_dataset_path()` tries Data/ then kagglehub cache. `load_transactions()` reads CSV.

4. **src/features/engine.py** — `build_model_matrix(df)` for pipeline preprocessing (IBM schema).

5. **src/model/gnn_inference.py** — Load GraphSAGE from .pkl, run inference, return account risk scores.

6. **src/utils/logging.py** — `setup_logging()`: file log under `outputs/logs/`, console at INFO.

---

## Config (src/config.py)

| Key | Purpose |
|-----|--------|
| `PROJECT_ROOT` | Project root (parent of `src`). |
| `DATA_DIR` | `Data/` — first place to look for CSV. |
| `DATASET_DIR` | Kaggle cache path for IBM dataset. |
| `DEFAULT_DATASET_FILE` | `"HI-Small_Trans.csv"`. |
| `OUTPUT_DIR` / `LOG_DIR` | `outputs/`, `outputs/logs/`. |
| `IBM_REQUIRED_COLUMNS` | Columns required in raw IBM CSV. |
| `MODEL_FEATURE_COLUMNS` | Features used in preprocessing. |
| `GNN_MODEL_PATH`, `GNN_ARTIFACTS_DIR` | Paths for saved GraphSAGE and processed_data/. |

---

## Other modules (not used by run_pipeline.py)

- **tests/test_scorer.py**  
  Tests for aggregation, context scoring, risk buckets. Imports `src.agents.base`, `src.aggregation.scorer`, `src.rules.aml_rules` and config keys (e.g. `SCORE_WEIGHT_AGENT`, `RULE_CATEGORIES`) that **do not exist** in the current codebase — tests are for a different/planned AML scoring stack.

- **tests/test_sparse_agent.py**  
  Present in repo; purpose not summarized here.

---

## Data layout

- **IBM pipeline** expects a single CSV with columns: `Timestamp`, `From Bank`, `Account`, `To Bank`, `Account.1`, `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format`, `Is Laundering`.
- CSV is looked up in: **Data/** then **kagglehub_cache/.../ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8**.
- **download_ibm_data.py** downloads via kagglehub and copies `HI-Small_Trans.csv` into **Data/**.

---

## Dependencies (requirements.txt)

- pandas, scikit-learn, tqdm, kagglehub

Install in the project venv:  
`pip install -r requirements.txt`
