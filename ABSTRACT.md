# GenAI-Genesis — Abstract

**GenAI-Genesis** is an **AML (anti–money laundering) fraud detection** system that combines **graph neural networks (GNN)**, **Railtracks** multi-agent explanations, and **IBM Watsonx** for on-demand AI summaries. The repo is organized as a **backend (FastAPI)** plus **frontend (Next.js)** application with a well-defined data pipeline and persistence (SQLite or IBM Db2).

---

## What It Does

- **Ingests** transaction data (e.g. IBM AML / Kaggle CSVs from `datasets/ibm_aml/`).
- **Builds a transaction graph** (accounts as nodes, payments as edges) and **detects patterns** (circular flows, hub accounts, rapid movement).
- **Scores accounts** with a trained **GNN** (GraphSAGE/GCN/GAT) to produce per-account risk scores.
- **Runs Railtracks agents** (Pattern, Risk, Investigator) on **flagged accounts** to generate structured explanations.
- **Uses Watsonx** for per-account/per-transaction **AI explanations** (cached in DB) when users request account details via the API.
- **Exposes** flagged accounts, graph data, and account details via REST; supports **PDF/LaTeX export** of flagged-accounts reports.
- **Serves a Next.js dashboard** for visualization (network graph, sidebar, account search) that consumes the backend API.

---

## Pipeline (End-to-End)

```
Data → Loader → Preprocess → Graph build → GNN scoring → Persist → Railtracks → API / Frontend
```

| Step | Component | Description |
|------|-----------|-------------|
| 1. **Loader** | `app.pipeline.loader` | Reads from `datasets/ibm_aml` or a given CSV path. |
| 2. **Preprocess** | `app.pipeline.preprocess` | Validates schema, builds features, adds `transaction_id` / `account_id`. |
| 3. **Graph** | `app.pipeline.graph_builder` | Builds transaction graph (nodes, edges); detects **circular**, **hub**, **rapid_movement** patterns. |
| 4. **GNN** | `app.pipeline.gnn_runner` + `app.models.gnn_models` | Loads checkpoint (e.g. GraphSAGE), scores accounts; outputs risk scores and optional top features. |
| 5. **Persist** | `app.services.db_client` | Saves predictions to SQLite/Db2 and writes `predictions.parquet` under `backend/outputs/`. |
| 6. **Railtracks** | `app.pipeline.railtracks_explainer` | Pattern / Risk / Investigator agents (LiteLLM/Gemini or Railtracks SDK) on flagged accounts. |
| 7. **Watsonx** | `app.services.watsonx_client` + account API | On-demand AI explanation for `GET /accounts/{id}`; results cached in DB. |
| 8. **API** | `app.main` + `app.api.*` | REST: pipeline run, flagged list, graph, account detail, health, export. |
| 9. **Frontend** | Next.js app | Dashboard with network graph, sidebar, search; calls backend for pipeline run, flagged, graph, account. |

---

## Repo Layout (Summary)

- **`backend/`** — FastAPI app, pipeline (loader → preprocess → graph → GNN → Railtracks), models, DB, Watsonx client, scripts, Dockerfile.
- **`frontend/`** — Next.js UI (dashboard, `NetworkGraph`, `Sidebar`, API client in `lib/api.ts`).
- **`datasets/ibm_aml/`** — IBM AML CSV data (e.g. `HI-Small_Trans.csv`).
- **`notebooks/`** — Jupyter notebooks for training and analysis (e.g. GNN training, Railtracks).
- **`backend/model/`** — GNN checkpoints (e.g. `.pkl`); **`backend/outputs/`** — DB file, parquet, logs.

---

## Main APIs

- **POST /pipeline/run** — Run full pipeline; caches result for `/flagged` and `/graph`.
- **GET /flagged** — Flagged accounts from last run (with Railtracks summaries).
- **GET /graph/{id}** — Graph (nodes, edges); optional `id` for account subgraph.
- **GET /accounts/{id}** — Account flag (NORMAL | SUSPICIOUS | LAUNDERING) + Watsonx explanation.
- **GET /health** — DB and model status.
- **Export** — PDF/LaTeX report of flagged accounts (with optional Watsonx explanations).

---

## Technologies

- **Backend:** Python, FastAPI, PyTorch (GNN), SQLite / IBM Db2, Watsonx, LiteLLM (Gemini), Railtracks.
- **Frontend:** Next.js, TypeScript, dynamic network graph and sidebar UI.
- **Data:** CSV (IBM AML schema); parquet for predictions; config via `.env` (e.g. `DATASET_SOURCE`, `MODEL_PATH`, `RISK_THRESHOLD`, Watsonx credentials).

This document summarizes the **entire codebase**: pipeline flow, components, repo layout, and main APIs.
