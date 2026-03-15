# GenAI-Genesis — Abstract

**GenAI-Genesis** is a **Palantir-style AML (anti–money laundering) investigation platform** that combines **graph neural networks (GNN)**, **Railtracks** multi-agent reasoning, **IBM Watsonx** AI explanations, and an **Investigator Copilot** for case-driven financial crime analysis. The system is organized as a **backend (FastAPI)** plus **frontend (Next.js)** application with a well-defined data pipeline, case management, and persistence (SQLite or IBM Db2).

---

## What It Does

- **Ingests** transaction data (IBM AML / Kaggle CSVs from `datasets/ibm_aml/`, supporting Small/Medium/Large datasets).
- **Builds a transaction graph** (accounts as nodes, payments as edges) and **detects patterns** (circular flows, hub accounts, rapid movement).
- **Scores accounts** with a trained **GNN** (GraphSAGE/GCN/GAT) to produce per-account risk scores.
- **Runs Railtracks agents** (Pattern, Risk, Investigator) on **flagged accounts** to generate structured multi-agent explanations via LiteLLM/Gemini.
- **Uses Watsonx** for per-account **AI explanations** (cached in DB) when users request account details via the API.
- **Generates investigation cases** from flagged accounts — each case includes subgraph extraction, account role assignment (source/mule/hub/sink), transaction timelines, pattern analysis, and AI-generated summaries.
- **Provides an AI Copilot** that answers investigator questions about cases (summarize, explain patterns, suggest next steps, draft SAR-ready narratives).
- **Exposes** flagged accounts, graph data, case management, and account details via REST; supports **PDF/LaTeX export** of flagged-accounts and case reports.
- **Serves a Next.js dashboard** for 3D network graph visualization, account search, and sidebar details that consumes the backend API.

---

## Pipeline (End-to-End)

```
Data → Loader → Preprocess → Graph build → GNN scoring → Persist → Railtracks → Case generation → API / Frontend
```

| Step | Component | Description |
|------|-----------|-------------|
| 1. **Loader** | `app.pipeline.loader` | Reads from `datasets/ibm_aml` or a given CSV path. |
| 2. **Preprocess** | `app.pipeline.preprocess` | Validates schema, builds temporal/amount features, adds `transaction_id` / `account_id`. |
| 3. **Graph** | `app.pipeline.graph_builder` | Builds transaction graph (nodes, edges); detects **circular**, **hub**, **rapid_movement** patterns. |
| 4. **GNN** | `app.pipeline.gnn_runner` + `app.models.gnn_models` | Loads checkpoint (e.g. GraphSAGE), scores accounts; outputs risk scores and optional top features. |
| 5. **Persist** | `app.services.db_client` | Saves predictions to SQLite/Db2 and writes `predictions.parquet` under `backend/outputs/`. |
| 6. **Railtracks** | `app.pipeline.railtracks_explainer` | Pattern / Risk / Investigator agents (LiteLLM/Gemini) analyze flagged accounts. |
| 7. **Watsonx** | `app.services.watsonx_client` + account API | On-demand AI explanation for `GET /accounts/{id}`; results cached in DB. |
| 8. **Cases** | `app.services.case_builder` + `case_store` | Auto-generates investigation cases from flagged accounts — subgraph extraction, role assignment, priority scoring, timeline construction, AI summaries. |
| 9. **Copilot** | `app.services.copilot_agent` | AI-powered Q&A over case data — summarize, explain, suggest next steps, draft SAR narratives. |
| 10. **API** | `app.main` + `app.api.*` | REST: pipeline run, flagged list, graph, account detail, case CRUD, copilot, health, export. |
| 11. **Frontend** | Next.js app | Dashboard with 3D network graph, sidebar, search; calls backend API. |

---

## Repo Layout (Summary)

- **`backend/`** — FastAPI app, pipeline (loader → preprocess → graph → GNN → Railtracks → cases), models, services (DB, Watsonx, case store, copilot, export), scripts, tests, Dockerfile.
- **`frontend/`** — Next.js UI (dashboard, `NetworkGraph`, `Sidebar`, API client in `lib/api.ts`).
- **`datasets/ibm_aml/`** — IBM AML CSV data (HI/LI-Small/Medium/Large transaction datasets).
- **`notebooks/`** — Jupyter notebooks for GNN training, inference, and Railtracks analysis.
- **`backend/model/`** — GNN checkpoints (`.pkl`); **`backend/outputs/`** — DB file, parquet, logs, case JSON files.

---

## Main APIs

### Core Pipeline
- **POST /pipeline/run** — Run full pipeline (data → GNN → Railtracks → case generation); caches result for `/flagged` and `/graph`.
- **GET /flagged** — Flagged accounts from last run (with Railtracks summaries).
- **GET /graph/{id}** — Graph (nodes, edges); optional `id` for account subgraph.
- **GET /accounts/{id}** — Account flag (NORMAL | SUSPICIOUS | LAUNDERING) + Watsonx explanation.
- **GET /health** — DB and model status.

### Case Management
- **POST /cases/generate** — Auto-generate cases from pipeline results.
- **POST /cases/create/{account_id}** — Manually create a case for a specific account.
- **GET /cases** — List cases with status/priority filters.
- **GET /cases/{case_id}** — Full case detail (accounts, patterns, timeline, notes).
- **PATCH /cases/{case_id}** — Update case status or priority.
- **POST /cases/{case_id}/notes** — Add investigator notes.
- **GET /cases/{case_id}/graph** — Case subgraph visualization data.
- **GET /cases/{case_id}/timeline** — Transaction timeline with event classification.
- **POST /cases/{case_id}/copilot** — Ask the AI copilot about a case.

### Export
- **GET /export/flagged-accounts/pdf** — PDF report of flagged accounts.
- **GET /export/flagged-accounts/latex** — LaTeX report of flagged accounts.
- **GET /cases/{case_id}/export** — Case investigation report (PDF).

---

## Technologies

- **Backend:** Python 3.12, FastAPI, PyTorch + PyTorch Geometric (GNN), SQLite / IBM Db2, IBM Watsonx.ai, LiteLLM (Google Gemini), ReportLab (PDF/LaTeX).
- **Frontend:** Next.js 16, React 19, TypeScript 5, Tailwind CSS 4, react-force-graph-3d, three.js.
- **Data:** CSV (IBM AML schema); parquet for predictions; JSON for case storage; config via `.env`.
- **Deploy:** Docker, IBM Code Engine (port 8080).

---

## Investigator Copilot

The platform's signature feature is the **Investigator Copilot** — an AI-driven case management and investigation system layered on top of the GNN pipeline:

- **Case Builder** — Extracts subgraphs around flagged accounts, assigns roles (source, mule, hub, sink), detects money flow paths, and computes case priority (CRITICAL/HIGH/MEDIUM/LOW).
- **Timeline Builder** — Classifies transactions into event types (source deposit, split, recombination, layering transfer, terminal withdrawal) with significance scoring.
- **AI Copilot** — Context-aware Q&A that answers investigator questions about any case, including pattern explanation, next-step recommendations, and SAR-ready narrative drafting.
- **Case Store** — JSON-file-based persistence (`backend/outputs/cases/`) for lightweight, self-contained case records with full audit trail (notes, status changes).

This transforms GenAI-Genesis from a detection tool into a **full investigation platform** — flagging suspicious activity, building structured cases, and assisting investigators with AI-powered analysis.
