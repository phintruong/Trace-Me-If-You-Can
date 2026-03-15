# GenAI-Genesis

AML (anti-money laundering) fraud detection powered by **Graph Neural Networks**, a **multi-agent Railtracks explainer**, and **IBM Watsonx** — with a full-stack **FastAPI + Next.js** architecture.

---

## Railtracks: Multi-Agent AML Explainer

Railtracks is the core intelligence layer that turns raw GNN risk scores into human-readable explanations. It orchestrates **three specialized AI agents** in sequence, each building on the previous agent's output:

```
Flagged Accounts ──► Pattern Agent ──► Risk Agent ──► Investigator Agent ──► Dashboard Explanation
```

### Agent Roles

| Agent | Role | Output |
|-------|------|--------|
| **Pattern Agent** | Analyzes detected laundering patterns (circular flows, hub accounts, rapid movement) across flagged accounts | 2-4 sentence pattern summary |
| **Risk Agent** | Evaluates transaction severity, statistical risk, and score distributions | 2-4 sentence risk assessment |
| **Investigator Agent** | Synthesizes pattern + risk summaries into a single, human-readable explanation for the dashboard | 2-3 sentence final explanation |

### How It Works

1. Accounts flagged by the GNN are filtered by `RISK_THRESHOLD` (default 0.3), and the top 50 by risk score are selected.
2. For each account, the system extracts: risk score, detected graph patterns, and sample connections.
3. The **Pattern Agent** identifies laundering typologies — circular transactions, hub nodes, and rapid-movement accounts.
4. The **Risk Agent** assesses severity using score distributions and transaction statistics.
5. The **Investigator Agent** fuses both summaries into a concise explanation displayed on the frontend dashboard.

### Detected Graph Patterns

- **Circular** — bidirectional transactions between accounts (potential layering)
- **Hub** — top 2% of accounts by degree (high transaction volume, possible aggregation points)
- **Rapid Movement** — accounts with 10+ transactions (fast fund transfers)

### LLM Backends (graceful fallback)

1. **Google Gemini** via LiteLLM (`GEMINI_API_KEY`)
2. **OpenAI GPT-4o** via Railtracks SDK (fallback)
3. **Static message** if both are unavailable

---

## Pipeline Flow

```
Data Load ─► Preprocess ─► Graph Build ─► GNN Inference ─► Save to DB ─► Railtracks Explainer ─► Graph Analysis ─► Export
```

1. **Loader** — reads IBM AML CSVs from `datasets/ibm_aml/` or a custom CSV path
2. **Preprocess** — validates schema, engineers features, assigns `transaction_id` / `account_id`
3. **Graph Builder** — constructs the transaction graph and detects structural patterns (circular, hub, rapid)
4. **GNN Runner** — scores accounts using GraphSAGE (or GCN/GAT) with behavioral (54-dim) and random walk (4-dim) features
5. **Persist** — saves predictions to SQLite + Parquet
6. **Railtracks** — runs the 3-agent explainer on flagged accounts (see above)
7. **Graph Analysis** — community detection (label propagation), role classification (source, sink, mule, hub, collector, distributor), suspicious flow detection (DFS path ranking), and account timelines
8. **Watsonx** — on-demand AI explanation for individual accounts via `GET /accounts/{id}`

---

## Tech Stack

### Backend

| Category | Technology |
|----------|-----------|
| **Framework** | FastAPI 0.104+, Uvicorn (ASGI) |
| **Language** | Python 3.11+ |
| **Graph Neural Networks** | PyTorch 2.0+, PyTorch Geometric 2.4+ (GCN, GraphSAGE, GAT) |
| **LLM Gateway** | LiteLLM 1.0+ (Gemini, OpenAI, Watsonx) |
| **Enterprise AI** | IBM Watsonx.ai SDK (granite-13b-instruct) |
| **Data Processing** | Pandas 1.5+, NumPy, PyArrow 12+ (Parquet), Joblib 1.3+ |
| **Database** | SQLite (default), IBM Db2 3.2+ (optional) |
| **PDF Export** | ReportLab 4.0+ |
| **Validation** | Pydantic 2.0+, python-dotenv 1.0+ |
| **Testing** | Pytest 7.0+, httpx 0.25+ |
| **Containerization** | Docker (python:3.11-slim) |

### Frontend

| Category | Technology |
|----------|-----------|
| **Framework** | Next.js 16, React 19, TypeScript 5+ |
| **Graph Visualization** | react-force-graph-2d/3d 1.29, Three.js 0.183 |
| **Styling** | Tailwind CSS 4.0, PostCSS |
| **Icons** | Lucide React |
| **PDF Export** | jsPDF 4.2, jsPDF-AutoTable 5.0 |
| **Linting** | ESLint 9+ with Next.js config |

---

## Project Structure

```
GenAI-Genesis/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI routes
│   │   ├── pipeline/
│   │   │   ├── railtracks_explainer.py   # 3-agent Railtracks system
│   │   │   ├── run_pipeline.py           # pipeline orchestration
│   │   │   ├── gnn_runner.py             # GNN inference
│   │   │   ├── graph_builder.py          # transaction graph construction
│   │   │   └── graph_analysis.py         # community/role/flow analysis
│   │   ├── models/          # GNN model definitions (GCN, GraphSAGE, GAT)
│   │   ├── services/        # Watsonx client, DB service
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── main.py
│   ├── model/               # GNN checkpoints (.pkl)
│   ├── outputs/             # SQLite DB, Parquet files, logs
│   ├── scripts/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                # Next.js dashboard
├── datasets/
│   └── ibm_aml/             # IBM AML CSV data
├── notebooks/               # Jupyter notebooks (GNN training, Railtracks)
├── .env                     # secrets and config
└── README.md
```

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/pipeline/run` | Run full pipeline (GNN + Railtracks); caches results |
| `GET` | `/accounts/{id}` | Account flag status + Watsonx AI explanation |
| `GET` | `/flagged` | All flagged accounts from last pipeline run |
| `GET` | `/graph/{id}` | Transaction graph (nodes/edges); optional account subgraph |
| `GET` | `/health` | DB and model status check |

## Quick Start

### Backend

```bash
pip install -r backend/requirements.txt

# Place IBM AML CSV in datasets/ibm_aml/ (e.g. HI-Small_Trans.csv)
# Place GNN checkpoint in backend/model/ (e.g. run_1_GraphSAGE_A+B_(Synergy).pkl)

# Run pipeline (CLI)
set PYTHONPATH=%CD%;%CD%\backend
python backend/run_pipeline.py --source ibm

# Start API
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Frontend

```bash
cd frontend && npm install && npm run dev
```

## Environment Variables (.env)

| Variable | Description |
|----------|-------------|
| `DATASET_SOURCE` | `ibm` or path to CSV |
| `MODEL_PATH` | GNN checkpoint path (relative to `backend/model/`) |
| `RISK_THRESHOLD` | Flagging threshold (default 0.7) |
| `DB_MODE` | `sqlite` (default) — DB file at `backend/outputs/fraud_backend.db` |
| `WATSONX_APIKEY` | IBM Watsonx API key |
| `WATSONX_PROJECT_ID` | Watsonx project ID |
| `WATSONX_MODEL_ID` | Watsonx model ID |
| `GEMINI_API_KEY` | Google Gemini API key (for Railtracks via LiteLLM) |

## Data

- Place IBM AML CSVs in `datasets/ibm_aml/` (e.g. from Kaggle or `python backend/scripts/download_ibm_data.py`)
- Default file: `HI-Small_Trans.csv`

## Notebooks

- `notebooks/` — GNN training, analysis, and Railtracks experimentation
- Save trained GNN models with `state_dict` + `config` into `backend/model/`
