# Trace Me If You Can

### An AI-powered Anti-Money Laundering investigation platform that sees what rule-based systems can't: the network.

> **$800 billion to $2 trillion** moves through money laundering networks every year. Traditional AML tools analyze transactions in isolation, drown compliance teams in false positives, and miss the sophisticated schemes hiding in plain sight. We built something different.

**Trace Me If You Can** transforms raw transaction data into an interactive, AI-explained investigation graph. Instead of reviewing transactions one by one, investigators see the full network — accounts as nodes, transactions as edges — with every suspicious cluster scored by a Graph Neural Network and explained in plain English by a multi-agent AI pipeline.

One API call. Millions of transactions. A complete investigation dashboard.

---

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=e97v8ee3I9w">
    <img src="frontend/public/image-removebg-preview (7).png" alt="Dashboard Preview" width="700"/>
  </a>
</p>

**[Watch the full demo on YouTube](https://www.youtube.com/watch?v=e97v8ee3I9w)**

---

## How It Works

```
Raw CSV Data
     │
     ▼
┌──────────────┐
│  Preprocess   │  Validate schema, engineer 58 features
└──────┬───────┘
       ▼
┌──────────────┐
│ Graph Build   │  Accounts → nodes, Transactions → edges
└──────┬───────┘  Detect circular flows, hubs, rapid movement
       ▼
┌──────────────┐
│ GNN Inference │  GraphSAGE scores every account (0–1 risk)
└──────┬───────┘
       ▼
┌──────────────────────────────────────────────────┐
│          Railtracks Multi-Agent Pipeline          │
│  Pattern Agent → Risk Agent → Investigator Agent │
└──────┬───────────────────────────────────────────┘
       ▼
┌──────────────┐
│  IBM Watsonx  │  Synthesizes findings into human-readable explanations
└──────┬───────┘
       ▼
┌──────────────┐
│  Dashboard    │  3D network graph, AI panel, case management, PDF export
└──────────────┘
```

### The Core Loop

1. **Ingest** — Load IBM AML transaction datasets (454 MB – 16 GB+)
2. **Graph Construction** — Build a directed transaction graph with pattern detection (circular flows, hub nodes, rapid-movement accounts)
3. **GNN Scoring** — A trained GraphSAGE model produces per-account risk scores from 58-dimensional feature vectors
4. **Multi-Agent Explanation** — Three AI agents analyze flagged accounts through Railtracks orchestration
5. **AI Synthesis** — IBM Watsonx converts structured agent output into clear, investigator-ready narratives
6. **Investigation** — Explore everything through an interactive 3D network dashboard with an AI copilot

---

## Railtracks: The Intelligence Behind the Explanations

**[Railtracks](https://github.com/RailtownAI/railtracks)** is the orchestration backbone that makes our AI explanations reliable, structured, and multi-perspective. Rather than sending a single prompt to an LLM and hoping for the best, Railtracks lets us compose a **pipeline of specialized AI agents** — each with its own role, system prompt, and domain focus — that build on each other's reasoning.

This is not a wrapper around a chat API. Railtracks provides a **declarative agent graph** where we define nodes (agents), connect them into flows, and let the framework handle execution, context passing, and fallback logic. It gave us the architecture to move from "ask an LLM a question" to "run a structured, reproducible AI investigation."

### Why Railtracks Was Essential

In AML investigation, a single LLM response isn't enough. Investigators need explanations that are **specific, multi-faceted, and defensible**. A generic summary like "this account looks suspicious" is useless — compliance teams need to know *what patterns were detected*, *how severe the risk is*, and *what an investigator should do next*.

Railtracks solved this by letting us decompose the explanation task into three specialized agents that each contribute a distinct analytical lens:

```
Flagged Accounts
       │
       ▼
┌─────────────────┐
│  Pattern Agent   │  Identifies laundering typologies:
│                  │  circular transactions, hub behavior,
│                  │  rapid fund movement
└────────┬────────┘
         ▼
┌─────────────────┐
│   Risk Agent     │  Evaluates severity using score
│                  │  distributions, transaction volumes,
│                  │  and statistical anomalies
└────────┬────────┘
         ▼
┌─────────────────┐
│ Investigator     │  Synthesizes both analyses into a
│    Agent         │  concise, human-readable explanation
│                  │  for the dashboard
└─────────────────┘
```

- **Pattern Agent** — Receives flagged account data (risk scores, graph patterns, edge samples) and identifies which laundering typologies are present: circular transactions suggesting layering, hub nodes acting as aggregation points, or rapid-movement accounts indicating velocity-based fund transfers.

- **Risk Agent** — Takes the same data and evaluates severity from a quantitative angle: how extreme are the risk scores, what do the transaction statistics reveal, and how does this cluster compare to normal activity.

- **Investigator Agent** — Receives the outputs of both the Pattern and Risk agents and synthesizes them into a single, clear explanation that an investigator can read on the dashboard and act on immediately.

### Graceful Degradation

Railtracks also powers our **3-tier fallback system**. The pipeline attempts Gemini first (via LiteLLM), falls back to OpenAI GPT-4o through Railtracks' native OpenAI integration, and degrades to a static message if all LLM providers are unavailable. This means the platform never crashes due to an API outage — it always returns *something* useful.

```python
# Primary: Gemini via LiteLLM
litellm.completion(model="gemini/gemini-3-flash-preview", ...)

# Fallback: Railtracks SDK with OpenAI
agent = rt.agent_node("Analyst", llm=rt.llm.OpenAILLM("gpt-4o"), ...)
flow = rt.Flow(name="Explain", entry_point=agent)

# Last resort: static summary
"Automated explanation unavailable. Review flagged accounts manually."
```

### What Railtracks Gave Us

| Capability | How Railtracks Helped |
|---|---|
| **Multi-agent orchestration** | Declarative agent graph with typed flows — no manual prompt chaining |
| **Separation of concerns** | Each agent specializes in one analytical dimension |
| **Reproducibility** | Same input → same agent pipeline → consistent explanation structure |
| **Provider flexibility** | Swap between Gemini, OpenAI, or any LiteLLM-supported model without changing agent logic |
| **Fault tolerance** | Built-in fallback paths so explanations never fail silently |

Railtracks transformed what would have been a fragile, single-prompt hack into a **production-grade multi-agent reasoning system**. It's the difference between asking one analyst to do everything and assembling a team of specialists.

---

## Key Features

### Network Investigation
- Interactive **2D / 3D transaction network** visualization (react-force-graph + Three.js)
- Risk-colored nodes showing suspicious accounts at a glance
- Click any account to inspect its activity, connections, and AI explanation

### AI Investigation Panel
Each flagged account includes:
- Risk score from the GNN
- Transaction volume and counterparty count
- Detected graph patterns (circular, hub, rapid movement)
- AI-generated explanation of why the account was flagged

### Investigator Copilot
An AI assistant that operates over structured case data:
- Answers investigator questions about any case
- Explains detected patterns and their significance
- Suggests next investigative steps
- Drafts **SAR (Suspicious Activity Report)**-ready narratives

### Case Management
- Auto-generated investigation cases with subgraph extraction
- Role assignment (source / mule / hub / sink)
- Transaction timelines with event classification
- Priority scoring for triage

### Export
- One-click **PDF and LaTeX reports** for flagged accounts
- Case investigation report export for compliance documentation

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 16, React 19, TypeScript 5, Tailwind CSS 4, react-force-graph-2d/3d, Three.js, jsPDF, Lucide React |
| **Backend** | Python 3.12, FastAPI, Uvicorn, Pydantic 2.0 |
| **ML / AI** | PyTorch 2.0+, PyTorch Geometric 2.4+ (GraphSAGE, GCN, GAT) |
| **LLM** | IBM Watsonx.ai (Granite 8B Instruct), Google Gemini 3.0 (via LiteLLM), OpenAI GPT-4o (fallback) |
| **Orchestration** | Railtracks (multi-agent pipeline) |
| **Database** | SQLite (default), IBM Db2 (enterprise) |
| **Export** | ReportLab (PDF/LaTeX) |
| **Data** | IBM AML datasets (Kaggle), Pandas, NumPy, PyArrow |
| **Deploy** | Docker, IBM Code Engine |

---

## GNN Architecture

Three Graph Neural Network architectures, unified under a single inference interface:

| Model | Description |
|-------|-------------|
| **GraphSAGE** (primary) | Inductive learning via neighborhood sampling — scales to millions of nodes |
| **GCN** | Spectral graph convolutions for transductive node classification |
| **GAT** | Attention-weighted neighbor aggregation for learned edge importance |

**Input:** 58-dimensional feature vectors per account
- Block A: 54 behavioral features (temporal, transactional, device, location)
- Block B: 4 random walk structural features

**Output:** Per-account risk score (0–1)

### Detected Graph Patterns
- **Circular** — Bidirectional transactions between accounts (potential layering)
- **Hub** — Top 2% of accounts by degree (high-volume aggregation points)
- **Rapid Movement** — Accounts with 10+ transactions (fast fund transfers)

### Graph Analysis
- Community detection via label propagation clustering
- Role classification: Source, Collector, Mule, Hub, Distributor, Sink
- Suspicious flow detection: DFS path ranking and value flow analysis

---

## API Endpoints

### Pipeline
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/pipeline/run` | Run full pipeline (data → GNN → Railtracks → cases) |
| `GET` | `/flagged` | Flagged accounts from last run |
| `GET` | `/graph/{id}` | Graph data (nodes, edges); optional account subgraph |
| `GET` | `/accounts/{id}` | Account flag status + Watsonx explanation |
| `GET` | `/health` | DB and model status |

### Case Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/cases/generate` | Auto-generate cases from pipeline results |
| `POST` | `/cases/create/{account_id}` | Manually create a case |
| `GET` | `/cases` | List cases (filterable by status/priority) |
| `GET` | `/cases/{case_id}` | Full case detail |
| `PATCH` | `/cases/{case_id}` | Update status or priority |
| `POST` | `/cases/{case_id}/notes` | Add investigator notes |
| `GET` | `/cases/{case_id}/graph` | Case subgraph visualization |
| `GET` | `/cases/{case_id}/timeline` | Transaction timeline |
| `POST` | `/cases/{case_id}/copilot` | Ask the AI copilot about a case |

### Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/export/flagged-accounts/pdf` | PDF report of flagged accounts |
| `GET` | `/export/flagged-accounts/latex` | LaTeX report |
| `GET` | `/cases/{case_id}/export` | Case investigation report (PDF) |

---

## Project Structure

```
GenAI-Genesis/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI routes
│   │   ├── pipeline/
│   │   │   ├── railtracks_explainer.py   # 3-agent Railtracks system
│   │   │   ├── run_pipeline.py           # Pipeline orchestration
│   │   │   ├── gnn_runner.py             # GNN inference
│   │   │   ├── graph_builder.py          # Transaction graph construction
│   │   │   └── graph_analysis.py         # Community/role/flow analysis
│   │   ├── models/          # GNN architectures (GCN, GraphSAGE, GAT)
│   │   ├── services/        # Watsonx client, DB client, export service
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── main.py
│   ├── model/               # GNN checkpoints (.pkl)
│   ├── outputs/             # SQLite DB, Parquet files, logs
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── components/      # NetworkGraph, Sidebar, CaseManager
│   │   ├── lib/             # API client, mock data
│   │   └── page.tsx         # Main dashboard
│   └── public/data/         # Pre-computed nodes.csv, edges.csv
├── datasets/
│   └── ibm_aml/             # IBM AML CSV data
├── notebooks/               # GNN training, Railtracks experimentation
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 22+
- IBM AML dataset CSVs in `datasets/ibm_aml/`

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend connects to `http://localhost:8080` by default (configurable via `NEXT_PUBLIC_API_URL`).

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `WATSONX_APIKEY` | IBM Watsonx API key |
| `WATSONX_PROJECT_ID` | Watsonx project ID |
| `WATSONX_URL` | Watsonx endpoint URL |
| `GEMINI_API_KEY` | Google Gemini API key (Railtracks agents) |
| `DB_MODE` | `sqlite` (default) or `db2` |
| `RISK_THRESHOLD` | GNN flagging threshold (default: 0.7) |
| `MODEL_PATH` | Path to GNN checkpoint |
| `DATASET_SOURCE` | `ibm` or path to CSV |

---

## Data

**Source:** [IBM Anti-Money Laundering datasets](https://www.kaggle.com/) (Kaggle)

| Dataset | Size |
|---------|------|
| HI-Small | ~454 MB |
| HI-Medium | ~2 GB |
| HI-Large | ~16 GB+ |

**Schema:** `Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`

---

## Architecture

```
  ┌─────────────────────────────────────────────┐
  │           Frontend (Next.js 16)              │
  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
  │  │ 3D Graph │ │ Sidebar  │ │ Case Manager │ │
  │  └──────────┘ └──────────┘ └──────────────┘ │
  └──────────────────┬──────────────────────────┘
                     │ REST API
  ┌──────────────────▼──────────────────────────┐
  │           Backend (FastAPI)                   │
  │                                               │
  │  ┌─────────────────────────────────────────┐ │
  │  │         Pipeline Orchestrator            │ │
  │  │  CSV → Features → Graph → GNN → Scores  │ │
  │  └────────────────┬────────────────────────┘ │
  │                   ▼                           │
  │  ┌─────────────────────────────────────────┐ │
  │  │    Railtracks Multi-Agent Explainer      │ │
  │  │  Pattern → Risk → Investigator Agents   │ │
  │  └────────────────┬────────────────────────┘ │
  │                   ▼                           │
  │  ┌────────────┐ ┌──────────┐ ┌────────────┐ │
  │  │  Watsonx   │ │ Copilot  │ │   Export    │ │
  │  └────────────┘ └──────────┘ └────────────┘ │
  │                                               │
  │  ┌────────────────────────────────────┐      │
  │  │   SQLite / IBM Db2                  │      │
  │  └────────────────────────────────────┘      │
  └───────────────────────────────────────────────┘
```

---

## AI Use Disclosure

Approximately **70% of the codebase was generated or assisted by AI tools** during development.

AI is also a **core runtime component** of the system:
- **Graph Neural Networks** detect suspicious financial behavior within transaction networks
- **Multi-agent AI (Railtracks)** explains flagged accounts through structured, multi-perspective analysis
- **IBM Watsonx** synthesizes agent findings into human-readable investigator narratives
- **Investigator Copilot** assists with case Q&A and SAR narrative drafting

---

## What We Learned

- **Graph neural networks** detect laundering patterns that rule-based systems miss — network topology carries signal that per-transaction analysis can't capture
- **Multi-agent LLM architectures** (via Railtracks) produce better explanations than single-prompt approaches — each agent specializes and builds on the previous agent's output
- **Investigators need narratives, not numbers** — the AI explanation layer was critical for making GNN scores actionable
- **Pre-computing features** as PyTorch tensors dramatically speeds up inference for large-scale deployment

---

## What's Next

- Real-time streaming — process transactions as they arrive
- SAR auto-filing — generate complete Suspicious Activity Reports from case data
- Multi-dataset fusion — combine transaction data with KYC, device fingerprint, and geolocation signals
- Fine-tuned GNN — train on organization-specific labeled data for higher precision
- Role-based access control — multi-user support with audit logging

---

## Built With

`Python` · `FastAPI` · `PyTorch` · `PyTorch Geometric` · `GraphSAGE` · `Railtracks` · `IBM Watsonx` · `Google Gemini` · `LiteLLM` · `Next.js` · `React` · `TypeScript` · `Tailwind CSS` · `Three.js` · `SQLite` · `Docker` · `Pandas` · `NumPy`

---

<p align="center"><b>Trace Me If You Can</b> — because money laundering is a network problem, and networks deserve network solutions.</p>
