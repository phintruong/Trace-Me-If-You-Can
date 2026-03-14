# GenAI-Genesis

AML (anti–money laundering) fraud detection: **GNN + Railtracks + Watsonx**, with a clear **backend / frontend** layout.

## Structure

```
GenAI-Genesis/
├── backend/          # FastAPI, pipeline, GNN, DB, Watsonx
│   ├── app/
│   │   ├── api/      # routes
│   │   ├── pipeline/ # loader → preprocess → graph → GNN → Railtracks
│   │   ├── models/   # GNN definitions
│   │   ├── services/# DB, Watsonx
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── main.py
│   ├── model/        # GNN checkpoints (.pkl)
│   ├── outputs/     # DB, parquet, logs
│   ├── scripts/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/         # Next.js dashboard
├── datasets/
│   └── ibm_aml/     # IBM AML CSV data
├── notebooks/       # Jupyter notebooks
├── README.md
├── requirements.txt # root deps if needed
└── .env             # secrets and config
```

## Pipeline flow

**Data → Preprocess → Graph → GNN → Railtracks → Watsonx → API → Frontend**

1. **Loader** – read from `datasets/ibm_aml` or a CSV path  
2. **Preprocess** – validate schema, build features, add `transaction_id` / `account_id`  
3. **Graph** – build transaction graph and detect patterns (circular, hub, rapid)  
4. **GNN** – score accounts with GraphSAGE (or GCN/GAT)  
5. **Persist** – save predictions to SQLite + parquet  
6. **Railtracks** – Pattern / Risk / Investigator agents for flagged accounts  
7. **Watsonx** – on-demand AI explanation for `GET /accounts/{id}`  

## API (backend)

- **POST /pipeline/run** – run full pipeline; caches result for `/flagged` and `/graph`  
- **GET /accounts/{id}** – account flag + Watsonx explanation  
- **GET /flagged** – flagged accounts from last run  
- **GET /graph/{id}** – graph (nodes, edges); optional `id` = account for subgraph  
- **GET /health** – DB and model status  

## Run backend

From project root:

```bash
# Install
pip install -r backend/requirements.txt

# Put IBM AML CSV in datasets/ibm_aml/ (e.g. HI-Small_Trans.csv)
# Put GNN checkpoint in backend/model/ (e.g. run_1_GraphSAGE_A+B_(Synergy).pkl)

# Run pipeline (CLI)
set PYTHONPATH=%CD%;%CD%\backend
python backend/run_pipeline.py --source ibm

# Start API
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8080
# or from root:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --app-dir backend
```

## Run frontend

```bash
cd frontend && npm install && npm run dev
```

## Environment (.env at root)

- `DATASET_SOURCE=ibm` or path to CSV  
- `MODEL_PATH=model/run_1_GraphSAGE_A+B_(Synergy).pkl` (resolved under `backend/model/`)  
- `RISK_THRESHOLD=0.7`  
- `DB_MODE=sqlite` – DB file: `backend/outputs/fraud_backend.db`  
- Watsonx: `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_MODEL_ID`  
- Optional: `GEMINI_API_KEY` for Railtracks (LiteLLM)  

## Data

- Place IBM AML CSVs in **datasets/ibm_aml/** (e.g. from Kaggle or `download_ibm_data.py`).  
- Default file: `HI-Small_Trans.csv`.

## Notebooks

- **notebooks/** – training and analysis (e.g. GNN training, Railtracks).  
- Save GNN with `state_dict` + `config` (or backend format) into **backend/model/**.
