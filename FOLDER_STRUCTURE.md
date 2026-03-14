# GenAI-Genesis — folder structure (restructured)

**Only two top-level app folders: `backend/` and `frontend/`.** No `model/`, `outputs/`, `scripts/`, or `tests/` at root—those live under `backend/`.

```
GenAI-Genesis/
├── .env
├── .gitignore
├── README.md
├── requirements.txt          # root; installs backend deps
├── download_ibm_data.py      # fetch IBM AML → datasets/ibm_aml
├── FOLDER_STRUCTURE.md
│
├── backend/                  # FastAPI, pipeline, GNN, DB (model/, outputs/, scripts/, tests/ live here)
│   ├── app/
│   │   ├── api/
│   │   │   ├── account.py    # GET /accounts/{id}
│   │   │   └── pipeline.py   # POST /pipeline/run
│   │   ├── pipeline/
│   │   │   ├── loader.py
│   │   │   ├── preprocess.py
│   │   │   ├── graph_builder.py
│   │   │   ├── gnn_runner.py
│   │   │   ├── railtracks_explainer.py
│   │   │   ├── watsonx_explainer.py
│   │   │   └── run_pipeline.py
│   │   ├── models/
│   │   │   └── gnn_models.py  # GCN, GraphSAGE, GAT + load_gnn_model
│   │   ├── services/
│   │   │   ├── db_client.py
│   │   │   └── watsonx_client.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── main.py           # FastAPI: /pipeline/run, /accounts/{id}, /flagged, /graph/{id}, /health
│   ├── model/                # GNN checkpoints (.pkl)
│   ├── outputs/              # DB, parquet, logs
│   ├── scripts/
│   │   └── save_gnn_for_pipeline.py
│   ├── tests/
│   ├── run_pipeline.py       # CLI entry
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                 # Next.js (from fraud-detect)
│   ├── app/
│   ├── public/
│   ├── package.json
│   ├── next.config.ts
│   └── tsconfig.json
│
├── datasets/
│   └── ibm_aml/             # IBM AML CSV data (e.g. HI-Small_Trans.csv)
│
└── notebooks/               # Jupyter notebooks
```

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | /pipeline/run | Run full pipeline; caches for /flagged, /graph |
| GET | /accounts/{id} | Account flag + Watsonx explanation |
| GET | /flagged | Flagged accounts from last run |
| GET | /graph/{id} | Graph (nodes, edges); optional account filter |
| GET | /health | DB and model status |
