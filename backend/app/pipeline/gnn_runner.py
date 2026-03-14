"""Load GNN, build graph from DataFrame, run inference, return scored df and account risk scores."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from app.config import MODEL_PATH, MODEL_DIR, MODEL_FEATURE_COLUMNS
from app.models.gnn_models import load_gnn_model

FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS


def _build_graph_from_df(
    df: pd.DataFrame,
    input_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int], list]:
    """Build (x, edge_index, account_to_idx, account_order) from preprocessed df."""
    src_col, dst_col = "Account", "Account.1" if "Account.1" in df.columns else "Account"
    all_ids = pd.Index(df[src_col].unique().tolist() + df[dst_col].unique().tolist()).unique()
    account_order = all_ids.tolist()
    account_to_idx = {int(aid): i for i, aid in enumerate(account_order)}
    n_nodes = len(account_order)
    use_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not use_cols:
        use_cols = [c for c in df.columns if df[c].dtype in (np.number, "int64", "float64")][:input_dim]
    feat_from = df.groupby(src_col)[use_cols].mean()
    feat_to = df.groupby(dst_col)[use_cols].mean() if dst_col in df.columns else feat_from
    agg_df = feat_from.reindex(account_order).fillna(feat_to.reindex(account_order)).fillna(0)
    feat = agg_df.values.astype(np.float32)
    if feat.shape[1] < input_dim:
        pad = np.zeros((n_nodes, input_dim - feat.shape[1]), dtype=np.float32)
        feat = np.hstack([feat, pad])
    elif feat.shape[1] > input_dim:
        feat = feat[:, :input_dim]
    x = torch.from_numpy(feat)
    src_idx = df[src_col].map(account_to_idx)
    dst_idx = df[dst_col].map(account_to_idx)
    valid = src_idx.notna() & dst_idx.notna()
    edge_index = np.stack([src_idx[valid].astype(int).values, dst_idx[valid].astype(int).values], axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    return x, edge_index, account_to_idx, account_order


def run_gnn(
    df: pd.DataFrame,
    model_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Load GNN from model_path (or MODEL_PATH), build graph from df, score.
    Returns (scored_df with risk_score and top_features, account_risk_scores).
    """
    path = model_path or MODEL_PATH
    if not path or not str(path).strip():
        raise RuntimeError("MODEL_PATH is not set and no model_path provided")
    path = Path(path)
    if not path.is_absolute():
        candidate = MODEL_DIR / path.name
        path = candidate if candidate.exists() else path
    # Fallback input_dim from feature columns when checkpoint has state_dict only
    fallback_input_dim = len([c for c in FEATURE_COLUMNS if c in df.columns])
    model, input_dim = load_gnn_model(path, input_dim=fallback_input_dim or None)
    x, edge_index, account_to_idx, account_order = _build_graph_from_df(df, input_dim)
    device = next(model.parameters()).device
    with torch.no_grad():
        log_probs = model(x.to(device), edge_index.to(device))
    prob_pos = log_probs.exp()[:, 1].cpu().numpy()
    # Map node index -> account (order is account_order which may be int)
    id_to_account = {i: str(account_order[i]) for i in range(len(account_order))}
    account_risk_scores = {}
    for node_idx in range(len(prob_pos)):
        acc = id_to_account.get(node_idx)
        if acc is not None:
            if acc not in account_risk_scores or prob_pos[node_idx] > account_risk_scores[acc]:
                account_risk_scores[acc] = float(prob_pos[node_idx])
    # Map each transaction to risk of its "from" account (factorized Account)
    node_idx = df["Account"].map(account_to_idx)
    node_idx = node_idx.fillna(0).astype(int).clip(0, len(prob_pos) - 1)
    risk_scores = np.clip(prob_pos[node_idx.values], 0.0, 1.0).astype(np.float64)
    out = df.copy()
    out["risk_score"] = risk_scores
    out["top_features"] = [[]] * len(out)
    return out, account_risk_scores
