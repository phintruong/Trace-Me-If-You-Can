"""Stage 4: Build transaction graph (accounts = nodes, transactions = edges)."""

import pandas as pd

from src.pipeline.types import PipelineContext


def stage_build_graph(ctx: PipelineContext) -> PipelineContext:
    """
    Build graph from raw_df. Sets ctx.graph_nodes, ctx.graph_edges, ctx.account_to_id, ctx.id_to_account.
    Nodes: unique accounts. Edges: each transaction with from_account, to_account, amount, etc.
    """
    if ctx.raw_df is None:
        raise ValueError("Stage 1 (load_data) must run before build_graph.")
    df = ctx.raw_df
    from_col = "Account"
    to_col = "Account.1"
    if from_col not in df.columns or to_col not in df.columns:
        raise ValueError(f"raw_df must contain {from_col} and {to_col}.")
    accounts = pd.Index(
        pd.concat([df[from_col].astype(str), df[to_col].astype(str)], ignore_index=True).unique()
    )
    account_to_id = {str(a): i for i, a in enumerate(accounts)}
    id_to_account = {i: str(a) for a, i in account_to_id.items()}
    ctx.account_to_id = account_to_id
    ctx.id_to_account = id_to_account
    ctx.graph_nodes = [{"id": str(acc), "label": str(acc)} for acc in accounts]
    edges = []
    amount_col = "Amount Paid" if "Amount Paid" in df.columns else "Amount Received"
    for _, row in df.iterrows():
        src = str(row[from_col])
        dst = str(row[to_col])
        amount = float(row.get(amount_col, 0))
        edges.append({
            "from": src,
            "to": dst,
            "amount": amount,
            "from_id": account_to_id.get(src),
            "to_id": account_to_id.get(dst),
        })
    ctx.graph_edges = edges
    return ctx
