"""Build transaction graph (nodes, edges, account maps) and detect patterns (circular, hub, rapid)."""

from collections import defaultdict
from typing import Any

import pandas as pd


def build_graph_from_raw(df: pd.DataFrame) -> tuple[list[dict], list[dict], dict[str, int], dict[int, str]]:
    """
    Build graph from raw_df (Account, Account.1 as strings).
    Returns (graph_nodes, graph_edges, account_to_id, id_to_account).
    """
    from_col, to_col = "Account", "Account.1"
    if from_col not in df.columns or to_col not in df.columns:
        raise ValueError(f"DataFrame must contain {from_col} and {to_col}")
    accounts = pd.Index(
        pd.concat([df[from_col].astype(str), df[to_col].astype(str)], ignore_index=True).unique()
    )
    account_to_id = {str(a): i for i, a in enumerate(accounts)}
    id_to_account = {i: str(a) for a, i in account_to_id.items()}
    graph_nodes = [{"id": str(acc), "label": str(acc)} for acc in accounts]
    amount_col = "Amount Paid" if "Amount Paid" in df.columns else "Amount Received"
    edges = []
    for _, row in df.iterrows():
        src = str(row[from_col])
        dst = str(row[to_col])
        amount = float(row.get(amount_col, 0))
        edges.append({
            "from": src, "to": dst, "amount": amount,
            "from_id": account_to_id.get(src), "to_id": account_to_id.get(dst),
        })
    return graph_nodes, edges, account_to_id, id_to_account


def _find_circular_accounts(edges: list[dict]) -> set[str]:
    out_adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        out_adj[e["from"]].add(e["to"])
    in_circle = set()
    for src, dst_set in out_adj.items():
        for dst in dst_set:
            if src in out_adj.get(dst, set()):
                in_circle.add(src)
                in_circle.add(dst)
    return in_circle


def _find_hub_accounts(edges: list[dict], top_frac: float = 0.02) -> set[str]:
    degree: dict[str, int] = defaultdict(int)
    for e in edges:
        degree[e["from"]] += 1
        degree[e["to"]] += 1
    if not degree:
        return set()
    sorted_accs = sorted(degree.keys(), key=lambda a: -degree[a])
    k = max(1, int(len(sorted_accs) * top_frac))
    return set(sorted_accs[:k])


def _find_rapid_movement_accounts(edges: list[dict], min_tx: int = 10) -> set[str]:
    tx_count: dict[str, int] = defaultdict(int)
    for e in edges:
        tx_count[e["from"]] += 1
        tx_count[e["to"]] += 1
    return {a for a, c in tx_count.items() if c >= min_tx}


def detect_patterns(
    graph_edges: list[dict],
    account_to_id: dict[str, int],
) -> dict[str, list[str]]:
    """Return account_id -> list of pattern labels (circular, hub, rapid_movement)."""
    circular = _find_circular_accounts(graph_edges)
    hubs = _find_hub_accounts(graph_edges)
    rapid = _find_rapid_movement_accounts(graph_edges)
    result: dict[str, list[str]] = defaultdict(list)
    for acc in account_to_id.keys():
        if acc in circular:
            result[acc].append("circular")
        if acc in hubs:
            result[acc].append("hub")
        if acc in rapid:
            result[acc].append("rapid_movement")
    return dict(result)


def txs_to_graph_for_api(
    transactions: pd.DataFrame,
    account_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build nodes/edges for API graph (account, transaction, device, merchant, ip).
    If account_id is set, filter to subgraph for that account.
    """
    if transactions is None or len(transactions) == 0:
        return [], []
    df = transactions.copy()
    if account_id is not None:
        acc_col = "account_id" if "account_id" in df.columns else "Account"
        if acc_col in df.columns:
            df = df[df[acc_col].astype(str) == str(account_id)]
        if len(df) == 0:
            return [], []
    nodes: list[dict[str, Any]] = []
    node_ids: set = set()
    edges: list[dict[str, Any]] = []
    def add_node(nid: str, ntype: str, label: str):
        if nid not in node_ids:
            node_ids.add(nid)
            nodes.append({"id": nid, "type": ntype, "label": str(label)})
    def add_edge(source: str, target: str, etype: str, weight: int = 1):
        edges.append({"source": source, "target": target, "type": etype, "weight": weight})
    acc_col = "account_id" if "account_id" in df.columns else "Account"
    for idx, row in df.iterrows():
        tx_id = str(row.get("transaction_id", f"tx_{idx}"))
        acc_id = str(row.get(acc_col, row.get("Account", "")))
        add_node(f"account_{acc_id}", "account", acc_id)
        add_node(tx_id, "transaction", tx_id)
        add_edge(f"account_{acc_id}", tx_id, "initiated", 1)
        dev = str(row.get("Payment Format", row.get("Payment_Format", "D0")))
        add_node(f"device_{dev}", "device", dev)
        add_edge(tx_id, f"device_{dev}", "uses", 1)
        merch = str(row.get("Account.1", row.get("merchant", "M0")))
        add_node(f"merchant_{merch}", "merchant", merch)
        add_edge(tx_id, f"merchant_{merch}", "purchased_at", 1)
        ip_val = str(row.get("From Bank", row.get("To Bank", "0")))
        add_node(f"ip_{ip_val}", "ip", ip_val)
        add_edge(tx_id, f"ip_{ip_val}", "from_ip", 1)
    return nodes, edges
