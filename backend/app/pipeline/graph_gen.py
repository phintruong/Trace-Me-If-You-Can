"""Convert transaction rows to graph nodes and edges for visualization."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def txs_to_graph(
    transactions: pd.DataFrame,
    account_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build nodes and edges from transaction data.
    Node types: account, transaction, device, merchant, ip.
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

    nodes: List[Dict[str, Any]] = []
    node_ids: set = set()
    edges: List[Dict[str, Any]] = []

    def add_node(nid: str, ntype: str, label: str):
        if nid not in node_ids:
            node_ids.add(nid)
            nodes.append({"id": nid, "type": ntype, "label": str(label)})

    def add_edge(source: str, target: str, etype: str, weight: int = 1):
        edges.append({"source": source, "target": target, "type": etype, "weight": weight})

    tx_col = "transaction_id" if "transaction_id" in df.columns else None
    acc_col = "account_id" if "account_id" in df.columns else "Account"
    for idx, row in df.iterrows():
        tx_id = str(row.get("transaction_id", f"tx_{idx}"))
        acc_id = str(row.get(acc_col, row.get("Account", "")))
        add_node(f"account_{acc_id}", "account", acc_id)
        add_node(tx_id, "transaction", tx_id)
        add_edge(f"account_{acc_id}", tx_id, "initiated", 1)

        # device: from Payment Format or placeholder
        dev = str(row.get("Payment Format", row.get("Payment_Format", "D0")))
        dev_id = f"device_{dev}"
        add_node(dev_id, "device", dev)
        add_edge(tx_id, dev_id, "uses", 1)

        # merchant: Account.1 (counterparty)
        merch = str(row.get("Account.1", row.get("merchant", "M0")))
        merch_id = f"merchant_{merch}"
        add_node(merch_id, "merchant", merch)
        add_edge(tx_id, merch_id, "purchased_at", 1)

        # ip/location proxy: From Bank or To Bank
        ip_val = str(row.get("From Bank", row.get("To Bank", "0")))
        ip_id = f"ip_{ip_val}"
        add_node(ip_id, "ip", ip_val)
        add_edge(tx_id, ip_id, "from_ip", 1)

    return nodes, edges
