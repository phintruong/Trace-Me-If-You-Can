"""Graph analysis: community detection, role classification, suspicious flow detection."""

from collections import defaultdict
from typing import Any

import random


# ---------------------------------------------------------------------------
# 1. Community / cluster detection  (label propagation — no extra deps)
# ---------------------------------------------------------------------------

def detect_communities(
    graph_nodes: list[dict],
    graph_edges: list[dict],
    account_risk_scores: dict[str, float],
    max_iterations: int = 20,
) -> dict[str, Any]:
    """
    Label-propagation community detection.
    Returns {
        "clusters": { cluster_id: { "accounts": [...], "edges": [...], "risk_score": float } },
        "account_cluster": { account_id: cluster_id },
    }
    """
    # Build adjacency
    adj: dict[str, set[str]] = defaultdict(set)
    for e in graph_edges:
        adj[e["from"]].add(e["to"])
        adj[e["to"]].add(e["from"])

    # Init: each node gets its own label
    labels: dict[str, str] = {n["id"]: n["id"] for n in graph_nodes}
    node_ids = list(labels.keys())

    for _ in range(max_iterations):
        changed = False
        random.shuffle(node_ids)
        for node in node_ids:
            neighbors = adj.get(node, set())
            if not neighbors:
                continue
            # Count neighbor labels
            label_counts: dict[str, int] = defaultdict(int)
            for nb in neighbors:
                label_counts[labels[nb]] += 1
            max_count = max(label_counts.values())
            candidates = [lb for lb, c in label_counts.items() if c == max_count]
            best = min(candidates)  # deterministic tie-break
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break

    # Group into clusters
    cluster_members: dict[str, list[str]] = defaultdict(list)
    for acc, label in labels.items():
        cluster_members[label].append(acc)

    # Assign integer cluster IDs, sorted by size desc
    sorted_labels = sorted(cluster_members.keys(), key=lambda lb: -len(cluster_members[lb]))
    label_to_id: dict[str, int] = {lb: i for i, lb in enumerate(sorted_labels)}

    account_cluster: dict[str, int] = {}
    clusters: dict[int, dict[str, Any]] = {}

    for label, members in cluster_members.items():
        cid = label_to_id[label]
        for acc in members:
            account_cluster[acc] = cid
        # Edges within this cluster
        member_set = set(members)
        cluster_edges = [
            e for e in graph_edges
            if e["from"] in member_set and e["to"] in member_set
        ]
        # Aggregate risk
        risks = [account_risk_scores.get(acc, 0.0) for acc in members]
        avg_risk = sum(risks) / len(risks) if risks else 0.0
        max_risk = max(risks) if risks else 0.0
        clusters[cid] = {
            "cluster_id": cid,
            "accounts": sorted(members),
            "edges": cluster_edges,
            "size": len(members),
            "avg_risk": round(avg_risk, 4),
            "max_risk": round(max_risk, 4),
            "risk_score": round(avg_risk * 0.4 + max_risk * 0.6, 4),
        }

    return {"clusters": clusters, "account_cluster": account_cluster}


# ---------------------------------------------------------------------------
# 2. Role classification
# ---------------------------------------------------------------------------

def classify_roles(
    graph_edges: list[dict],
    account_risk_scores: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """
    Classify each account into a role based on graph features.
    Roles: source, collector, mule, hub, distributor, sink.
    Returns { account_id: { "role": str, "fan_in": int, "fan_out": int, ... } }
    """
    fan_in: dict[str, int] = defaultdict(int)
    fan_out: dict[str, int] = defaultdict(int)
    in_value: dict[str, float] = defaultdict(float)
    out_value: dict[str, float] = defaultdict(float)

    for e in graph_edges:
        src, dst = e["from"], e["to"]
        amt = float(e.get("amount", 0))
        fan_out[src] += 1
        fan_in[dst] += 1
        out_value[src] += amt
        in_value[dst] += amt

    all_accounts = set(fan_in.keys()) | set(fan_out.keys())
    roles: dict[str, dict[str, Any]] = {}

    for acc in all_accounts:
        fi = fan_in.get(acc, 0)
        fo = fan_out.get(acc, 0)
        total_degree = fi + fo
        inv = in_value.get(acc, 0.0)
        outv = out_value.get(acc, 0.0)
        risk = account_risk_scores.get(acc, 0.0)

        # Ratio-based classification
        if total_degree == 0:
            role = "source"
        elif fi == 0 and fo > 0:
            role = "source"
        elif fo == 0 and fi > 0:
            role = "sink"
        else:
            ratio = fo / fi if fi > 0 else float("inf")
            if total_degree >= 10 and 0.3 <= ratio <= 3.0:
                role = "hub"
            elif ratio > 3.0:
                role = "distributor"
            elif ratio < 0.3:
                role = "collector"
            else:
                # Mid-range: check if value retention is low (pass-through = mule)
                retention = abs(inv - outv) / max(inv, outv, 1.0)
                if retention < 0.2 and risk >= 0.5:
                    role = "mule"
                elif ratio >= 1.0:
                    role = "distributor"
                else:
                    role = "collector"

        roles[acc] = {
            "role": role,
            "fan_in": fi,
            "fan_out": fo,
            "total_degree": total_degree,
            "in_value": round(inv, 2),
            "out_value": round(outv, 2),
            "risk_score": round(risk, 4),
        }

    return roles


# ---------------------------------------------------------------------------
# 3. Suspicious flow / path detection
# ---------------------------------------------------------------------------

def detect_flows(
    graph_edges: list[dict],
    account_risk_scores: dict[str, float],
    account_roles: dict[str, dict[str, Any]],
    max_path_length: int = 6,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Detect suspicious money-flow chains using DFS from high-risk sources.
    Returns list of paths sorted by suspiciousness score.
    """
    # Build directed adjacency with edge data
    adj: dict[str, list[dict]] = defaultdict(list)
    for e in graph_edges:
        adj[e["from"]].append(e)

    # Start from sources / high-risk accounts
    starts = []
    for acc, risk in account_risk_scores.items():
        role_info = account_roles.get(acc, {})
        role = role_info.get("role", "")
        if risk >= 0.5 or role in ("source", "distributor"):
            starts.append((acc, risk))
    starts.sort(key=lambda x: -x[1])
    starts = starts[:100]  # cap for performance

    all_paths: list[dict[str, Any]] = []

    for start_acc, _ in starts:
        # DFS
        stack = [(start_acc, [start_acc], [], 0.0)]
        visited_paths: set[str] = set()

        while stack:
            current, path, edges_in_path, total_value = stack.pop()
            if len(path) >= 3:
                path_key = "->".join(path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    risks = [account_risk_scores.get(a, 0.0) for a in path]
                    avg_risk = sum(risks) / len(risks)
                    path_score = round(
                        avg_risk * 0.4
                        + min(len(path) / max_path_length, 1.0) * 0.3
                        + min(total_value / 100000, 1.0) * 0.3,
                        4,
                    )
                    all_paths.append({
                        "accounts": list(path),
                        "transactions": list(edges_in_path),
                        "path_length": len(path),
                        "total_value": round(total_value, 2),
                        "avg_risk": round(avg_risk, 4),
                        "path_score": path_score,
                        "roles": [account_roles.get(a, {}).get("role", "unknown") for a in path],
                    })

            if len(path) >= max_path_length:
                continue

            for edge in adj.get(current, []):
                nxt = edge["to"]
                if nxt not in path:  # no cycles
                    stack.append((
                        nxt,
                        path + [nxt],
                        edges_in_path + [edge],
                        total_value + float(edge.get("amount", 0)),
                    ))

    # Sort by score desc, take top_k
    all_paths.sort(key=lambda p: -p["path_score"])
    return all_paths[:top_k]


def get_account_flows(
    account_id: str,
    graph_edges: list[dict],
    account_risk_scores: dict[str, float],
    account_roles: dict[str, dict[str, Any]],
    max_path_length: int = 6,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Get suspicious flows involving a specific account."""
    # Build directed adjacency
    adj: dict[str, list[dict]] = defaultdict(list)
    rev_adj: dict[str, list[dict]] = defaultdict(list)
    for e in graph_edges:
        adj[e["from"]].append(e)
        rev_adj[e["to"]].append(e)

    paths: list[dict[str, Any]] = []

    # Forward paths (account as source)
    stack = [(account_id, [account_id], [], 0.0)]
    while stack:
        current, path, edges_in_path, total_value = stack.pop()
        if len(path) >= 2:
            risks = [account_risk_scores.get(a, 0.0) for a in path]
            avg_risk = sum(risks) / len(risks)
            path_score = round(
                avg_risk * 0.4
                + min(len(path) / max_path_length, 1.0) * 0.3
                + min(total_value / 100000, 1.0) * 0.3,
                4,
            )
            paths.append({
                "accounts": list(path),
                "transactions": list(edges_in_path),
                "path_length": len(path),
                "total_value": round(total_value, 2),
                "avg_risk": round(avg_risk, 4),
                "path_score": path_score,
                "direction": "outgoing",
                "roles": [account_roles.get(a, {}).get("role", "unknown") for a in path],
            })
        if len(path) >= max_path_length:
            continue
        for edge in adj.get(current, []):
            nxt = edge["to"]
            if nxt not in path:
                stack.append((nxt, path + [nxt], edges_in_path + [edge], total_value + float(edge.get("amount", 0))))

    # Backward paths (account as destination)
    stack = [(account_id, [account_id], [], 0.0)]
    while stack:
        current, path, edges_in_path, total_value = stack.pop()
        if len(path) >= 2:
            risks = [account_risk_scores.get(a, 0.0) for a in path]
            avg_risk = sum(risks) / len(risks)
            path_score = round(
                avg_risk * 0.4
                + min(len(path) / max_path_length, 1.0) * 0.3
                + min(total_value / 100000, 1.0) * 0.3,
                4,
            )
            paths.append({
                "accounts": list(path[::-1]),  # reverse for natural order
                "transactions": list(edges_in_path[::-1]),
                "path_length": len(path),
                "total_value": round(total_value, 2),
                "avg_risk": round(avg_risk, 4),
                "path_score": path_score,
                "direction": "incoming",
                "roles": [account_roles.get(a, {}).get("role", "unknown") for a in path[::-1]],
            })
        if len(path) >= max_path_length:
            continue
        for edge in rev_adj.get(current, []):
            prev = edge["from"]
            if prev not in path:
                stack.append((prev, path + [prev], edges_in_path + [edge], total_value + float(edge.get("amount", 0))))

    paths.sort(key=lambda p: -p["path_score"])
    return paths[:top_k]


# ---------------------------------------------------------------------------
# 4. Timeline reconstruction
# ---------------------------------------------------------------------------

def build_timeline(
    account_id: str,
    graph_edges: list[dict],
    account_risk_scores: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Build ordered timeline of transactions for an account.
    Uses timestamp field from edges if available, otherwise orders by index.
    """
    events = []
    for i, e in enumerate(graph_edges):
        if e["from"] == account_id or e["to"] == account_id:
            direction = "outgoing" if e["from"] == account_id else "incoming"
            counterparty = e["to"] if direction == "outgoing" else e["from"]
            events.append({
                "index": i,
                "timestamp": e.get("timestamp", ""),
                "direction": direction,
                "counterparty": counterparty,
                "amount": float(e.get("amount", 0)),
                "counterparty_risk": round(account_risk_scores.get(counterparty, 0.0), 4),
            })

    # Sort by timestamp if available, then by index
    events.sort(key=lambda ev: (ev.get("timestamp", ""), ev["index"]))

    # Add running balance
    balance = 0.0
    for ev in events:
        if ev["direction"] == "incoming":
            balance += ev["amount"]
        else:
            balance -= ev["amount"]
        ev["running_balance"] = round(balance, 2)
        del ev["index"]

    return events


# ---------------------------------------------------------------------------
# 5. Run all analysis  (called from pipeline)
# ---------------------------------------------------------------------------

def run_graph_analysis(
    graph_nodes: list[dict],
    graph_edges: list[dict],
    account_risk_scores: dict[str, float],
) -> dict[str, Any]:
    """Run all graph analysis and return combined results."""
    # Roles
    roles = classify_roles(graph_edges, account_risk_scores)

    # Communities
    community_result = detect_communities(graph_nodes, graph_edges, account_risk_scores)

    # Top suspicious flows
    flows = detect_flows(graph_edges, account_risk_scores, roles)

    return {
        "roles": roles,
        "clusters": community_result["clusters"],
        "account_cluster": community_result["account_cluster"],
        "top_flows": flows,
    }
