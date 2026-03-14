"""Stage 5: Detect suspicious patterns (circular, hub, rapid movement)."""

from collections import defaultdict

from src.pipeline.types import PipelineContext


def _find_circular_accounts(edges: list[dict]) -> set[str]:
    """Accounts that participate in a cycle (simplified: 2-hop return)."""
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
    """Accounts with highest in+out degree (top fraction)."""
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
    """Accounts with many transactions (rapid movement)."""
    tx_count: dict[str, int] = defaultdict(int)
    for e in edges:
        tx_count[e["from"]] += 1
        tx_count[e["to"]] += 1
    return {a for a, c in tx_count.items() if c >= min_tx}


def stage_detect_patterns(ctx: PipelineContext) -> PipelineContext:
    """Detect circular, hub, and rapid-movement patterns. Sets ctx.account_patterns."""
    if not ctx.graph_edges:
        raise ValueError("Stage 4 (build_graph) must run before detect_patterns.")
    circular = _find_circular_accounts(ctx.graph_edges)
    hubs = _find_hub_accounts(ctx.graph_edges)
    rapid = _find_rapid_movement_accounts(ctx.graph_edges)
    account_patterns: dict[str, list[str]] = defaultdict(list)
    for acc in set(ctx.account_to_id.keys()):
        if acc in circular:
            account_patterns[acc].append("circular")
        if acc in hubs:
            account_patterns[acc].append("hub")
        if acc in rapid:
            account_patterns[acc].append("rapid_movement")
    ctx.account_patterns = dict(account_patterns)
    return ctx
