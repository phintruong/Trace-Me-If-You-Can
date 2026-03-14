"""Run Railtracks/Gemini explanation agents (Pattern, Risk, Investigator) on flagged accounts."""

import os
from typing import Any


def _get_connections_for_account(
    account_id: str,
    graph_edges: list[dict],
    max_edges: int = 50,
) -> list[dict]:
    """Edges where account is sender or receiver."""
    out = []
    for e in graph_edges:
        if e["from"] == account_id or e["to"] == account_id:
            out.append({"from": e["from"], "to": e["to"], "amount": e.get("amount", 0)})
            if len(out) >= max_edges:
                break
    return out


def _call_llm(system: str, user: str, model_gemini: str = "gemini/gemini-2.0-flash") -> str:
    """Single completion via LiteLLM (Gemini) or OpenAI via Railtracks."""
    if os.environ.get("GEMINI_API_KEY"):
        import litellm
        r = litellm.completion(
            model=model_gemini,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()
    import railtracks as rt
    agent = rt.agent_node("Analyst", tool_nodes=(), llm=rt.llm.OpenAILLM("gpt-4o"), system_message=system)
    flow = rt.Flow(name="Explain", entry_point=agent)
    result = flow.invoke(user)
    return (getattr(result, "text", None) or str(result)).strip()


def run_railtracks_explainer(
    account_risk_scores: dict[str, float],
    account_patterns: dict[str, list[str]],
    graph_edges: list[dict],
    risk_threshold: float = 0.3,
    max_flagged: int = 50,
) -> list[dict[str, Any]]:
    """
    Run Pattern, Risk, and Investigator agents on flagged accounts.
    Returns list of flagged account dicts with pattern_agent_summary, risk_agent_summary,
    investigator_explanation, graph_connections.
    """
    flagged = sorted(
        [(acc, score) for acc, score in account_risk_scores.items() if score >= risk_threshold],
        key=lambda x: -x[1],
    )[:max_flagged]
    if not flagged:
        return []
    account_ids = [a for a, _ in flagged]
    pattern_list = account_patterns
    risk_scores = account_risk_scores
    lines = []
    for acc in account_ids:
        score = risk_scores.get(acc, 0)
        pats = pattern_list.get(acc, [])
        conns = _get_connections_for_account(acc, graph_edges, max_edges=20)
        lines.append(f"Account: {acc} | Risk: {score:.3f} | Patterns: {pats} | Edges sample: {len(conns)}")
    summary = "\n".join(lines)
    pattern_system = """You are an AML pattern analyst. Given a list of flagged accounts with risk scores and pattern labels (circular, hub, rapid_movement), summarize the laundering patterns observed. Be concise (2-4 sentences)."""
    pattern_out = _call_llm(pattern_system, f"Flagged accounts:\n{summary}")
    risk_system = """You are an AML risk analyst. Given the same flagged accounts with risk scores and patterns, comment on severity and transaction statistics. Be concise (2-4 sentences)."""
    risk_out = _call_llm(risk_system, f"Flagged accounts:\n{summary}")
    inv_system = """You are an AML investigator. Given (1) a pattern analyst summary and (2) a risk analyst summary about flagged accounts, write one short human-readable explanation (2-3 sentences) of why these accounts are suspicious, for a dashboard."""
    inv_user = f"Pattern analyst summary:\n{pattern_out}\n\nRisk analyst summary:\n{risk_out}\n\nWrite one short explanation for the dashboard."
    investigator_out = _call_llm(inv_system, inv_user)
    result = []
    for acc in account_ids:
        result.append({
            "account_id": acc,
            "risk_score": risk_scores.get(acc, 0),
            "detected_patterns": pattern_list.get(acc, []),
            "pattern_agent_summary": pattern_out,
            "risk_agent_summary": risk_out,
            "investigator_explanation": investigator_out,
            "graph_connections": _get_connections_for_account(acc, graph_edges),
        })
    return result
