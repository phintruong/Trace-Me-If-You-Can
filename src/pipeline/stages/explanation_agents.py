"""Stage 7: Send flagged results to 3 explanation agents (Pattern, Risk, Investigator)."""

import os

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.types import PipelineContext


def _get_connections_for_account(account_id: str, graph_edges: list[dict], max_edges: int = 50) -> list[dict]:
    """Edges where account is sender or receiver, for frontend graph."""
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


def stage_explanation_agents(
    ctx: PipelineContext,
    risk_threshold: float = 0.3,
    max_flagged: int = 50,
) -> PipelineContext:
    """
    Run Pattern, Risk, and Investigator agents on flagged accounts. Sets ctx.flagged_accounts.
    """
    if not ctx.account_risk_scores:
        raise ValueError("Stage 6 (risk_scores) must run before explanation_agents.")
    flagged = sorted(
        [(acc, score) for acc, score in ctx.account_risk_scores.items() if score >= risk_threshold],
        key=lambda x: -x[1],
    )[:max_flagged]
    if not flagged:
        ctx.flagged_accounts = []
        return ctx
    account_ids = [a for a, _ in flagged]
    pattern_list = ctx.account_patterns
    risk_scores = ctx.account_risk_scores
    edges = ctx.graph_edges

    # Build a text summary of flagged accounts for agents
    lines = []
    for acc in account_ids:
        score = risk_scores.get(acc, 0)
        pats = pattern_list.get(acc, [])
        conns = _get_connections_for_account(acc, edges, max_edges=20)
        lines.append(f"Account: {acc} | Risk: {score:.3f} | Patterns: {pats} | Edges sample: {len(conns)}")
    summary = "\n".join(lines)

    # 1. Pattern Agent
    pattern_system = """You are an AML pattern analyst. Given a list of flagged accounts with risk scores and pattern labels (circular, hub, rapid_movement), summarize the laundering patterns observed: circular transfers, hub accounts, rapid movement of funds. Be concise (2-4 sentences)."""
    pattern_out = _call_llm(pattern_system, f"Flagged accounts:\n{summary}")

    # 2. Risk Agent
    risk_system = """You are an AML risk analyst. Given the same flagged accounts with risk scores and patterns, comment on severity and transaction statistics. Be concise (2-4 sentences)."""
    risk_out = _call_llm(risk_system, f"Flagged accounts:\n{summary}")

    # 3. Investigator Agent (per-account or one summary)
    inv_system = """You are an AML investigator. Given (1) a pattern analyst summary and (2) a risk analyst summary about flagged accounts, write one short human-readable explanation (2-3 sentences) of why these accounts are suspicious, for a dashboard."""
    inv_user = f"Pattern analyst summary:\n{pattern_out}\n\nRisk analyst summary:\n{risk_out}\n\nWrite one short explanation for the dashboard."
    investigator_out = _call_llm(inv_system, inv_user)

    # Build per-account results (same pattern/risk/investigator text for all in batch; graph_connections per account)
    ctx.flagged_accounts = []
    for acc in account_ids:
        ctx.flagged_accounts.append({
            "account_id": acc,
            "risk_score": risk_scores.get(acc, 0),
            "detected_patterns": pattern_list.get(acc, []),
            "pattern_agent_summary": pattern_out,
            "risk_agent_summary": risk_out,
            "investigator_explanation": investigator_out,
            "graph_connections": _get_connections_for_account(acc, edges),
        })
    return ctx
