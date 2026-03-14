"""Stage 8: Output structured results for the dashboard API."""

from src.pipeline.types import PipelineContext


def stage_output(ctx: PipelineContext) -> PipelineContext:
    """Build API-ready payload. Sets ctx.api_output."""
    ctx.api_output = {
        "flagged_accounts": [
            {
                "account_id": fa["account_id"],
                "risk_score": fa["risk_score"],
                "detected_patterns": fa["detected_patterns"],
                "pattern_agent_summary": fa["pattern_agent_summary"],
                "risk_agent_summary": fa["risk_agent_summary"],
                "investigator_explanation": fa["investigator_explanation"],
                "graph_connections": fa["graph_connections"],
            }
            for fa in ctx.flagged_accounts
        ],
        "graph": {
            "nodes": ctx.graph_nodes,
            "edges": ctx.graph_edges,
        },
        "meta": {
            "total_flagged": len(ctx.flagged_accounts),
            "total_nodes": len(ctx.graph_nodes),
            "total_edges": len(ctx.graph_edges),
        },
    }
    return ctx
