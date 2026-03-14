"""
Run the full AML pipeline: load -> preprocess -> detect -> graph -> patterns -> risk -> agents -> output.
Swap models or add stages by editing the pipeline list.
"""

from src.pipeline.types import PipelineContext
from src.pipeline.stages import (
    stage_load_data,
    stage_preprocess,
    stage_fraud_detection,
    stage_build_graph,
    stage_detect_patterns,
    stage_risk_scores,
    stage_explanation_agents,
    stage_output,
)


def run_pipeline(
    file_name: str | None = None,
    risk_threshold: float = 0.3,
    max_flagged: int = 50,
) -> PipelineContext:
    """
    Execute all 8 stages in order. Returns context with ctx.api_output for the dashboard.
    """
    ctx = PipelineContext()
    ctx = stage_load_data(ctx, file_name=file_name)
    ctx = stage_preprocess(ctx)
    ctx = stage_fraud_detection(ctx)
    ctx = stage_build_graph(ctx)
    ctx = stage_detect_patterns(ctx)
    ctx = stage_risk_scores(ctx)
    ctx = stage_explanation_agents(ctx, risk_threshold=risk_threshold, max_flagged=max_flagged)
    ctx = stage_output(ctx)
    return ctx


if __name__ == "__main__":
    import json
    from src.config import OUTPUT_DIR
    ctx = run_pipeline(max_flagged=20)
    print("Flagged count:", len(ctx.flagged_accounts))
    if ctx.api_output:
        print("API output keys:", list(ctx.api_output.keys()))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "aml_pipeline_output.json"
        with open(path, "w") as f:
            json.dump(ctx.api_output, f, indent=2, default=str)
        print("Written:", path)
