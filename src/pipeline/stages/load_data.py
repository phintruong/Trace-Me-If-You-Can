"""Stage 1: Load IBM AML dataset."""

from src.pipeline.types import PipelineContext
from src.data.ibm_loader import load_transactions


def stage_load_data(ctx: PipelineContext, file_name: str | None = None) -> PipelineContext:
    """Load transaction CSV into context. Sets ctx.raw_df and ctx.file_name."""
    ctx.file_name = file_name
    ctx.raw_df = load_transactions(file_name=file_name)
    return ctx
