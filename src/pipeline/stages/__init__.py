"""AML pipeline stages. Each stage takes a PipelineContext and returns it (possibly mutated)."""

from src.pipeline.stages.load_data import stage_load_data
from src.pipeline.stages.preprocess import stage_preprocess
from src.pipeline.stages.fraud_detection import stage_fraud_detection
from src.pipeline.stages.build_graph import stage_build_graph
from src.pipeline.stages.detect_patterns import stage_detect_patterns
from src.pipeline.stages.risk_scores import stage_risk_scores
from src.pipeline.stages.explanation_agents import stage_explanation_agents
from src.pipeline.stages.output import stage_output

__all__ = [
    "stage_load_data",
    "stage_preprocess",
    "stage_fraud_detection",
    "stage_build_graph",
    "stage_detect_patterns",
    "stage_risk_scores",
    "stage_explanation_agents",
    "stage_output",
]
