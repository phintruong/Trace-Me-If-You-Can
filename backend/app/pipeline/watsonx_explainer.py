"""Thin wrapper around Watsonx for per-transaction/account AI explanations."""

from app.services import watsonx_client


def generate_explanation(transaction_row: dict, top_features: list[tuple[str, float]] | None = None) -> str:
    """Generate 1-3 sentence investigator summary via Watsonx. Caller caches in DB."""
    return watsonx_client.generate_summary(
        transaction_row=transaction_row,
        top_features=top_features or None,
    )
