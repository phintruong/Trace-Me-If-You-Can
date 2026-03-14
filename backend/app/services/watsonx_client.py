"""Watsonx.ai client for generating investigation summaries. No caching here; caller uses DB cache."""

import logging
from typing import List, Tuple

from app.config import WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL, WATSONX_MODEL_ID

logger = logging.getLogger(__name__)
WATSONX_TIMEOUT = 5


def _build_prompt(transaction_row: dict, top_features: List[Tuple[str, float]] = None) -> str:
    """Build the exact prompt from spec."""
    tx_id = transaction_row.get("transaction_id", "N/A")
    acc_id = transaction_row.get("account_id", "N/A")
    amount = transaction_row.get("amount", 0)
    timestamp = transaction_row.get("timestamp", "N/A")
    merchant = transaction_row.get("merchant", transaction_row.get("Account.1", "N/A"))
    location = transaction_row.get("location", str(transaction_row.get("From Bank", "N/A")))
    device = transaction_row.get("device", transaction_row.get("Payment Format", "N/A"))
    risk_score = transaction_row.get("risk_score", 0)

    if top_features:
        feature_list = "\n".join(f"{name}: {val:+.2f}" for name, val in top_features[:10])
    else:
        feature_list = "N/A"

    return f"""You are a fraud investigator assistant. Given the following transaction facts and model feature contributions, generate a 1-3 sentence investigative summary focusing on the most important reasons this transaction was flagged.

Transaction:
- transaction_id: {tx_id}
- account_id: {acc_id}
- amount: ${amount}
- timestamp: {timestamp}
- merchant: {merchant}
- location: {location}
- device: {device}
- risk_score: {risk_score}

Feature contributions (highest → lowest):
{feature_list}

Provide:
1) A short plain-language summary (1-3 sentences).
2) The top 3 reasons ranked.

Keep the response concise (truncate beyond 200 tokens)."""


def generate_summary(
    transaction_row: dict,
    top_features: List[Tuple[str, float]] = None,
) -> str:
    """
    Call watsonx.ai to generate 1-3 sentence investigator summary.
    Uses WATSONX_URL, WATSONX_APIKEY, WATSONX_PROJECT_ID. 5s timeout.
    Returns summary text or raises on error. Caller must cache in DB.
    """
    prompt = _build_prompt(transaction_row, top_features)

    try:
        # Prefer official SDK
        try:
            from ibm_watsonx_ai import APIClient
            from ibm_watsonx_ai.foundation_models import ModelInference

            client = APIClient(api_key=WATSONX_APIKEY)
            client.set.default_project(WATSONX_PROJECT_ID)
            model = ModelInference(model_id=WATSONX_MODEL_ID, params={"max_new_tokens": 200})
            response = model.generate(prompt, timeout=WATSONX_TIMEOUT)
            text = response.get("results", [{}])[0].get("generated_text", "") if isinstance(response, dict) else str(response)
            return (text or "")[:1500].strip()
        except ImportError:
            pass

        # Fallback: HTTP POST to generate endpoint
        import requests
        url = (WATSONX_URL or "https://us-south.ml.cloud.ibm.com/ml/v1").rstrip("/") + "/text/generation"
        headers = {
            "Authorization": f"Bearer {WATSONX_APIKEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": prompt,
            "model_id": WATSONX_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "parameters": {"max_new_tokens": 200},
        }
        r = requests.post(url, json=payload, headers=headers, timeout=WATSONX_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        text = (data.get("results", [{}])[0].get("generated_text", "") or "").strip()
        return text[:1500]
    except Exception as e:
        logger.exception("watsonx generate_summary failed: %s", e)
        raise RuntimeError(f"watsonx summary generation failed: {e}") from e
