"""GET /account/{account_id} — account flag and AI explanation."""

import json
import logging
from typing import List, Tuple

from fastapi import APIRouter, HTTPException

from app.config import RISK_THRESHOLD, WATSONX_MODEL_ID
from app.schemas import AccountResult
from app.services import db_client
from app.services import watsonx_client

router = APIRouter()
logger = logging.getLogger(__name__)

# Flag thresholds: LAUNDERING >= 0.9, SUSPICIOUS >= RISK_THRESHOLD (default 0.7), else NORMAL
LAUNDERING_THRESHOLD = 0.9


def _top_features_from_row(row: dict) -> List[Tuple[str, float]]:
    """Parse top_features from DB row (list of [name, value] or dict)."""
    raw = row.get("top_features") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        out = []
        for x in raw:
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                out.append((str(x[0]), float(x[1])))
            elif isinstance(x, dict):
                name = str(x.get("name") or x.get("feature") or "")
                val = float(x.get("value") or x.get("importance") or 0)
                out.append((name, val))
        return out
    return []


def _flag_from_max_risk(max_risk: float) -> str:
    if max_risk >= LAUNDERING_THRESHOLD:
        return "LAUNDERING"
    if max_risk >= RISK_THRESHOLD:
        return "SUSPICIOUS"
    return "NORMAL"


@router.get("/account/{account_id}", response_model=AccountResult)
def account(account_id: str, force: bool = False):
    """
    Return account_id (int), flag (NORMAL | SUSPICIOUS | LAUNDERING), and aiExplanation.
    Uses highest-risk transaction for the account to derive flag and explanation.
    """
    if not account_id or not str(account_id).strip():
        raise HTTPException(status_code=400, detail="account_id required")
    account_id_str = str(account_id).strip()

    rows = db_client.get_account_transactions(account_id=account_id_str, limit=1000)
    if not rows:
        raise HTTPException(status_code=404, detail="account not found")

    max_risk = max(float(r.get("risk_score", 0)) for r in rows)
    flag = _flag_from_max_risk(max_risk)

    # Use highest-risk transaction for explanation
    top_row = db_client.get_account_highest_risk_row(account_id_str)
    if not top_row:
        ai_explanation = f"Account has {len(rows)} transaction(s). Max risk score: {max_risk:.2f}. Flag: {flag}."
    else:
        tx_id = top_row.get("transaction_id", "")
        if not force:
            cached = db_client.get_explain_cache(tx_id)
            if cached:
                ai_explanation = cached["summary_text"]
            else:
                ai_explanation = _generate_explanation(top_row)
                db_client.set_explain_cache(tx_id, ai_explanation, WATSONX_MODEL_ID)
        else:
            ai_explanation = _generate_explanation(top_row)
            db_client.set_explain_cache(tx_id, ai_explanation, WATSONX_MODEL_ID)

    try:
        account_id_int = int(account_id_str)
    except ValueError:
        account_id_int = hash(account_id_str) % (2**31)

    return AccountResult(
        account_id=account_id_int,
        flag=flag,
        aiExplanation=ai_explanation,
    )


def _generate_explanation(transaction_row: dict) -> str:
    """Generate AI explanation for the transaction (used as account explanation)."""
    try:
        top_f = _top_features_from_row(transaction_row)
        return watsonx_client.generate_summary(
            transaction_row=transaction_row,
            top_features=top_f or None,
        )
    except Exception as e:
        logger.warning("watsonx explanation failed, using fallback: %s", e)
        risk = transaction_row.get("risk_score", 0)
        return f"Risk score {risk:.2f}. Automated explanation unavailable."
