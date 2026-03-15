"""
Populate aiExplanation in nodes.csv using:
  Stage 1: Railtracks 3-agent analysis (Gemini gemini-3-flash-preview) — batched 70 accounts
  Stage 2: Watsonx human-readable rewrite — per-account

Each flagged account gets a unique explanation referencing its specific data.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load .env BEFORE importing anything that reads env vars
load_dotenv(BACKEND_DIR / ".env")

sys.path.insert(0, str(BACKEND_DIR))

from app.config import WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL, WATSONX_MODEL_ID

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
NODES_CSV = DATA_DIR / "nodes.csv"
EDGES_CSV = DATA_DIR / "edges.csv"

BATCH_SIZE = 70

# ─── IAM token cache ───
_iam_token = ""
_iam_token_expiry = 0.0


def _get_iam_token() -> str:
    global _iam_token, _iam_token_expiry
    if _iam_token and time.time() < _iam_token_expiry - 60:
        return _iam_token
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_APIKEY},
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    _iam_token = body["access_token"]
    _iam_token_expiry = body.get("expiration", time.time() + 3600)
    return _iam_token


# ─── LLM helpers ───

def _call_gemini(system: str, user: str, max_retries: int = 3) -> str:
    """Call Gemini gemini-3-flash-preview via LiteLLM with rate-limit retry."""
    import litellm
    for attempt in range(max_retries):
        try:
            r = litellm.completion(
                model="gemini/gemini-3-flash-preview",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "RateLimitError" in err_str:
                wait = 35 * (attempt + 1)
                log.warning("Rate limited, waiting %ds (retry %d/%d)...", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                log.warning("Gemini call failed: %s", e)
                break
    return ""


def _call_watsonx(prompt: str) -> str:
    """Call Watsonx text generation. Falls back to Gemini if Watsonx fails."""
    if WATSONX_APIKEY and WATSONX_PROJECT_ID:
        try:
            token = _get_iam_token()
            base = (WATSONX_URL or "https://us-south.ml.cloud.ibm.com").rstrip("/")
            url = f"{base}/ml/v1/text/generation?version=2024-05-31"
            payload = {
                "input": prompt,
                "model_id": WATSONX_MODEL_ID or "ibm/granite-3-3-8b-instruct",
                "project_id": WATSONX_PROJECT_ID,
                "parameters": {"max_new_tokens": 250},
            }
            r = requests.post(
                url, json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            r.raise_for_status()
            text = (r.json().get("results", [{}])[0].get("generated_text", "") or "").strip()
            if text:
                return text[:500]
        except Exception as e:
            log.warning("Watsonx failed, falling back to Gemini: %s", e)

    # Fallback: Gemini
    return _call_gemini(
        system="You are a compliance dashboard writer. Rewrite the given analysis into a clear, human-readable 1-3 sentence explanation.",
        user=prompt,
    )


def _parse_json_response(raw: str) -> dict:
    """Parse a JSON response, handling markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned)


# ─── Edge helpers ───

def _build_edge_lookup(df_edges: pd.DataFrame) -> dict[str, list[dict]]:
    """Build account → list of edges."""
    lookup: dict[str, list[dict]] = {}
    for _, e in df_edges.iterrows():
        src, tgt = str(e["source"]), str(e["target"])
        edge = {"source": src, "target": tgt, "amount": float(e["amount"])}
        lookup.setdefault(src, []).append(edge)
        if src != tgt:
            lookup.setdefault(tgt, []).append(edge)
    return lookup


def _account_edge_summary(account_id: str, edges: list[dict]) -> str:
    """Summarize an account's edges for prompting."""
    if not edges:
        return "No transactions found."
    amounts = [e["amount"] for e in edges]
    counterparties = set()
    for e in edges:
        if e["source"] != account_id:
            counterparties.add(e["source"])
        if e["target"] != account_id:
            counterparties.add(e["target"])
    self_loops = sum(1 for e in edges if e["source"] == e["target"] == account_id)
    total_vol = sum(amounts)
    avg_amt = total_vol / len(amounts) if amounts else 0
    max_amt = max(amounts) if amounts else 0

    parts = [
        f"{len(edges)} transactions, {len(counterparties)} counterparties",
        f"total_volume=${total_vol:,.2f}, avg=${avg_amt:,.2f}, max=${max_amt:,.2f}",
    ]
    if self_loops:
        parts.append(f"{self_loops} self-loop transactions")

    # Top 5 edges by amount
    top = sorted(edges, key=lambda e: e["amount"], reverse=True)[:5]
    edge_lines = [f"  {e['source']}→{e['target']}: ${e['amount']:,.2f}" for e in top]
    parts.append("Top transactions:\n" + "\n".join(edge_lines))

    return "\n".join(parts)


# ─── Main ───

def main():
    df_nodes = pd.read_csv(NODES_CSV)
    df_edges = pd.read_csv(EDGES_CSV)
    log.info("Loaded %d nodes, %d edges", len(df_nodes), len(df_edges))

    # Normal accounts
    mask_normal = df_nodes["risk"] == "normal"
    df_nodes.loc[mask_normal, "aiExplanation"] = "No anomalies detected. Account activity within normal parameters."
    log.info("Set %d normal accounts", mask_normal.sum())

    # Flagged accounts
    flagged_mask = df_nodes["risk"].isin(["laundering", "suspicious"])
    flagged = df_nodes[flagged_mask].copy()
    log.info("Flagged accounts: %d", len(flagged))
    if len(flagged) == 0:
        df_nodes.to_csv(NODES_CSV, index=False)
        return

    # Resume: load existing good explanations
    existing = {}
    for _, r in df_nodes.iterrows():
        val = str(r.get("aiExplanation", ""))
        if val and val not in ("TOBEFILLED", "nan", "",
                               "Automated explanation unavailable. Review flagged accounts manually.",
                               "No anomalies detected. Account activity within normal parameters."):
            # Also skip the generic investigator blob from previous run
            if len(val) > 50 and val.count("hub-and-spoke") > 0:
                continue
            existing[str(r["id"])] = val
    log.info("Resuming with %d existing good explanations", len(existing))

    # Build edge lookup
    edge_lookup = _build_edge_lookup(df_edges)

    # ════════════════════════════════════════════════════
    # STAGE 1: RAILTRACKS — 3 Gemini agents, batched
    # ════════════════════════════════════════════════════

    flagged_ids = [str(fid) for fid in flagged["id"].tolist()]
    flagged_data = {str(r["id"]): r for _, r in flagged.iterrows()}

    # Filter to only accounts needing work
    todo_ids = [fid for fid in flagged_ids if fid not in existing]
    log.info("Accounts needing Railtracks: %d (skipping %d)", len(todo_ids), len(existing))

    investigator_results: dict[str, str] = {}

    for batch_start in range(0, len(todo_ids), BATCH_SIZE):
        batch_ids = todo_ids[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(todo_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        log.info("═══ Railtracks batch %d/%d (%d accounts) ═══", batch_num, total_batches, len(batch_ids))

        # Build per-account context with edges
        account_contexts = []
        for aid in batch_ids:
            row = flagged_data[aid]
            edges = edge_lookup.get(aid, [])
            pat = row.get("pattern", "")
            if pd.isna(pat):
                pat = "None"
            edge_summary = _account_edge_summary(aid, edges)
            account_contexts.append(
                f"ACCOUNT {aid}:\n"
                f"  risk_score={row['riskScore']:.4f}, risk_level={row['risk']}, "
                f"tx_count={row['txCount']}, patterns={pat}\n"
                f"  Edges:\n  {edge_summary}"
            )
        all_context = "\n\n".join(account_contexts)

        # Agent 1: Pattern
        log.info("  Pattern Agent...")
        pattern_raw = _call_gemini(
            system="""You are an AML pattern analyst. For EACH account below, analyze its transaction edges and identify specific laundering patterns (layering, structuring, circular flows, fan-out/fan-in, rapid movement, self-loops, high-value transfers).

Return a JSON object mapping account_id to a 1-2 sentence pattern analysis. Be specific to each account's data. No markdown fences.""",
            user=all_context,
        )
        pattern_map = {}
        try:
            pattern_map = _parse_json_response(pattern_raw)
            log.info("  Pattern Agent: parsed %d/%d", len(pattern_map), len(batch_ids))
        except Exception as e:
            log.warning("  Pattern Agent JSON parse failed: %s", e)

        time.sleep(2)

        # Agent 2: Risk
        log.info("  Risk Agent...")
        risk_raw = _call_gemini(
            system="""You are an AML risk analyst. For EACH account below, assess its risk severity based on its risk score, transaction volume, counterparties, and amounts.

Return a JSON object mapping account_id to a 1-2 sentence risk assessment. Reference specific numbers. No markdown fences.""",
            user=all_context,
        )
        risk_map = {}
        try:
            risk_map = _parse_json_response(risk_raw)
            log.info("  Risk Agent: parsed %d/%d", len(risk_map), len(batch_ids))
        except Exception as e:
            log.warning("  Risk Agent JSON parse failed: %s", e)

        time.sleep(2)

        # Agent 3: Investigator — synthesize pattern + risk
        log.info("  Investigator Agent...")
        inv_input_parts = []
        for aid in batch_ids:
            pat_text = pattern_map.get(aid, "No pattern data")
            risk_text = risk_map.get(aid, "No risk data")
            row = flagged_data[aid]
            edges = edge_lookup.get(aid, [])
            edge_summary = _account_edge_summary(aid, edges)
            inv_input_parts.append(
                f"ACCOUNT {aid} (risk={row['riskScore']:.4f}, txCount={row['txCount']}):\n"
                f"  Pattern analysis: {pat_text}\n"
                f"  Risk assessment: {risk_text}\n"
                f"  Edges: {edge_summary}"
            )

        inv_raw = _call_gemini(
            system="""You are an AML investigator. Given pattern and risk analysis for each account, write a technical summary of why each account is suspicious.

Return a JSON object mapping account_id to a 2-3 sentence investigator summary. Each must be unique and reference that account's specific data. No markdown fences.""",
            user="\n\n".join(inv_input_parts),
        )
        try:
            inv_map = _parse_json_response(inv_raw)
            investigator_results.update(inv_map)
            log.info("  Investigator Agent: parsed %d/%d", len(inv_map), len(batch_ids))
        except Exception as e:
            log.warning("  Investigator JSON parse failed: %s", e)
            # Fallback: combine pattern + risk as investigator output
            for aid in batch_ids:
                pat_text = pattern_map.get(aid, "")
                risk_text = risk_map.get(aid, "")
                if pat_text or risk_text:
                    investigator_results[aid] = f"{pat_text} {risk_text}".strip()

        time.sleep(2)

    log.info("Railtracks done. Got investigator summaries for %d accounts", len(investigator_results))

    # ════════════════════════════════════════════════════
    # STAGE 2: WATSONX — per-account human-readable rewrite
    # ════════════════════════════════════════════════════

    explanations = dict(existing)
    watsonx_done = 0
    watsonx_fail = 0

    for i, aid in enumerate(todo_ids):
        if aid in explanations:
            continue

        inv_summary = investigator_results.get(aid, "")
        if not inv_summary:
            continue

        row = flagged_data[aid]
        edges = edge_lookup.get(aid, [])
        edge_summary = _account_edge_summary(aid, edges)

        prompt = f"""You are a fraud investigator assistant writing for a compliance dashboard. Given the analysis below, write a clear, human-readable 1-3 sentence explanation of why this specific account was flagged. Reference the account's specific numbers (risk score, transaction count, amounts). Make it unique to this account.

Account: {aid}
Risk Score: {row['riskScore']:.4f} ({row['risk']})
Transaction Count: {row['txCount']}
Edge Data: {edge_summary}

Investigator Analysis: {inv_summary}

Write ONLY the 1-3 sentence explanation, nothing else."""

        try:
            explanation = _call_watsonx(prompt)
            if explanation:
                explanations[aid] = explanation
                watsonx_done += 1
        except Exception as e:
            log.warning("Watsonx failed for %s: %s", aid, e)
            watsonx_fail += 1

        # Checkpoint every 70 accounts
        if (i + 1) % 70 == 0:
            log.info("Checkpoint: %d/%d done, saving...", watsonx_done, len(todo_ids))
            _save_csv(df_nodes, explanations)

        time.sleep(1)  # pace Watsonx calls

    log.info("Watsonx done: %d succeeded, %d failed", watsonx_done, watsonx_fail)

    # ════════════════════════════════════════════════════
    # STAGE 3: Write final CSV
    # ════════════════════════════════════════════════════

    _save_csv(df_nodes, explanations)

    remaining = 0
    for _, r in df_nodes.iterrows():
        val = str(r.get("aiExplanation", ""))
        if r["risk"] in ("laundering", "suspicious") and val in ("TOBEFILLED", "nan", ""):
            remaining += 1
    log.info("Done! Remaining without explanation: %d", remaining)


def _save_csv(df_nodes: pd.DataFrame, explanations: dict[str, str]):
    """Write explanations to nodes.csv and node_data/nodes.csv."""
    for idx, row in df_nodes.iterrows():
        aid = str(row["id"])
        if aid in explanations:
            df_nodes.at[idx, "aiExplanation"] = explanations[aid]

    df_nodes.to_csv(NODES_CSV, index=False)
    log.info("Wrote %s", NODES_CSV)

    node_data_dir = PROJECT_ROOT / "frontend" / "public" / "node_data"
    if node_data_dir.exists():
        df_nodes.to_csv(node_data_dir / "nodes.csv", index=False)
        log.info("Also wrote %s", node_data_dir / "nodes.csv")


if __name__ == "__main__":
    main()
