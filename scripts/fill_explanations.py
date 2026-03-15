"""
Fill aiExplanation in nodes.csv using Railtracks (Gemini) + Watsonx (Granite).

Reads nodes.csv & edges.csv from frontend/public/node_data/,
calls Railtracks multi-agent pipeline for pattern/risk context,
then calls Watsonx for per-account investigator summaries,
and writes the updated nodes.csv back.

Usage:
    cd GenAI-Genesis
    python scripts/fill_explanations.py
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NODE_CSV = PROJECT_ROOT / "frontend" / "public" / "node_data" / "nodes.csv"
EDGE_CSV = PROJECT_ROOT / "frontend" / "public" / "node_data" / "edges.csv"

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM helpers (same fallback chain as railtracks_explainer.py)
# ---------------------------------------------------------------------------

def _call_gemini(system: str, user: str) -> str | None:
    """Call Gemini via LiteLLM. Returns None on failure."""
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        import litellm
        r = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("Gemini call failed: %s", e)
        return None


def _call_watsonx(prompt: str) -> str | None:
    """Call Watsonx Granite. Returns None on failure."""
    api_key = os.environ.get("WATSONX_APIKEY", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")
    model_id = os.environ.get("WATSONX_MODEL_ID", "granite-13b-instruct")
    base_url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com/ml/v1")
    if not api_key or not project_id:
        return None
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        creds = Credentials(url=base_url, api_key=api_key)
        client = APIClient(creds)
        client.set.default_project(project_id)
        model = ModelInference(model_id=model_id, api_client=client, params={"max_new_tokens": 200})
        response = model.generate(prompt)
        text = response.get("results", [{}])[0].get("generated_text", "") if isinstance(response, dict) else str(response)
        return (text or "").strip()[:1500]
    except ImportError:
        pass
    except Exception as e:
        log.warning("Watsonx SDK failed: %s", e)

    # Fallback: direct HTTP
    try:
        import requests

        # Get IAM token first
        token_resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        token_resp.raise_for_status()
        bearer = token_resp.json()["access_token"]

        url = base_url.rstrip("/") + "/text/generation?version=2023-05-29"
        r = requests.post(
            url,
            json={
                "input": prompt,
                "model_id": model_id,
                "project_id": project_id,
                "parameters": {"max_new_tokens": 200},
            },
            headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        text = (r.json().get("results", [{}])[0].get("generated_text", "") or "").strip()
        return text[:1500]
    except Exception as e:
        log.warning("Watsonx HTTP failed: %s", e)
        return None


def call_llm(system: str, user: str) -> str:
    """Try Gemini first, then Watsonx, then static fallback."""
    result = _call_gemini(system, user)
    if result:
        return result
    # Combine system+user into a single prompt for Watsonx
    result = _call_watsonx(f"{system}\n\n{user}")
    if result:
        return result
    return "Automated explanation unavailable."


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    if not NODE_CSV.exists() or not EDGE_CSV.exists():
        log.error("nodes.csv or edges.csv not found at %s", NODE_CSV.parent)
        sys.exit(1)

    df_nodes = pd.read_csv(NODE_CSV)
    df_edges = pd.read_csv(EDGE_CSV)
    log.info("Loaded %d nodes, %d edges", len(df_nodes), len(df_edges))

    # Split nodes by risk tier
    laundering = df_nodes[df_nodes["risk"] == "laundering"]
    suspicious = df_nodes[df_nodes["risk"] == "suspicious"]
    normal = df_nodes[df_nodes["risk"] == "normal"]
    log.info("Breakdown: %d laundering, %d suspicious, %d normal", len(laundering), len(suspicious), len(normal))

    # ── Step 1: Railtracks-style global context (Pattern + Risk agents) ──
    # Build a summary of the top flagged accounts for the pattern/risk agents
    top_flagged = df_nodes[df_nodes["risk"].isin(["laundering", "suspicious"])].head(30)
    lines = []
    for _, row in top_flagged.iterrows():
        acc = row["id"]
        n_edges = len(df_edges[(df_edges["source"] == acc) | (df_edges["target"] == acc)])
        total_amt = df_edges.loc[(df_edges["source"] == acc) | (df_edges["target"] == acc), "amount"].sum()
        lines.append(f"Account: {acc} | Risk: {row['risk']} | Score: {row['riskScore']:.3f} | Transactions: {row['txCount']} | Edges in subgraph: {n_edges} | Total amount: ${total_amt:,.2f}")
    summary_block = "\n".join(lines)

    log.info("Running Pattern Agent...")
    pattern_summary = call_llm(
        "You are an AML pattern analyst. Given a list of flagged accounts with risk scores and transaction data, identify the laundering patterns observed (e.g. circular flows, hub accounts, rapid movement, layering, structuring). Be concise (3-5 sentences).",
        f"Flagged accounts:\n{summary_block}",
    )
    log.info("Pattern Agent done.")

    log.info("Running Risk Agent...")
    risk_summary = call_llm(
        "You are an AML risk analyst. Given flagged accounts with risk scores, transaction counts, and amounts, comment on the severity, volume anomalies, and which accounts pose the highest concern. Be concise (3-5 sentences).",
        f"Flagged accounts:\n{summary_block}",
    )
    log.info("Risk Agent done.")

    # ── Step 2: Per-account Investigator explanation (Watsonx/Gemini) ──
    explanations = {}
    needs_explanation = df_nodes[df_nodes["risk"].isin(["laundering", "suspicious"])]
    total = len(needs_explanation)
    log.info("Generating per-account explanations for %d accounts...", total)

    for i, (_, row) in enumerate(needs_explanation.iterrows()):
        acc = str(row["id"])

        # Gather this account's edge context
        acc_edges = df_edges[(df_edges["source"] == acc) | (df_edges["target"] == acc)]
        n_edges = len(acc_edges)
        total_amt = acc_edges["amount"].sum()
        top_partners = pd.unique(acc_edges[["source", "target"]].to_numpy().ravel())
        top_partners = [p for p in top_partners if p != acc][:5]

        user_prompt = (
            f"Pattern analyst context:\n{pattern_summary}\n\n"
            f"Risk analyst context:\n{risk_summary}\n\n"
            f"Account: {acc}\n"
            f"- Risk label: {row['risk']}\n"
            f"- Risk score: {row['riskScore']:.4f}\n"
            f"- Transaction count: {row['txCount']}\n"
            f"- Edges in subgraph: {n_edges}\n"
            f"- Total amount: ${total_amt:,.2f}\n"
            f"- Top connected accounts: {top_partners}\n\n"
            f"Write 1-2 sentences explaining why this account was flagged, for a fraud investigation dashboard."
        )

        explanation = call_llm(
            "You are an AML investigator. Write a short, specific explanation of why this account was flagged for potential money laundering. Reference concrete data (score, volume, connections). No bullet points.",
            user_prompt,
        )
        explanations[acc] = explanation

        if (i + 1) % 10 == 0 or (i + 1) == total:
            log.info("  %d / %d accounts done", i + 1, total)

    # Normal accounts get a simple label
    for _, row in normal.iterrows():
        explanations[str(row["id"])] = "No anomalies detected. Account activity within normal parameters."

    # ── Step 3: Write back to CSV ──
    df_nodes["aiExplanation"] = df_nodes["id"].astype(str).map(explanations).fillna("No anomalies detected.")
    df_nodes.to_csv(NODE_CSV, index=False)
    log.info("Updated %s with AI explanations.", NODE_CSV)

    # Print a few samples
    print("\n--- Sample explanations ---")
    for _, row in df_nodes.head(5).iterrows():
        print(f"\n[{row['risk']}] {row['id']} (score={row['riskScore']:.4f}):")
        print(f"  {row['aiExplanation']}")


if __name__ == "__main__":
    main()
