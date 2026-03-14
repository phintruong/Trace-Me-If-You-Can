"""
Main pipeline orchestrator.

Executes the 6-stage AML risk scoring pipeline:
  1. Data Loading
  2. Feature Engineering
  3. Agent Scoring
  4. Rule-Based Scoring
  5. Risk Aggregation (agent + rules + context)
  6. Output Generation

Adding a new agent requires only:
  1. Create src/agents/my_new_agent.py subclassing BaseAgent
  2. Add it to the `agents` list in run_pipeline() below
"""

import sys
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    OUTPUT_DIR, OBSERVATION_WINDOW_DAYS, REFERENCE_DATE,
    FEATURE_COLUMNS, AGENT_WEIGHTS, DATA_RAW,
    SCORE_WEIGHT_AGENT, SCORE_WEIGHT_RULES, SCORE_WEIGHT_CONTEXT,
    SCORE_WEIGHT_SUPERVISED,
)
from src.utils.logging import setup_logging
from src.pipeline.shutdown import (
    install_signal_handlers, check_shutdown, set_stage, complete_stage,
)
from src.pipeline.checkpoint import (
    save_checkpoint, load_checkpoint, has_checkpoint, clear_checkpoints,
)
from src.data.loaders import load_kyc_individual, load_kyc_smallbusiness, load_all_transactions, load_supervised_scores
from src.features.engine import compute_features
from src.agents.base import AgentResult
from src.agents.isolation_forest_agent import IsolationForestAgent
from src.agents.sparse_statistical_agent import SparseStatisticalAgent
from src.agents.lof_agent import LOFAgent
from src.rules.aml_rules import compute_rule_scores
from src.aggregation.scorer import aggregate_scores, assign_risk_bucket
from src.reporting.reports import generate_account_report

warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger("aml_pipeline")


def run_pipeline(target_customer_id=None, resume=False):
    """Execute the full AML risk scoring pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("=" * 60)
    log.info("  AML RISK SCORING PIPELINE")
    log.info("=" * 60)

    if resume:
        log.info("Resume mode: will load checkpoints where available.")
    else:
        clear_checkpoints()

    # ===== Step 1: Load data =====
    set_stage("[1/6] Data Loading")

    if resume and has_checkpoint("transactions"):
        transactions = load_checkpoint("transactions")
        kyc_individual = load_checkpoint("kyc_individual")
        kyc_smallbusiness = load_checkpoint("kyc_smallbusiness")
        log.info(f"Loaded {len(transactions):,} transactions from checkpoint")
    else:
        kyc_individual = load_kyc_individual()
        kyc_smallbusiness = load_kyc_smallbusiness()
        log.info(f"KYC Individual: {len(kyc_individual):,} records")
        log.info(f"KYC Small Business: {len(kyc_smallbusiness):,} records")

        transactions = load_all_transactions()

        save_checkpoint("transactions", transactions)
        save_checkpoint("kyc_individual", kyc_individual)
        save_checkpoint("kyc_smallbusiness", kyc_smallbusiness)

    complete_stage("data_loading")
    check_shutdown("after_data_loading")

    # ===== Step 2: Feature engineering =====
    set_stage("[2/6] Feature Engineering")

    if resume and has_checkpoint("features"):
        features_df = load_checkpoint("features")
        log.info(f"Loaded {len(features_df):,} feature rows from checkpoint")
    else:
        features_df = compute_features(transactions, kyc_individual, kyc_smallbusiness)
        save_checkpoint("features", features_df)
        log.info(f"Features computed for {len(features_df):,} customers")

    # Free transaction memory — no longer needed
    del transactions

    if target_customer_id:
        if target_customer_id not in features_df["customer_id"].values:
            log.error(f"Customer {target_customer_id} not found in transaction data.")
            log.error("They may have no transactions in the observation window.")
            return
        log.info(f"Target customer: {target_customer_id}")

    complete_stage("feature_engineering")
    check_shutdown("after_feature_engineering")

    # ===== Step 3: Agent scoring =====
    # ---------------------------------------------------------------
    # Agents score individual and business accounts SEPARATELY so that
    # "normal" is defined within each cohort (a business doing $50k/day
    # is normal; an individual doing the same is suspicious).
    # Results are stitched back into the full DataFrame by index.
    # ---------------------------------------------------------------
    set_stage("[3/6] Agent Scoring")

    agents = [IsolationForestAgent(), SparseStatisticalAgent(), LOFAgent()]

    if resume and has_checkpoint("agent_results"):
        cached = load_checkpoint("agent_results")
        agent_results = cached["agent_results"]
        log.info("Loaded agent results from checkpoint")
    else:
        ind_mask = features_df["account_type"] == "individual"
        biz_mask = features_df["account_type"] == "business"
        unk_mask = ~(ind_mask | biz_mask)

        features_ind = features_df[ind_mask].reset_index(drop=True)
        features_biz = features_df[biz_mask].reset_index(drop=True)
        n_ind, n_biz, n_unk = len(features_ind), len(features_biz), int(unk_mask.sum())

        log.info(f"Cohort split: {n_ind:,} individual, {n_biz:,} business, {n_unk:,} unknown")

        # Score each cohort separately, then stitch results back
        agent_results = []
        for agent in agents:
            log.info(f"Running agent: {agent.name} v{agent.version}")

            # --- Individual cohort ---
            log.info(f"  [{agent.name}] Scoring individual cohort ({n_ind:,})")
            result_ind = agent.score(features_ind)

            # --- Business cohort ---
            log.info(f"  [{agent.name}] Scoring business cohort ({n_biz:,})")
            result_biz = agent.score(features_biz)

            # --- Stitch back into full array by original index ---
            n_total = len(features_df)
            full_scores = np.zeros(n_total)
            n_feat_ind = result_ind.explanations.shape[1]
            n_feat_biz = result_biz.explanations.shape[1]
            n_feat = max(n_feat_ind, n_feat_biz)
            full_explanations = np.zeros((n_total, n_feat))

            full_scores[ind_mask.values] = result_ind.scores
            full_scores[biz_mask.values] = result_biz.scores
            full_explanations[ind_mask.values, :n_feat_ind] = result_ind.explanations
            full_explanations[biz_mask.values, :n_feat_biz] = result_biz.explanations
            # Unknown accounts stay at 0 (no score)

            combined_result = AgentResult(
                agent_name=agent.name,
                scores=full_scores,
                explanations=full_explanations,
                metadata={
                    "individual": result_ind.metadata,
                    "business": result_biz.metadata,
                },
            )
            agent_results.append(combined_result)

            log.info(f"  Score range (combined): [{full_scores.min():.4f}, {full_scores.max():.4f}]")
            log.info(f"  Score mean  (combined): {full_scores.mean():.4f}")
            log.info(f"    Individual mean: {result_ind.scores.mean():.4f}")
            log.info(f"    Business mean:   {result_biz.scores.mean():.4f}")

        save_checkpoint("agent_results", {"agent_results": agent_results})

    # Store per-agent scores in the DataFrame for output CSV
    for r in agent_results:
        features_df[f"score_{r.agent_name}"] = r.scores
    # Combined agent score (diversity-weighted) for the output CSV
    weights = np.array(AGENT_WEIGHTS[:len(agent_results)], dtype=float)
    weights = weights / weights.sum()
    agent_combined = np.average([r.scores for r in agent_results], axis=0, weights=weights)
    features_df["model_score"] = agent_combined

    complete_stage("agent_scoring")
    check_shutdown("after_agent_scoring")

    # ===== Step 4: Rule scoring =====
    set_stage("[4/6] Rule-Based Scoring")

    rule_scores, triggered_rules = compute_rule_scores(features_df)
    features_df["rule_score"] = rule_scores

    n_rule_triggered = sum(1 for t in triggered_rules if len(t) > 0)
    log.info(f"Accounts with at least one rule triggered: {n_rule_triggered:,}")
    log.info(f"Rule score range: [{rule_scores.min():.4f}, {rule_scores.max():.4f}]")

    complete_stage("rule_scoring")
    check_shutdown("after_rule_scoring")

    # ===== Step 4.5: Supervised Model =====
    set_stage("[4.5/6] Supervised Model")

    from src.supervised.adapter import run_supervised_in_pipeline

    if resume and has_checkpoint("supervised_scores"):
        supervised_df = load_checkpoint("supervised_scores")
        log.info("Loaded supervised scores from checkpoint")
    else:
        supervised_df = run_supervised_in_pipeline()
        if supervised_df is not None:
            save_checkpoint("supervised_scores", supervised_df)
        else:
            # Fallback: try loading from pre-generated CSV
            supervised_df = load_supervised_scores()
            if supervised_df is not None:
                log.info("Fell back to moeez_2.csv for supervised scores")

    complete_stage("supervised_model")
    check_shutdown("after_supervised_model")

    # ===== Step 5: Aggregation =====
    set_stage("[5/6] Risk Aggregation")

    # Merge supervised scores into features_df (customer_id is a column, not index)
    supervised_scores = None
    if supervised_df is not None:
        features_df = features_df.merge(
            supervised_df, on="customer_id", how="left",
        )
        features_df["supervised_risk_score"] = features_df["supervised_risk_score"].fillna(0.0)
        supervised_scores = features_df["supervised_risk_score"].values
        log.info(f"Supervised scores merged: {(supervised_scores > 0).sum():,} non-zero")
    else:
        log.warning("No supervised scores — using zeros (fallback to 3-component formula)")

    final_scores, context_scores = aggregate_scores(
        agent_results, rule_scores, features_df, triggered_rules=triggered_rules,
        supervised_scores=supervised_scores,
    )
    features_df["context_score"] = context_scores
    features_df["supervised_score"] = supervised_scores if supervised_scores is not None else 0.0
    features_df["final_score"] = final_scores

    features_df["risk_bucket"] = features_df["final_score"].apply(assign_risk_bucket)

    log.info(f"Scoring formula: {SCORE_WEIGHT_AGENT}*agent + {SCORE_WEIGHT_RULES}*rules + {SCORE_WEIGHT_CONTEXT}*context + {SCORE_WEIGHT_SUPERVISED}*supervised")

    bucket_counts = features_df["risk_bucket"].value_counts()
    log.info("Risk distribution (all):")
    for bucket in ["LOW", "MEDIUM", "HIGH"]:
        count = bucket_counts.get(bucket, 0)
        pct = 100 * count / len(features_df)
        log.info(f"  {bucket:10s}: {count:6d} ({pct:.1f}%)")

    for acct_type in ["individual", "business"]:
        subset = features_df[features_df["account_type"] == acct_type]
        if len(subset) == 0:
            continue
        bc = subset["risk_bucket"].value_counts()
        log.info(f"Risk distribution ({acct_type}):")
        for bucket in ["LOW", "MEDIUM", "HIGH"]:
            count = bc.get(bucket, 0)
            pct = 100 * count / len(subset)
            log.info(f"  {bucket:10s}: {count:6d} ({pct:.1f}%)")

    # Job category risk breakdown
    if "job_category" in features_df.columns:
        log.info("Risk distribution by job category:")
        for cat in ["CASH_HEAVY", "SALARIED", "BUSINESS", "STUDENT", "UNKNOWN"]:
            subset = features_df[features_df["job_category"] == cat]
            if len(subset) == 0:
                continue
            bc = subset["risk_bucket"].value_counts()
            high_n = bc.get("HIGH", 0)
            log.info(f"  {cat:12s}: {len(subset):6,} total, {high_n:4d} HIGH ({100*high_n/len(subset):.1f}%)")

    complete_stage("aggregation")

    # ===== Step 6: Output =====
    set_stage("[6/6] Output Generation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Percentile ranks
    percentile_ranks = {}
    for col in FEATURE_COLUMNS:
        if col in features_df.columns:
            percentile_ranks[col] = features_df[col].rank(pct=True) * 100

    # Save full scored dataset
    agent_score_cols = [f"score_{r.agent_name}" for r in agent_results]
    output_cols = (
        ["customer_id", "account_type", "job_category"] + FEATURE_COLUMNS
        + agent_score_cols
        + ["model_score", "rule_score", "context_score", "supervised_score", "final_score", "risk_bucket"]
    )
    # Only include columns that exist in the DataFrame
    output_cols = [c for c in output_cols if c in features_df.columns]
    output_df = features_df[output_cols].sort_values("final_score", ascending=False)
    output_path = OUTPUT_DIR / f"risk_scores_{timestamp}.csv"
    output_df.to_csv(output_path, index=False)
    log.info(f"Full scores saved to: {output_path}")

    # Save triggered rules as JSON
    rules_output = {}
    for i, (_, row) in enumerate(features_df.iterrows()):
        cid = row["customer_id"]
        if triggered_rules[i]:
            rules_output[cid] = triggered_rules[i]
    rules_path = OUTPUT_DIR / f"triggered_rules_{timestamp}.json"
    with open(rules_path, "w") as f:
        json.dump(rules_output, f, indent=2)
    log.info(f"Triggered rules saved to: {rules_path}")

    # Generate reports for flagged accounts (HIGH tier only) or target
    if target_customer_id:
        report_indices = features_df[features_df["customer_id"] == target_customer_id].index
    else:
        report_indices = features_df[features_df["risk_bucket"] == "HIGH"].index

    reports = []
    for idx in report_indices:
        row = features_df.loc[idx]
        cid = row["customer_id"]
        i = features_df.index.get_loc(idx)

        acct_percentiles = {
            col: percentile_ranks[col].iloc[i]
            for col in FEATURE_COLUMNS
            if col in percentile_ranks
        }

        report = generate_account_report(
            customer_id=cid,
            features_row=row,
            agent_results=agent_results,
            sample_index=i,
            rule_score=row["rule_score"],
            context_score=row["context_score"],
            final_score=row["final_score"],
            triggered_rules=triggered_rules[i],
            feature_percentiles=acct_percentiles,
        )
        reports.append(report)

    if reports:
        report_path = OUTPUT_DIR / f"risk_reports_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(reports))
        log.info(f"Risk reports saved to: {report_path}")
        log.info(f"Reports generated for {len(reports)} account(s)")

        n_print = min(10, len(reports))
        print(f"\n{'=' * 60}")
        print(f"  TOP {n_print} RISK REPORTS")
        print(f"{'=' * 60}")
        for report in reports[:n_print]:
            print(report)
            print()
    else:
        log.info("No accounts in HIGH risk tier.")

    complete_stage("output_generation")

    # --- Summary ---
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"Customers scored:       {len(features_df):,}")
    log.info(f"Observation window:     {OBSERVATION_WINDOW_DAYS} days ending {REFERENCE_DATE.date()}")
    log.info(f"Agents:                 {', '.join(r.agent_name for r in agent_results)}")
    log.info(f"Scoring formula:        {SCORE_WEIGHT_AGENT}*agent + {SCORE_WEIGHT_RULES}*rules + {SCORE_WEIGHT_CONTEXT}*context + {SCORE_WEIGHT_SUPERVISED}*supervised")
    log.info(f"HIGH risk accounts:     {len(report_indices)}")
    log.info(f"Output directory:       {OUTPUT_DIR}")

    # ===== Risk Distribution =====
    print(f"\n{'=' * 70}")
    print(f"  RISK DISTRIBUTION")
    print(f"{'=' * 70}")
    for bucket in ["HIGH", "MEDIUM", "LOW"]:
        count = bucket_counts.get(bucket, 0)
        pct = 100 * count / len(features_df)
        print(f"  {bucket:<10s} {count:>6,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<10s} {len(features_df):>6,}")
    print(f"{'=' * 70}")

    # ===== SAR Label Check =====
    labels_path = DATA_RAW / "labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        sar_ids = labels[labels["label"] == 1]["customer_id"].tolist()
        if sar_ids:
            sar_rows = features_df[features_df["customer_id"].isin(sar_ids)]
            print(f"\n{'=' * 70}")
            print(f"  SAR LABEL CHECK — {len(sar_ids)} labeled accounts")
            print(f"{'=' * 70}")
            if len(sar_rows) == 0:
                print("  WARNING: None of the SAR accounts found in scored data!")
            else:
                print(f"  {'CUSTOMER_ID':<22} {'SCORE':>7} {'BUCKET':<8} {'RULE':>6} {'AGENT':>7} {'CTX':>6} {'SUP':>6}")
                print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")
                for _, row in sar_rows.sort_values("final_score", ascending=False).iterrows():
                    sup = row.get("supervised_score", 0.0)
                    print(f"  {row['customer_id']:<22} {row['final_score']:>7.3f} {row['risk_bucket']:<8} {row['rule_score']:>6.3f} {row['model_score']:>7.3f} {row['context_score']:>6.3f} {sup:>6.3f}")

                # Summary
                in_high = len(sar_rows[sar_rows["risk_bucket"] == "HIGH"])
                in_med = len(sar_rows[sar_rows["risk_bucket"] == "MEDIUM"])
                in_low = len(sar_rows[sar_rows["risk_bucket"] == "LOW"])
                print(f"\n  Result: {in_high} HIGH / {in_med} MEDIUM / {in_low} LOW  (out of {len(sar_rows)} found)")
                if in_high == len(sar_ids):
                    print("  ALL SAR accounts captured in HIGH tier!")
                elif in_high + in_med == len(sar_ids):
                    print("  All SAR accounts in HIGH or MEDIUM.")
                else:
                    print(f"  WARNING: {in_low} SAR account(s) in LOW tier — may need threshold tuning.")
            print(f"{'=' * 70}")

    # ===== Labeled Data Breakdown =====
    if labels_path.exists():
        labels_all = pd.read_csv(labels_path, dtype={"customer_id": str})
        labeled_in_scored = features_df[features_df["customer_id"].isin(labels_all["customer_id"])]
        if len(labeled_in_scored) > 0:
            merged = labeled_in_scored.merge(labels_all, on="customer_id", how="left")
            print(f"\n{'=' * 70}")
            print(f"  LABELED DATA BREAKDOWN — {len(merged)} labeled accounts in scored data")
            print(f"{'=' * 70}")
            for lbl, lbl_name in [(1, "Label=1 (SAR/suspicious)"), (0, "Label=0 (clean)")]:
                subset = merged[merged["label"] == lbl]
                if len(subset) == 0:
                    continue
                h = len(subset[subset["risk_bucket"] == "HIGH"])
                m = len(subset[subset["risk_bucket"] == "MEDIUM"])
                lo = len(subset[subset["risk_bucket"] == "LOW"])
                total = len(subset)
                print(f"  {lbl_name} — {total} accounts:")
                print(f"    HIGH:   {h:>5}  ({100*h/total:>5.1f}%)")
                print(f"    MEDIUM: {m:>5}  ({100*m/total:>5.1f}%)")
                print(f"    LOW:    {lo:>5}  ({100*lo/total:>5.1f}%)")
            # Precision/recall summary for label=1
            sar_subset = merged[merged["label"] == 1]
            clean_subset = merged[merged["label"] == 0]
            if len(sar_subset) > 0:
                sar_high = len(sar_subset[sar_subset["risk_bucket"] == "HIGH"])
                sar_med_high = len(sar_subset[sar_subset["risk_bucket"].isin(["HIGH", "MEDIUM"])])
                clean_high = len(clean_subset[clean_subset["risk_bucket"] == "HIGH"])
                clean_med_high = len(clean_subset[clean_subset["risk_bucket"].isin(["HIGH", "MEDIUM"])])
                print(f"\n  Detection rates (label=1):")
                print(f"    HIGH capture:        {sar_high}/{len(sar_subset)} ({100*sar_high/len(sar_subset):.0f}%)")
                print(f"    HIGH+MEDIUM capture: {sar_med_high}/{len(sar_subset)} ({100*sar_med_high/len(sar_subset):.0f}%)")
                if len(clean_subset) > 0:
                    print(f"  False positive rates (label=0):")
                    print(f"    HIGH false positives:        {clean_high}/{len(clean_subset)} ({100*clean_high/len(clean_subset):.1f}%)")
                    print(f"    HIGH+MEDIUM false positives: {clean_med_high}/{len(clean_subset)} ({100*clean_med_high/len(clean_subset):.1f}%)")
            print(f"{'=' * 70}")

    # ===== Team Overlap Check =====
    high_ids = set(features_df[features_df["risk_bucket"] == "HIGH"]["customer_id"])
    med_high_ids = set(features_df[features_df["risk_bucket"].isin(["HIGH", "MEDIUM"])]["customer_id"])

    supervised_mod_list_path = DATA_RAW.parent / "moeez.csv"
    eshan_path = DATA_RAW.parent / "eshan.csv"

    if supervised_mod_list_path.exists() or eshan_path.exists():
        print(f"\n{'=' * 70}")
        print(f"  TEAM OVERLAP CHECK")
        print(f"{'=' * 70}")

        if supervised_mod_list_path.exists():
            supervised_mod_list_df = pd.read_csv(supervised_mod_list_path, header=None)
            supervised_mod_list_ids = set(supervised_mod_list_df.iloc[:, 0])
            m_high = len(supervised_mod_list_ids & high_ids)
            m_medhigh = len(supervised_mod_list_ids & med_high_ids)
            m_total = len(supervised_mod_list_ids)
            print(f"  Supervised_mod ({m_total} accounts):")
            print(f"    in HIGH:        {m_high:>5}  ({100*m_high/m_total:.1f}%)")
            print(f"    in HIGH+MEDIUM: {m_medhigh:>5}  ({100*m_medhigh/m_total:.1f}%)")
            print(f"    not flagged:    {m_total - m_medhigh:>5}  ({100*(m_total-m_medhigh)/m_total:.1f}%)")

        if eshan_path.exists():
            eshan_df = pd.read_csv(eshan_path)
            eshan_ids = set(eshan_df.iloc[:, 0])
            e_high = len(eshan_ids & high_ids)
            e_medhigh = len(eshan_ids & med_high_ids)
            e_total = len(eshan_ids)
            print(f"  Eshan ({e_total} accounts):")
            print(f"    in HIGH:        {e_high:>5}  ({100*e_high/e_total:.1f}%)")
            print(f"    in HIGH+MEDIUM: {e_medhigh:>5}  ({100*e_medhigh/e_total:.1f}%)")
            print(f"    not flagged:    {e_total - e_medhigh:>5}  ({100*(e_total-e_medhigh)/e_total:.1f}%)")

        print(f"{'=' * 70}\n")

    # Clear checkpoints on successful completion
    clear_checkpoints()


def main():
    """CLI entrypoint — parses args, sets up logging/signals, runs pipeline."""
    setup_logging()
    install_signal_handlers()

    target = None
    resume = False

    args = sys.argv[1:]
    if "--resume" in args:
        resume = True
        args.remove("--resume")
    if args:
        target = args[0]

    run_pipeline(target_customer_id=target, resume=resume)
