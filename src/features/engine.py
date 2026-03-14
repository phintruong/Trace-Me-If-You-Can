"""
Feature engineering for the AML risk scoring pipeline.

Computes all features per customer:
  F1-F14:  Money flow & behavioral features
  A1-A4:   Account age features
  C1-C5:   Cash behavior features (ABM-derived)
  W1-W5:   Wire transfer features
  Q1-Q4:   Quiet laundering features
  S1-S4:   Stealth laundering features
  N1-N5:   New rule features (inactive_days, exchange_outflow_ratio,
           raw_tx_count, declared_income, balance_cycles)

Log-transforms heavy-tailed features before output to prevent
downstream statistical saturation.

Pure computation — no ML, no rules, no file I/O.
"""

import logging
from datetime import timedelta
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    REFERENCE_DATE, OBSERVATION_WINDOW_DAYS, EPS, LOG_TRANSFORM_COLUMNS,
    BALANCE_CYCLE_FILL_THRESHOLD, BALANCE_CYCLE_DRAIN_THRESHOLD,
)
from src.pipeline.shutdown import check_shutdown
from src.jobs.taxonomy import build_occupation_lookup, classify_job_series

log = logging.getLogger("aml_pipeline")


def _compute_passthrough_gaps(inflows_sorted, outflows_sorted, customer_ids):
    """Compute median inflow-to-outflow gap per customer.

    Pre-groups into dicts of numpy arrays to avoid repeated
    DataFrame filtering (O(1) dict lookup vs O(n) boolean mask per customer).
    """
    in_groups = {}
    for cid, grp in inflows_sorted.groupby("customer_id"):
        in_groups[cid] = grp["transaction_datetime"].values

    out_groups = {}
    for cid, grp in outflows_sorted.groupby("customer_id"):
        out_groups[cid] = grp["transaction_datetime"].values

    gap_results = {}
    for cid in tqdm(customer_ids, desc="  Pass-through gaps", unit="cust", mininterval=1.0):
        cid_in = in_groups.get(cid)
        cid_out = out_groups.get(cid)

        if cid_in is None or cid_out is None or len(cid_in) == 0 or len(cid_out) == 0:
            gap_results[cid] = np.nan
            continue

        gaps = []
        out_idx = 0
        for in_time in cid_in:
            while out_idx < len(cid_out) and cid_out[out_idx] <= in_time:
                out_idx += 1
            if out_idx < len(cid_out):
                delta_hours = (cid_out[out_idx] - in_time) / np.timedelta64(1, "h")
                gaps.append(delta_hours)

        gap_results[cid] = np.median(gaps) if gaps else np.nan

    return gap_results


def _compute_per_customer_features(tx):
    """Compute round%, repeat_ratio, burst_score, offhours_ratio in a SINGLE groupby pass.

    Merges 4 separate groupby loops into one pass, reducing overhead by 4x.
    """
    round_pct = {}
    repeat_ratio = {}
    burst_scores = {}
    offhours = {}

    grouped = tx.groupby("customer_id")
    for cid, grp in tqdm(grouped, desc="  Per-customer features", unit="cust", mininterval=1.0):
        amounts = grp["amount_cad"].dropna().values
        n_amounts = len(amounts)

        # --- Round number % (F10) ---
        if n_amounts == 0:
            round_pct[cid] = 0.0
            repeat_ratio[cid] = 0.0
        else:
            n_round = np.sum(np.mod(amounts, 100) == 0)
            round_pct[cid] = n_round / n_amounts

            # --- Repeated amount ratio (F11) ---
            rounded_amounts = np.round(amounts, -1)
            counts = Counter(rounded_amounts)
            max_repeat_count = max(counts.values()) if counts else 0
            repeat_ratio[cid] = max_repeat_count / n_amounts

        # --- Burst score (F12) ---
        dates = grp["transaction_datetime"].dt.date
        daily_counts = dates.value_counts()
        if len(daily_counts) == 0:
            burst_scores[cid] = 0.0
        else:
            burst_scores[cid] = daily_counts.max() / (daily_counts.mean() + 0.01)

        # --- Off-hours ratio (F14) ---
        hours = grp["transaction_datetime"].dt.hour
        n_total = len(hours)
        if n_total == 0:
            offhours[cid] = 0.0
        else:
            n_off = ((hours >= 22) | (hours < 6)).sum()
            offhours[cid] = n_off / n_total

    return round_pct, repeat_ratio, burst_scores, offhours


def _compute_stealth_features(tx, features):
    """Compute stealth laundering features: channel loops, balance retention, merchant concentration.

    These catch slow, low-profile laundering that evades anomaly detection:
    S1: channel_loop_count — repeated tx_type sequences (e.g. EFT→EMT→Card repeated 3x)
    S2: balance_retention — abs(net_flow) / total_flow — low = pass-through drain
    S3: merchant_concentration — fraction of card spend at top merchant category
    """
    customer_ids = features.index

    # --- S1: Channel loop count ---
    # For each customer, extract their tx_type sequence sorted by time,
    # then count how many times the most common 3-gram repeats.
    tx_sorted = tx.sort_values(["customer_id", "transaction_datetime"])
    loop_counts = {}
    for cid, grp in tqdm(tx_sorted.groupby("customer_id"), desc="  Channel loops", unit="cust", mininterval=1.0):
        types = grp["tx_type"].values
        if len(types) < 6:
            loop_counts[cid] = 0
            continue
        # Build 3-grams of channel types
        trigrams = Counter()
        for i in range(len(types) - 2):
            trigram = (types[i], types[i + 1], types[i + 2])
            trigrams[trigram] += 1
        # Count = max repetitions of any single 3-gram pattern
        loop_counts[cid] = max(trigrams.values()) if trigrams else 0

    features["channel_loop_count"] = pd.Series(loop_counts).reindex(customer_ids).fillna(0).astype(int)

    # --- S2: Balance retention ratio ---
    # Low retention = money passes straight through (drain behavior).
    # Computed from raw (pre-log) inflow/outflow stored in features.
    # At this point total_inflow/total_outflow are still raw values.
    total_flow = features["total_inflow"] + features["total_outflow"]
    net_flow = (features["total_inflow"] - features["total_outflow"]).abs()
    features["balance_retention"] = net_flow / (total_flow + 1.0)

    # --- S3: Merchant concentration (proxy for counterparty reuse) ---
    # What fraction of card transactions go to the single most common merchant_category?
    card_tx = tx[tx["tx_type"] == "card"]
    if "merchant_category" in card_tx.columns and len(card_tx) > 0:
        card_with_cat = card_tx[card_tx["merchant_category"].str.len() > 0]
        cat_counts = card_with_cat.groupby("customer_id")["merchant_category"].agg(
            lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0
        )
        features["merchant_concentration"] = cat_counts.reindex(customer_ids).fillna(0)
    else:
        features["merchant_concentration"] = 0.0

    log.info("  S1-S3 (channel loops, balance retention, merchant concentration) complete")


def _compute_channel_switch(tx, features):
    """Compute channel switch ratio per customer.

    For each customer:
    1. Find their dominant inflow channel (tx_type with highest credit volume)
    2. Measure what fraction of outflow volume goes through OTHER channels
    High ratio = money enters via one channel, exits via different channels.
    """
    # Dominant inflow channel per customer
    inflow_by_channel = (
        tx[tx["debit_credit"] == "C"]
        .groupby(["customer_id", "tx_type"])["amount_cad"]
        .sum()
        .reset_index()
    )
    if len(inflow_by_channel) == 0:
        features["channel_switch_ratio"] = 0.0
        return

    idx_max = inflow_by_channel.groupby("customer_id")["amount_cad"].idxmax()
    main_channel = inflow_by_channel.loc[idx_max].set_index("customer_id")["tx_type"]

    # Outflow volume NOT through main inflow channel
    outflows = tx[tx["debit_credit"] == "D"].copy()
    outflows["main_channel"] = outflows["customer_id"].map(main_channel)
    outflows["is_switch"] = outflows["tx_type"] != outflows["main_channel"]

    switch_vol = outflows[outflows["is_switch"]].groupby("customer_id")["amount_cad"].sum()
    total_vol = features["total_inflow"] + features["total_outflow"]

    features["channel_switch_ratio"] = (
        switch_vol.reindex(features.index).fillna(0) / (total_vol + 1.0)
    )


def _compute_balance_cycles(tx, features):
    """Compute balance cycle count per customer via pseudo-balance reconstruction.

    Accounts that repeatedly fill up then drain to near-zero exhibit a
    "fill-and-flush" laundering pattern (FATF Typology). Legitimate accounts
    retain a balance base or show gradual accumulation.

    Algorithm: sort transactions by time, build running pseudo-balance,
    count transitions from above fill_threshold to below drain_threshold.
    Thresholds are fractions of the account's peak pseudo-balance.
    """
    tx_sorted = tx.sort_values(["customer_id", "transaction_datetime"])
    cycle_counts = {}

    for cid, grp in tqdm(tx_sorted.groupby("customer_id"),
                         desc="  Balance cycles", unit="cust", mininterval=1.0):
        amounts = grp["amount_cad"].values
        directions = grp["debit_credit"].values

        signed = np.where(directions == "C", amounts, -amounts)
        pseudo_bal = np.cumsum(signed)

        if len(pseudo_bal) < 4 or pseudo_bal.max() <= 0:
            cycle_counts[cid] = 0
            continue

        peak = pseudo_bal.max()
        fill_thresh = peak * BALANCE_CYCLE_FILL_THRESHOLD
        drain_thresh = peak * BALANCE_CYCLE_DRAIN_THRESHOLD

        cycles = 0
        state = "drain"
        for bal in pseudo_bal:
            if state == "drain" and bal >= fill_thresh:
                state = "fill"
            elif state == "fill" and bal <= drain_thresh:
                state = "drain"
                cycles += 1

        cycle_counts[cid] = cycles

    features["balance_cycles"] = (
        pd.Series(cycle_counts).reindex(features.index).fillna(0).astype(int)
    )


def compute_features(transactions_df, kyc_individual, kyc_smallbusiness):
    """Compute all 22 features for every customer.

    Returns a DataFrame with one row per customer_id and 22 feature columns.
    """
    # Filter to observation window
    window_start = REFERENCE_DATE - timedelta(days=OBSERVATION_WINDOW_DAYS)
    tx = transactions_df[
        (transactions_df["transaction_datetime"] >= window_start)
        & (transactions_df["transaction_datetime"] <= REFERENCE_DATE)
    ].copy()

    customer_ids = tx["customer_id"].unique()
    log.info(f"Computing features for {len(customer_ids):,} customers in {OBSERVATION_WINDOW_DAYS}-day window")

    # --- Pre-compute per-customer grouped data ---
    inflows = tx[tx["debit_credit"] == "C"]
    outflows = tx[tx["debit_credit"] == "D"]

    inflow_agg = inflows.groupby("customer_id")["amount_cad"].agg(["sum", "count"])
    inflow_agg.columns = ["total_inflow", "inflow_count"]

    outflow_agg = outflows.groupby("customer_id")["amount_cad"].agg(["sum", "count"])
    outflow_agg.columns = ["total_outflow", "outflow_count"]

    all_agg = tx.groupby("customer_id")["amount_cad"].agg(["count", "mean", "std"])
    all_agg.columns = ["tx_count", "avg_tx_size", "std_tx_size"]

    # Build feature DataFrame
    features = pd.DataFrame(index=customer_ids)
    features.index.name = "customer_id"

    # F1-F8: vectorized aggregations
    features["total_inflow"] = inflow_agg["total_inflow"].reindex(features.index).fillna(0)
    features["total_outflow"] = outflow_agg["total_outflow"].reindex(features.index).fillna(0)
    features["net_flow"] = features["total_inflow"] - features["total_outflow"]
    features["io_ratio"] = features["total_inflow"] / (features["total_outflow"] + EPS)
    features["raw_tx_count"] = all_agg["tx_count"].reindex(features.index).fillna(0).astype(int)
    features["tx_velocity"] = features["raw_tx_count"] / OBSERVATION_WINDOW_DAYS
    features["outflow_count"] = outflow_agg["outflow_count"].reindex(features.index).fillna(0).astype(int)
    features["avg_tx_size"] = all_agg["avg_tx_size"].reindex(features.index).fillna(0)
    features["std_tx_size"] = all_agg["std_tx_size"].reindex(features.index).fillna(0)
    features["cv_tx_size"] = features["std_tx_size"] / (features["avg_tx_size"] + EPS)
    log.info("  F1-F8 (vectorized aggregations) complete")

    check_shutdown("feature_engineering_f1_f8")

    # F9: Median inflow-to-outflow time gap (hours)
    inflows_sorted = inflows.sort_values(["customer_id", "transaction_datetime"])
    outflows_sorted = outflows.sort_values(["customer_id", "transaction_datetime"])
    gap_results = _compute_passthrough_gaps(inflows_sorted, outflows_sorted, customer_ids)
    features["median_in_out_gap_hours"] = pd.Series(gap_results)
    log.info("  F9 (pass-through gaps) complete")

    check_shutdown("feature_engineering_f9")

    # F10, F11, F12, F14: single-pass per-customer features
    round_pct, repeat_ratio, burst_scores, offhours = _compute_per_customer_features(tx)
    features["pct_round"] = pd.Series(round_pct)
    features["repeat_amt_ratio"] = pd.Series(repeat_ratio)
    features["burst_score"] = pd.Series(burst_scores)
    log.info("  F10-F12 (round/repeat/burst) complete")

    # F13: Max single transaction fraction
    max_tx = tx.groupby("customer_id")["amount_cad"].max()
    total_flow = features["total_inflow"] + features["total_outflow"]
    features["max_tx_fraction"] = max_tx.reindex(features.index).fillna(0) / (total_flow + EPS)

    # F14 result
    features["offhours_ratio"] = pd.Series(offhours)
    log.info("  F13-F14 (max fraction, offhours) complete")

    check_shutdown("feature_engineering_f10_f14")

    # --- AGE FEATURES ---
    log.info("  Computing age features...")

    # Vectorized age lookup
    kyc_ind = kyc_individual.set_index("customer_id")
    kyc_ind["account_age_days"] = (REFERENCE_DATE - kyc_ind["onboard_date"]).dt.days.clip(lower=1)
    kyc_ind["person_age_years"] = (REFERENCE_DATE - kyc_ind["birth_date"]).dt.days / 365.25

    kyc_biz = kyc_smallbusiness.set_index("customer_id")
    kyc_biz["account_age_days"] = (REFERENCE_DATE - kyc_biz["onboard_date"]).dt.days.clip(lower=1)
    kyc_biz["person_age_years"] = np.nan

    kyc_ind["account_type"] = "individual"
    kyc_biz["account_type"] = "business"

    age_df = pd.concat([
        kyc_ind[["account_age_days", "person_age_years", "account_type"]],
        kyc_biz[["account_age_days", "person_age_years", "account_type"]],
    ])

    features["account_age_days"] = age_df["account_age_days"].reindex(features.index)
    features["person_age_years"] = age_df["person_age_years"].reindex(features.index)
    features["account_type"] = age_df["account_type"].reindex(features.index).fillna("unknown")
    features["flow_per_account_day"] = total_flow / (features["account_age_days"].fillna(1) + EPS)

    # N4: Declared income from KYC — used for R43 (income mismatch detection)
    # Only available for individuals; businesses get 0 (excluded from R43 by design).
    if "income" in kyc_individual.columns:
        features["declared_income"] = kyc_ind["income"].reindex(features.index).fillna(0)
    else:
        features["declared_income"] = 0.0
    log.info("  N4 (declared_income) complete")

    # --- JOB CATEGORY ---
    log.info("  Classifying job categories from occupation codes...")
    occupation_lookup = build_occupation_lookup()
    occ_codes = kyc_ind["occupation_code"].reindex(features.index).fillna("UNKNOWN")
    features["job_category"] = classify_job_series(
        occ_codes, features["account_type"], occupation_lookup,
    )

    # A4: Flow acceleration
    quarter_cutoff = REFERENCE_DATE - timedelta(days=OBSERVATION_WINDOW_DAYS * 0.25)
    prior_cutoff = REFERENCE_DATE - timedelta(days=OBSERVATION_WINDOW_DAYS)

    recent_flow = tx[tx["transaction_datetime"] >= quarter_cutoff].groupby("customer_id")["amount_cad"].sum()
    prior_flow = tx[
        (tx["transaction_datetime"] >= prior_cutoff)
        & (tx["transaction_datetime"] < quarter_cutoff)
    ].groupby("customer_id")["amount_cad"].sum()

    prior_per_quarter = prior_flow / 3.0
    features["flow_acceleration"] = (
        recent_flow.reindex(features.index).fillna(0)
        / (prior_per_quarter.reindex(features.index).fillna(0) + EPS)
    )
    log.info("  A1-A4 (age features) complete")

    # N1: Inactive days — days since last transaction across ALL history.
    # Uses full transactions_df (pre-window-filter) to detect dormant accounts
    # that reactivated within the current observation window. R39 uses this.
    last_tx_dates = transactions_df.groupby("customer_id")["transaction_datetime"].max()
    inactive_days_raw = (REFERENCE_DATE - last_tx_dates).dt.days
    features["inactive_days"] = inactive_days_raw.reindex(features.index).fillna(0).clip(lower=0)
    log.info("  N1 (inactive_days) complete")

    check_shutdown("feature_engineering_age")

    # --- CASH FEATURES (C1-C5) ---
    # All ABM (ATM) transactions are cash. C = cash inflow, D = cash outflow.
    log.info("  Computing cash features...")
    cash_tx = tx[tx["tx_type"] == "abm"]
    cash_inflows = cash_tx[cash_tx["debit_credit"] == "C"]
    cash_outflows = cash_tx[cash_tx["debit_credit"] == "D"]

    cash_in_agg = cash_inflows.groupby("customer_id")["amount_cad"].sum()
    cash_out_agg = cash_outflows.groupby("customer_id")["amount_cad"].sum()

    features["cash_in_total"] = cash_in_agg.reindex(features.index).fillna(0)
    features["cash_out_total"] = cash_out_agg.reindex(features.index).fillna(0)
    features["cash_in_ratio"] = features["cash_in_total"] / (features["total_inflow"] + EPS)
    features["cash_out_ratio"] = features["cash_out_total"] / (features["total_outflow"] + EPS)
    total_volume = features["total_inflow"] + features["total_outflow"]
    features["cash_cycle_ratio"] = (
        np.minimum(features["cash_in_total"], features["cash_out_total"])
        / (total_volume + EPS)
    )
    log.info("  C1-C5 (cash features) complete")

    check_shutdown("feature_engineering_cash")

    # --- WIRE FEATURES (W1-W5) ---
    # Wire transactions are high-risk for layering and cross-border movement.
    log.info("  Computing wire features...")
    wire_tx = tx[tx["tx_type"] == "wire"]

    wire_count_agg = wire_tx.groupby("customer_id")["amount_cad"].count()
    wire_volume_agg = wire_tx.groupby("customer_id")["amount_cad"].sum()

    features["wire_tx_count"] = wire_count_agg.reindex(features.index).fillna(0).astype(int)
    features["wire_total_volume"] = wire_volume_agg.reindex(features.index).fillna(0)
    features["wire_velocity"] = features["wire_tx_count"] / OBSERVATION_WINDOW_DAYS
    features["wire_volume_ratio"] = features["wire_total_volume"] / (total_volume + EPS)

    # W5: Max wire transactions in any single week (clustering detection)
    if len(wire_tx) > 0:
        wire_weekly = wire_tx.copy()
        wire_weekly["iso_week"] = wire_weekly["transaction_datetime"].dt.isocalendar().week.astype(int)
        wire_weekly["iso_year"] = wire_weekly["transaction_datetime"].dt.isocalendar().year.astype(int)
        weekly_counts = wire_weekly.groupby(
            ["customer_id", "iso_year", "iso_week"]
        ).size().reset_index(name="wk_count")
        max_weekly = weekly_counts.groupby("customer_id")["wk_count"].max()
        features["wire_max_weekly_count"] = max_weekly.reindex(features.index).fillna(0).astype(int)
    else:
        features["wire_max_weekly_count"] = 0

    log.info("  W1-W5 (wire features) complete")

    # N2: Exchange outflow ratio — fraction of total outflow via wire + western union.
    # Cross-border transfer channels are high-risk for layering/smurfing.
    # Used by R42 (exchange funnel). Computed before log-transforms.
    wu_outflow = (
        tx[(tx["tx_type"] == "westernunion") & (tx["debit_credit"] == "D")]
        .groupby("customer_id")["amount_cad"].sum()
    )
    wire_outflow = (
        tx[(tx["tx_type"] == "wire") & (tx["debit_credit"] == "D")]
        .groupby("customer_id")["amount_cad"].sum()
    )
    exchange_out = (
        wu_outflow.reindex(features.index).fillna(0)
        + wire_outflow.reindex(features.index).fillna(0)
    )
    features["exchange_outflow_ratio"] = exchange_out / (features["total_outflow"] + EPS)
    log.info("  N2 (exchange_outflow_ratio) complete")

    check_shutdown("feature_engineering_wire")

    # --- QUIET LAUNDERING FEATURES (Q1-Q4) ---
    log.info("  Computing quiet laundering features...")

    # Q1: Channel switch ratio — outflow via different channel than primary inflow channel.
    # For each customer: find their dominant inflow channel, then measure
    # what fraction of outflow volume goes through OTHER channels.
    _compute_channel_switch(tx, features)

    # Q2: Retail ratio — card spend as fraction of total outflow.
    # Low retail + high flow = money moving without real economic activity.
    card_out = tx[(tx["tx_type"] == "card") & (tx["debit_credit"] == "D")]
    card_out_agg = card_out.groupby("customer_id")["amount_cad"].sum()
    features["retail_ratio"] = (
        card_out_agg.reindex(features.index).fillna(0)
        / (features["total_outflow"] + EPS)
    )

    # Q3: Distinct provinces — geographic spread across card + ABM transactions.
    geo_tx = tx[tx["province"].str.len() > 0]  # only rows with province data
    province_counts = geo_tx.groupby("customer_id")["province"].nunique()
    features["distinct_provinces"] = province_counts.reindex(features.index).fillna(0).astype(int)

    # Q4: Cash conversion — ATM deposits converted to card/electronic spending.
    # atm_in = ABM credits, card_out already computed above.
    features["atm_in_total"] = cash_in_agg.reindex(features.index).fillna(0)
    features["card_out_total"] = card_out_agg.reindex(features.index).fillna(0)

    log.info("  Q1-Q4 (quiet laundering features) complete")

    check_shutdown("feature_engineering_quiet")

    # --- STEALTH LAUNDERING FEATURES (S1-S4) ---
    log.info("  Computing stealth laundering features...")

    _compute_stealth_features(tx, features)

    # S4: Sustained passthrough count — number of inflow-outflow pairs
    # with similar amounts (within 20%) and gap < 72 hours.
    # Reuses the pre-sorted inflow/outflow data.
    passthrough_counts = {}
    in_groups_s = {}
    for cid, grp in inflows_sorted.groupby("customer_id"):
        in_groups_s[cid] = (grp["transaction_datetime"].values, grp["amount_cad"].values)
    out_groups_s = {}
    for cid, grp in outflows_sorted.groupby("customer_id"):
        out_groups_s[cid] = (grp["transaction_datetime"].values, grp["amount_cad"].values)

    for cid in tqdm(customer_ids, desc="  Sustained passthrough", unit="cust", mininterval=1.0):
        in_data = in_groups_s.get(cid)
        out_data = out_groups_s.get(cid)
        if in_data is None or out_data is None:
            passthrough_counts[cid] = 0
            continue
        in_times, in_amts = in_data
        out_times, out_amts = out_data
        count = 0
        out_idx = 0
        for j in range(len(in_times)):
            while out_idx < len(out_times) and out_times[out_idx] <= in_times[j]:
                out_idx += 1
            if out_idx < len(out_times):
                delta_h = (out_times[out_idx] - in_times[j]) / np.timedelta64(1, "h")
                if delta_h < 72 and in_amts[j] > 0:
                    ratio = out_amts[out_idx] / in_amts[j]
                    if 0.8 <= ratio <= 1.2:
                        count += 1
        passthrough_counts[cid] = count

    features["sustained_passthrough_count"] = pd.Series(passthrough_counts).reindex(features.index).fillna(0).astype(int)
    log.info("  S1-S4 (stealth laundering features) complete")

    check_shutdown("feature_engineering_stealth")

    # N5: Balance cycle count — fill-to-drain transitions in pseudo-balance
    log.info("  Computing balance cycle features...")
    _compute_balance_cycles(tx, features)
    log.info("  N5 (balance_cycles) complete")

    # Fill remaining NaNs (numeric only — preserve account_type string)
    numeric_cols = features.select_dtypes(include="number").columns
    features[numeric_cols] = features[numeric_cols].fillna(0)
    features = features.drop(columns=["tx_count"], errors="ignore")

    # --- LOG-TRANSFORM heavy-tailed features ---
    # Applied after all features are computed but before output.
    # log1p(x) = ln(1+x) compresses extreme values while preserving ordering.
    for col in LOG_TRANSFORM_COLUMNS:
        if col in features.columns:
            features[col] = np.log1p(features[col])
    log.info("  Log-transform applied to heavy-tailed features")

    return features.reset_index()
