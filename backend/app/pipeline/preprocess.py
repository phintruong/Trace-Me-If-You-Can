"""Preprocess raw transactions: validate, derive features, add transaction_id and account_id."""

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate schema, fill NA, normalize amounts, add derived features.
    Returns DataFrame with MODEL_FEATURE_COLUMNS plus transaction_id, account_id, timestamp, amount,
    amount_ratio, time_since_last_tx, device_change_flag, location_distance_km.
    """
    from src.features.engine import validate_ibm_schema, build_model_matrix
    from src.config import MODEL_FEATURE_COLUMNS

    validate_ibm_schema(df)
    df = df.copy()

    # Fill NA for key columns
    for col in ["Amount Received", "Amount Paid", "From Bank", "To Bank", "Payment Format"]:
        if col in df.columns:
            if "Amount" in col or "Bank" in col:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            elif col == "Payment Format":
                df[col] = df[col].fillna(0).astype(str)

    # Build model matrix (same logic as training) to get feature columns
    X, _ = build_model_matrix(df)
    out = X.copy()

    # Identifiers and API fields
    out["transaction_id"] = [f"tx_{i}" for i in range(len(df))]
    out["account_id"] = df["Account"].astype(str).values
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    out["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("").values
    amount = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0)
    out["amount"] = amount.values

    # Normalize amount (log1p) for derived features
    amount_log = np.log1p(amount.replace(0, np.nan).fillna(1))
    amount_recv = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0)
    amount_recv_log = np.log1p(amount_recv.replace(0, np.nan).fillna(1))

    # amount_ratio: paid vs received
    out["amount_ratio"] = (amount / (amount_recv + 1e-6)).values

    # time_since_last_tx per account (seconds)
    ts_sec = ts.astype("int64") // 10**9
    last_tx = df.groupby("Account")["Timestamp"].transform(
        lambda x: pd.to_datetime(x, errors="coerce").astype("int64").shift(1) // 10**9
    )
    out["time_since_last_tx"] = (ts_sec - last_tx.fillna(ts_sec)).fillna(0).values

    # device_change_flag: placeholder (Payment Format change vs previous for same account)
    out["device_change_flag"] = 0
    if "Payment Format" in df.columns:
        prev_fmt = df.groupby("Account")["Payment Format"].shift(1)
        out["device_change_flag"] = (df["Payment Format"] != prev_fmt).fillna(False).astype(int).values

    # location_distance_km: placeholder from From Bank / To Bank difference
    from_bank = pd.to_numeric(df["From Bank"], errors="coerce").fillna(0)
    to_bank = pd.to_numeric(df["To Bank"], errors="coerce").fillna(0)
    out["location_distance_km"] = np.abs(from_bank.values - to_bank.values) * 0.01

    return out
