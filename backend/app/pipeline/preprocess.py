"""Preprocess raw transactions: validate schema, build feature matrix, add transaction_id, account_id, etc."""

import numpy as np
import pandas as pd

from app.config import IBM_REQUIRED_COLUMNS, MODEL_FEATURE_COLUMNS


def validate_ibm_schema(df: pd.DataFrame) -> None:
    """Validate DataFrame has required IBM AML columns."""
    missing = [c for c in IBM_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build X (features) and y (Is Laundering) from raw transaction rows."""
    validate_ibm_schema(df)
    X = df[
        [
            "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format",
        ]
    ].copy()
    y = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).astype(int)
    ts = pd.to_datetime(X["Timestamp"], errors="coerce")
    X["Hour"] = ts.dt.hour.fillna(-1).astype("int16")
    X["DayOfWeek"] = ts.dt.dayofweek.fillna(-1).astype("int16")
    X["Day"] = ts.dt.day.fillna(-1).astype("int16")
    X["Month"] = ts.dt.month.fillna(-1).astype("int16")
    X = X.drop(columns=["Timestamp"])
    for col in ["From Bank", "To Bank", "Amount Received", "Amount Paid"]:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    for col in ["Account", "Account.1", "Receiving Currency", "Payment Currency", "Payment Format"]:
        X[col] = pd.factorize(X[col], sort=True)[0].astype("int32")
    X = X[MODEL_FEATURE_COLUMNS].copy()
    return X, y


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, build feature matrix, add transaction_id, account_id, timestamp, amount, and derived fields.
    Returns DataFrame with MODEL_FEATURE_COLUMNS plus API fields. Account/Account.1 are factorized.
    """
    validate_ibm_schema(df)
    df = df.copy()
    for col in ["Amount Received", "Amount Paid", "From Bank", "To Bank", "Payment Format"]:
        if col in df.columns:
            if "Amount" in col or "Bank" in col:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = df[col].fillna(0).astype(str)
    X, _ = build_model_matrix(df)
    out = X.copy()
    out["transaction_id"] = [f"tx_{i}" for i in range(len(df))]
    out["account_id"] = df["Account"].astype(str).values
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    out["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("").values
    amount = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0)
    out["amount"] = amount.values
    amount_log = np.log1p(amount.replace(0, np.nan).fillna(1))
    amount_recv = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0)
    out["amount_ratio"] = (amount / (amount_recv + 1e-6)).values
    ts_sec = ts.astype("int64") // 10**9
    last_tx = df.groupby("Account")["Timestamp"].transform(
        lambda x: pd.to_datetime(x, errors="coerce").astype("int64").shift(1) // 10**9
    )
    out["time_since_last_tx"] = (ts_sec - last_tx.fillna(ts_sec)).fillna(0).values
    out["device_change_flag"] = 0
    if "Payment Format" in df.columns:
        prev_fmt = df.groupby("Account")["Payment Format"].shift(1)
        out["device_change_flag"] = (df["Payment Format"] != prev_fmt).fillna(False).astype(int).values
    from_bank = pd.to_numeric(df["From Bank"], errors="coerce").fillna(0)
    to_bank = pd.to_numeric(df["To Bank"], errors="coerce").fillna(0)
    out["location_distance_km"] = np.abs(from_bank.values - to_bank.values) * 0.01
    return out
