"""Database client: SQLite (default) or IBM Db2. Predictions and explanation cache."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import DB_CONN_STRING, DB_MODE, DEFAULT_SQLITE_PATH

# Optional DB2
try:
    import ibm_db
    import ibm_db_dbi
    _HAS_DB2 = True
except ImportError:
    _HAS_DB2 = False


def _get_connection():
    if DB_MODE == "db2" and DB_CONN_STRING and _HAS_DB2:
        return ibm_db_dbi.connect(DB_CONN_STRING)
    # SQLite
    path = DB_CONN_STRING or str(DEFAULT_SQLITE_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


@contextmanager
def _cursor():
    conn = _get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        cur.close()
        conn.close()


def _init_sqlite(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            transaction_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            timestamp TEXT,
            amount REAL,
            risk_score REAL,
            top_features TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS explanations (
            transaction_id TEXT PRIMARY KEY,
            summary_text TEXT NOT NULL,
            model_version TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    cur.close()


def _init_db2(conn):
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE predictions (
                transaction_id VARCHAR(128) NOT NULL PRIMARY KEY,
                account_id VARCHAR(128) NOT NULL,
                timestamp VARCHAR(64),
                amount DOUBLE,
                risk_score DOUBLE,
                top_features CLOB,
                created_at VARCHAR(64)
            )
        """)
    except Exception:
        pass
    try:
        cur.execute("""
            CREATE TABLE explanations (
                transaction_id VARCHAR(128) NOT NULL PRIMARY KEY,
                summary_text CLOB NOT NULL,
                model_version VARCHAR(128) NOT NULL,
                created_at VARCHAR(64) NOT NULL
            )
        """)
    except Exception:
        pass
    conn.commit()
    cur.close()


def init_db():
    """Create tables if they do not exist."""
    conn = _get_connection()
    try:
        if DB_MODE == "db2" and _HAS_DB2:
            _init_db2(conn)
        else:
            _init_sqlite(conn)
    finally:
        conn.close()


def save_predictions(df) -> None:
    """Persist predictions DataFrame to DB. Expects columns: transaction_id, account_id, timestamp, amount, risk_score, top_features (optional)."""
    import pandas as pd
    init_db()
    conn = _get_connection()
    cur = conn.cursor()
    try:
        created = pd.Timestamp.now().isoformat()
        for _, row in df.iterrows():
            tx_id = str(row.get("transaction_id", ""))
            acc_id = str(row.get("account_id", ""))
            ts = str(row.get("timestamp", "")) if pd.notna(row.get("timestamp")) else ""
            amount = float(row.get("amount", 0))
            risk = float(row.get("risk_score", 0))
            top_f = row.get("top_features")
            if hasattr(top_f, "__iter__") and not isinstance(top_f, str):
                top_f = json.dumps(list(top_f) if not isinstance(top_f, dict) else top_f)
            else:
                top_f = json.dumps([]) if top_f is None else str(top_f)
            if DB_MODE == "db2" and _HAS_DB2:
                try:
                    cur.execute("DELETE FROM predictions WHERE transaction_id = ?", (tx_id,))
                except Exception:
                    pass
                cur.execute(
                    "INSERT INTO predictions (transaction_id, account_id, timestamp, amount, risk_score, top_features, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (tx_id, acc_id, ts, amount, risk, top_f, created),
                )
            else:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO predictions (transaction_id, account_id, timestamp, amount, risk_score, top_features, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tx_id, acc_id, ts, amount, risk, top_f, created),
                )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def get_alerts(threshold: float, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
    """Return list of high-risk transactions (risk_score >= threshold), ordered by risk desc, capped by limit."""
    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.transaction_id, p.account_id, p.timestamp, p.amount, p.risk_score, p.top_features,
                   e.summary_text
            FROM predictions p
            LEFT JOIN explanations e ON p.transaction_id = e.transaction_id
            WHERE p.risk_score >= ?
            ORDER BY p.risk_score DESC, p.timestamp DESC
            LIMIT ?
            """,
            (threshold, limit),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            result.append({
                "transaction_id": d.get("transaction_id", ""),
                "account_id": d.get("account_id", ""),
                "timestamp": d.get("timestamp", ""),
                "amount": float(d.get("amount", 0)),
                "risk_score": float(d.get("risk_score", 0)),
                "summary": d.get("summary_text"),
                "explain_cached": d.get("summary_text") is not None and len(str(d.get("summary_text", "")).strip()) > 0,
            })
        return result
    except Exception as e:
        raise RuntimeError(f"alerts fetch failed: {e}") from e
    finally:
        conn.close()


def get_flagged_accounts(
    threshold: float, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Return one row per account where the account's max risk_score >= threshold.
    Each row: account_id, risk_score (max), transaction_id (top-risk tx), summary_text,
    transaction_count, total_amount, last_transaction_date.
    Ordered by risk_score descending.
    """
    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        # One row per account: the transaction with max risk for that account.
        # Join explanations for summary_text and aggregate subquery for counts.
        if DB_MODE == "db2" and _HAS_DB2:
            sql = """
            SELECT t.transaction_id, t.account_id, t.timestamp, t.amount, t.risk_score,
                   e.summary_text, a.transaction_count, a.total_amount, a.last_transaction_date
            FROM (
                SELECT p.transaction_id, p.account_id, p.timestamp, p.amount, p.risk_score,
                       ROW_NUMBER() OVER (PARTITION BY p.account_id ORDER BY p.risk_score DESC) AS rn
                FROM predictions p
                WHERE p.risk_score >= ?
            ) t
            LEFT JOIN explanations e ON t.transaction_id = e.transaction_id
            LEFT JOIN (
                SELECT account_id, COUNT(*) AS transaction_count,
                       COALESCE(SUM(amount), 0) AS total_amount,
                       MAX(timestamp) AS last_transaction_date
                FROM predictions GROUP BY account_id
            ) a ON t.account_id = a.account_id
            WHERE t.rn = 1
            ORDER BY t.risk_score DESC
            """
        else:
            sql = """
            WITH ranked AS (
                SELECT p.transaction_id, p.account_id, p.timestamp, p.amount, p.risk_score,
                       ROW_NUMBER() OVER (PARTITION BY p.account_id ORDER BY p.risk_score DESC) AS rn
                FROM predictions p
                WHERE p.risk_score >= ?
            ),
            aggs AS (
                SELECT account_id, COUNT(*) AS transaction_count,
                       COALESCE(SUM(amount), 0) AS total_amount,
                       MAX(timestamp) AS last_transaction_date
                FROM predictions GROUP BY account_id
            )
            SELECT t.transaction_id, t.account_id, t.timestamp, t.amount, t.risk_score,
                   e.summary_text, a.transaction_count, a.total_amount, a.last_transaction_date
            FROM ranked t
            LEFT JOIN explanations e ON t.transaction_id = e.transaction_id
            LEFT JOIN aggs a ON t.account_id = a.account_id
            WHERE t.rn = 1
            ORDER BY t.risk_score DESC
            """
        params: List[Any] = [threshold]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            tc = d.get("transaction_count")
            ta = d.get("total_amount")
            ltd = d.get("last_transaction_date")
            result.append({
                "account_id": str(d.get("account_id", "")),
                "risk_score": float(d.get("risk_score", 0)),
                "transaction_id": str(d.get("transaction_id", "")),
                "timestamp": str(d.get("timestamp") or ""),
                "amount": float(d.get("amount", 0)),
                "summary_text": d.get("summary_text"),
                "transaction_count": int(tc) if tc is not None else None,
                "total_amount": float(ta) if ta is not None else None,
                "last_transaction_date": str(ltd) if ltd is not None else None,
            })
        return result
    except Exception as e:
        raise RuntimeError(f"flagged accounts fetch failed: {e}") from e
    finally:
        conn.close()


def get_account_highest_risk_row(account_id: str) -> Optional[Dict[str, Any]]:
    """Return the single prediction row for this account with the highest risk_score (for explain)."""
    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT transaction_id, account_id, timestamp, amount, risk_score, top_features
            FROM predictions
            WHERE account_id = ?
            ORDER BY risk_score DESC
            LIMIT 1
            """,
            (str(account_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
        if d.get("top_features"):
            try:
                d["top_features"] = json.loads(d["top_features"])
            except Exception:
                d["top_features"] = []
        return d
    finally:
        conn.close()


def get_account_transactions(account_id: str, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
    """Return latest N transactions for an account with risk_score."""
    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT transaction_id, timestamp, amount, risk_score
            FROM predictions
            WHERE account_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(account_id), limit),
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        raise RuntimeError(f"account fetch failed: {e}") from e
    finally:
        conn.close()


def get_transaction_row(transaction_id: str) -> Optional[Dict[str, Any]]:
    """Get one prediction row for explain context (transaction_id, account_id, timestamp, amount, risk_score, top_features, etc.)."""
    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT transaction_id, account_id, timestamp, amount, risk_score, top_features FROM predictions WHERE transaction_id = ?",
            (str(transaction_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
        if d.get("top_features"):
            try:
                d["top_features"] = json.loads(d["top_features"])
            except Exception:
                d["top_features"] = []
        return d
    finally:
        conn.close()


def get_explain_cache(transaction_id: str) -> Optional[Dict[str, Any]]:
    """Return cached explanation if present: { summary_text, model_version }."""
    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT summary_text, model_version FROM explanations WHERE transaction_id = ?",
            (str(transaction_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"summary_text": row[0], "model_version": row[1]}
    finally:
        conn.close()


def set_explain_cache(transaction_id: str, summary_text: str, model_version: str) -> None:
    """Store explanation in cache."""
    from datetime import datetime
    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        created = datetime.utcnow().isoformat() + "Z"
        if DB_MODE == "sqlite":
            cur.execute(
                "INSERT OR REPLACE INTO explanations (transaction_id, summary_text, model_version, created_at) VALUES (?, ?, ?, ?)",
                (transaction_id, summary_text, model_version, created),
            )
        else:
            try:
                cur.execute("DELETE FROM explanations WHERE transaction_id = ?", (transaction_id,))
            except Exception:
                pass
            cur.execute(
                "INSERT INTO explanations (transaction_id, summary_text, model_version, created_at) VALUES (?, ?, ?, ?)",
                (transaction_id, summary_text, model_version, created),
            )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def get_all_predictions_for_graph(account_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return prediction rows for graph or account. If account_id given, filter by it."""
    init_db()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        if account_id:
            cur.execute(
                "SELECT transaction_id, account_id, timestamp, amount, risk_score, top_features FROM predictions WHERE account_id = ?",
                (str(account_id),),
            )
        else:
            cur.execute(
                "SELECT transaction_id, account_id, timestamp, amount, risk_score, top_features FROM predictions"
            )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        out = []
        for row in rows:
            d = dict(zip(cols, row))
            if d.get("top_features"):
                try:
                    d["top_features"] = json.loads(d["top_features"])
                except Exception:
                    d["top_features"] = []
            out.append(d)
        return out
    finally:
        conn.close()
