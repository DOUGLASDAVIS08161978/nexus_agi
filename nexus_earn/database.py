"""
Nexus Earn — Database
SQLite-backed storage for users, API keys, usage, and billing.
No external database needed to get started.
"""

import sqlite3
import secrets
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

DB_PATH = Path(__file__).parent / "nexus_earn.db"

TIERS = {
    "free":      {"req_per_day": 10,    "price_monthly": 0},
    "starter":   {"req_per_day": 500,   "price_monthly": 19},
    "pro":       {"req_per_day": 5_000, "price_monthly": 49},
    "unlimited": {"req_per_day": -1,    "price_monthly": 99},
}

STRIPE_PRICE_IDS: dict = {
    # Fill these in from your Stripe dashboard after creating products
    "starter":   "",   # e.g. price_xxxxxxxxxxxx
    "pro":       "",
    "unlimited": "",
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                email           TEXT UNIQUE NOT NULL,
                api_key         TEXT UNIQUE NOT NULL,
                tier            TEXT NOT NULL DEFAULT 'free',
                stripe_customer TEXT,
                stripe_sub      TEXT,
                active          INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                endpoint    TEXT NOT NULL,
                tokens_in   INTEGER DEFAULT 0,
                tokens_out  INTEGER DEFAULT 0,
                cost_usd    REAL DEFAULT 0,
                created_at  TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_usage_user_date
                ON usage(user_id, created_at);
        """)


# ── User management ───────────────────────────────────────────────────────────

def create_user(email: str) -> dict:
    api_key = "nxs_" + secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as db:
        db.execute(
            "INSERT INTO users (email, api_key, tier, created_at) VALUES (?,?,?,?)",
            (email.lower().strip(), api_key, "free", now)
        )
    return {"email": email, "api_key": api_key, "tier": "free"}


def get_user_by_key(api_key: str) -> Optional[sqlite3.Row]:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM users WHERE api_key=? AND active=1", (api_key,)
        ).fetchone()
    return row


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM users WHERE email=?", (email.lower().strip(),)
        ).fetchone()
    return row


def set_user_tier(user_id: int, tier: str,
                  stripe_customer: str = "", stripe_sub: str = "") -> None:
    with get_db() as db:
        db.execute(
            """UPDATE users
               SET tier=?, stripe_customer=COALESCE(NULLIF(?,''
               ), stripe_customer),
                   stripe_sub=COALESCE(NULLIF(?,''), stripe_sub)
               WHERE id=?""",
            (tier, stripe_customer, stripe_sub, user_id)
        )


def cancel_user_sub(stripe_sub: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE users SET tier='free', stripe_sub=NULL WHERE stripe_sub=?",
            (stripe_sub,)
        )


# ── Usage tracking ────────────────────────────────────────────────────────────

def log_usage(user_id: int, endpoint: str,
              tokens_in: int, tokens_out: int, cost_usd: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as db:
        db.execute(
            "INSERT INTO usage (user_id, endpoint, tokens_in, tokens_out, cost_usd, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (user_id, endpoint, tokens_in, tokens_out, cost_usd, now)
        )


def get_usage_today(user_id: int) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    with get_db() as db:
        row = db.execute(
            "SELECT COUNT(*) as cnt FROM usage WHERE user_id=? AND created_at >= ?",
            (user_id, today)
        ).fetchone()
    return row["cnt"] if row else 0


def get_usage_summary(user_id: int) -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    month = today[:7]   # YYYY-MM
    with get_db() as db:
        today_row = db.execute(
            "SELECT COUNT(*) cnt, SUM(tokens_in+tokens_out) toks, SUM(cost_usd) cost"
            " FROM usage WHERE user_id=? AND created_at>=?",
            (user_id, today)
        ).fetchone()
        month_row = db.execute(
            "SELECT COUNT(*) cnt, SUM(tokens_in+tokens_out) toks, SUM(cost_usd) cost"
            " FROM usage WHERE user_id=? AND created_at>=?",
            (user_id, month)
        ).fetchone()
    return {
        "today":     {"requests": today_row["cnt"] or 0, "tokens": today_row["toks"] or 0},
        "this_month":{"requests": month_row["cnt"] or 0, "tokens": month_row["toks"] or 0,
                      "cost_usd": round(month_row["cost"] or 0, 4)},
    }
