"""
Append-only audit log for radiologist decisions (SQLite).

* Every Approve/Reject decision is one row; rows can never be updated or
  deleted (SQLite triggers abort UPDATE/DELETE).
* Rows are hash-chained: each row stores sha256(previous_hash + its own
  content), so any later tampering with the file is detectable via verify().
"""
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.environ.get("AUDIT_DB", str(Path(__file__).parent / "audit.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    scan_id       TEXT NOT NULL,
    filename      TEXT,
    decision      TEXT NOT NULL CHECK (decision IN ('approved','rejected')),
    reviewer      TEXT NOT NULL,
    comment       TEXT,
    confidence    REAL,
    tumour_volume_ml REAL,
    model_checkpoint TEXT,
    prev_hash     TEXT NOT NULL,
    row_hash      TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""

FIELDS = ("timestamp_utc", "scan_id", "filename", "decision", "reviewer",
          "comment", "confidence", "tumour_volume_ml", "model_checkpoint")
GENESIS = "0" * 64


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.executescript(SCHEMA)


def _hash(prev_hash, record):
    payload = json.dumps([record[f] for f in FIELDS], sort_keys=False, default=str)
    return hashlib.sha256((prev_hash + payload).encode()).hexdigest()


def record_decision(scan_id, decision, reviewer, comment="", filename=None,
                    confidence=None, tumour_volume_ml=None, model_checkpoint=None):
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'")
    if not reviewer or not reviewer.strip():
        raise ValueError("reviewer name is required")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scan_id": scan_id,
        "filename": filename,
        "decision": decision,
        "reviewer": reviewer.strip(),
        "comment": (comment or "").strip(),
        "confidence": confidence,
        "tumour_volume_ml": tumour_volume_ml,
        "model_checkpoint": model_checkpoint,
    }
    with _conn() as c:
        c.execute("BEGIN IMMEDIATE")  # serialise writers so the chain stays linear
        row = c.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev = row["row_hash"] if row else GENESIS
        h = _hash(prev, record)
        cur = c.execute(
            f"INSERT INTO audit_log ({','.join(FIELDS)}, prev_hash, row_hash) "
            f"VALUES ({','.join('?' * len(FIELDS))}, ?, ?)",
            [record[f] for f in FIELDS] + [prev, h],
        )
        record.update(id=cur.lastrowid, prev_hash=prev, row_hash=h)
    return record


def list_decisions(limit=100):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def verify():
    """Recompute the whole chain. Returns (ok, number_of_rows, first_bad_id)."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
    prev = GENESIS
    for r in rows:
        rec = {f: r[f] for f in FIELDS}
        if r["prev_hash"] != prev or r["row_hash"] != _hash(prev, rec):
            return False, len(rows), r["id"]
        prev = r["row_hash"]
    return True, len(rows), None
