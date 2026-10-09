"""SQLite: the durable record (documents, chunks, summaries, chat messages).

Redis can be wiped at any time; everything in it is rebuilt from here.
"""
import json
import os
import sqlite3
from contextlib import contextmanager

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    kind TEXT,                      -- 'text' or 'table', set by ingestion
    status TEXT NOT NULL,           -- queued | processing | ready | failed
    error TEXT,
    n_chunks INTEGER DEFAULT 0,
    n_chars INTEGER DEFAULT 0,
    meta TEXT,                      -- JSON; table schema for tabular files
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    doc_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    loc TEXT,                       -- e.g. 'p.12' for PDFs
    text TEXT NOT NULL,
    PRIMARY KEY (doc_id, seq)
);
CREATE TABLE IF NOT EXISTS summaries (
    doc_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,           -- queued | running | done | failed
    summary TEXT,
    coverage TEXT,
    error TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT,                      -- JSON; sources or SQL behind an answer
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_doc ON messages (doc_id, id);

-- One row per model call: where speed and cost figures come from.
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    source TEXT NOT NULL,           -- 'live' (real use) or 'eval' (test runs)
    doc_id TEXT,
    ref TEXT,                       -- groups the calls behind one answer or summary
    purpose TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ref ON llm_calls (ref);

-- Evaluation: one row per run, per test document, per test question.
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    status TEXT NOT NULL,           -- running | done
    settings TEXT,                  -- JSON: models and retrieval settings used
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS eval_docs (
    run_id INTEGER NOT NULL,
    doc TEXT NOT NULL,
    data TEXT NOT NULL,             -- JSON: ingestion and summary results
    PRIMARY KEY (run_id, doc)
);
CREATE TABLE IF NOT EXISTS eval_results (
    run_id INTEGER NOT NULL,
    case_id TEXT NOT NULL,
    data TEXT NOT NULL,             -- JSON: question, answer, scores, evidence
    PRIMARY KEY (run_id, case_id)
);
"""

# Columns added after the first version; applied to existing databases on start.
MIGRATIONS = [
    "ALTER TABLE documents ADD COLUMN is_eval INTEGER DEFAULT 0",
    "ALTER TABLE messages ADD COLUMN feedback INTEGER",  # 1 = thumbs up, 0 = thumbs down
]


@contextmanager
def connect():
    conn = sqlite3.connect(config.SQLITE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init():
    for d in (config.DATA_DIR, config.UPLOAD_DIR, config.TABLE_DIR):
        os.makedirs(d, exist_ok=True)
    with connect() as c:
        c.execute("PRAGMA journal_mode=WAL")  # app and worker share the file
        c.executescript(SCHEMA)
        for statement in MIGRATIONS:
            try:
                c.execute(statement)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise


# --- documents ---------------------------------------------------------------

def add_document(doc_id, filename, ext, size_bytes, is_eval=False):
    with connect() as c:
        c.execute(
            "INSERT INTO documents (id, filename, ext, size_bytes, status, is_eval) VALUES (?,?,?,?, 'queued', ?)",
            (doc_id, filename, ext, size_bytes, int(is_eval)),
        )


def update_document(doc_id, **fields):
    if "meta" in fields and not isinstance(fields["meta"], (str, type(None))):
        fields["meta"] = json.dumps(fields["meta"])
    sets = ", ".join(f"{k} = ?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE documents SET {sets} WHERE id = ?", (*fields.values(), doc_id))


def _doc(row):
    if row is None:
        return None
    d = dict(row)
    d["meta"] = json.loads(d["meta"]) if d["meta"] else {}
    return d


def get_document(doc_id):
    with connect() as c:
        return _doc(c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone())


def list_documents():
    with connect() as c:
        rows = c.execute("SELECT * FROM documents WHERE COALESCE(is_eval, 0) = 0 "
                         "ORDER BY created_at DESC, rowid DESC").fetchall()
    return [_doc(r) for r in rows]


def delete_document(doc_id):
    with connect() as c:
        for table, col in (("chunks", "doc_id"), ("summaries", "doc_id"),
                           ("messages", "doc_id"), ("documents", "id")):
            c.execute(f"DELETE FROM {table} WHERE {col} = ?", (doc_id,))


# --- chunks ------------------------------------------------------------------

def add_chunks(rows):
    """rows: iterable of (doc_id, seq, loc, text)."""
    with connect() as c:
        c.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?)", rows)


def delete_chunks(doc_id):
    with connect() as c:
        c.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))


def iter_chunks(doc_id, batch=1000):
    """Yield (seq, loc, text) in document order without loading everything."""
    last = -1
    while True:
        with connect() as c:
            rows = c.execute(
                "SELECT seq, loc, text FROM chunks WHERE doc_id = ? AND seq > ? ORDER BY seq LIMIT ?",
                (doc_id, last, batch),
            ).fetchall()
        if not rows:
            return
        for r in rows:
            yield r["seq"], r["loc"], r["text"]
        last = rows[-1]["seq"]


# --- summaries ---------------------------------------------------------------

def set_summary(doc_id, status, summary=None, coverage=None, error=None):
    with connect() as c:
        c.execute(
            "INSERT INTO summaries (doc_id, status, summary, coverage, error) VALUES (?,?,?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET status=excluded.status, summary=excluded.summary, "
            "coverage=excluded.coverage, error=excluded.error, updated_at=CURRENT_TIMESTAMP",
            (doc_id, status, summary, coverage, error),
        )


def get_summary(doc_id):
    with connect() as c:
        row = c.execute("SELECT * FROM summaries WHERE doc_id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


# --- chat messages -----------------------------------------------------------

def add_message(doc_id, role, content, meta=None):
    with connect() as c:
        c.execute(
            "INSERT INTO messages (doc_id, role, content, meta) VALUES (?,?,?,?)",
            (doc_id, role, content, json.dumps(meta) if meta else None),
        )


def get_messages(doc_id, last=None):
    sql = "SELECT id, role, content, meta, feedback FROM messages WHERE doc_id = ? ORDER BY id"
    with connect() as c:
        rows = c.execute(sql, (doc_id,)).fetchall()
    out = [{"id": r["id"], "role": r["role"], "content": r["content"], "feedback": r["feedback"],
            "meta": json.loads(r["meta"]) if r["meta"] else {}} for r in rows]
    return out[-last:] if last else out


def delete_messages(doc_id):
    with connect() as c:
        c.execute("DELETE FROM messages WHERE doc_id = ?", (doc_id,))


def set_feedback(message_id, value):
    with connect() as c:
        c.execute("UPDATE messages SET feedback = ? WHERE id = ?", (value, message_id))


# --- usage log ---------------------------------------------------------------

def log_llm_call(source, doc_id, ref, purpose, model, input_tokens, output_tokens, duration_ms):
    with connect() as c:
        c.execute(
            "INSERT INTO llm_calls (source, doc_id, ref, purpose, model, input_tokens, output_tokens, duration_ms) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (source, doc_id, ref, purpose, model, input_tokens, output_tokens, duration_ms),
        )


def llm_calls(ref=None, source=None):
    """Logged model calls, filtered by the answer/summary they belong to or by source."""
    where, args = [], []
    if ref is not None:
        where.append("ref = ?")
        args.append(ref)
    if source is not None:
        where.append("source = ?")
        args.append(source)
    sql = "SELECT * FROM llm_calls" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id"
    with connect() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def live_answers():
    """Answers given to real users: timing, feedback and usage reference."""
    with connect() as c:
        rows = c.execute(
            "SELECT m.meta, m.feedback FROM messages m JOIN documents d ON d.id = m.doc_id "
            "WHERE m.role = 'assistant' AND COALESCE(d.is_eval, 0) = 0").fetchall()
    out = []
    for r in rows:
        meta = json.loads(r["meta"]) if r["meta"] else {}
        out.append({"elapsed_ms": meta.get("elapsed_ms"), "ref": meta.get("ref"), "feedback": r["feedback"]})
    return out


# --- evaluation --------------------------------------------------------------

def start_eval_run(label, settings):
    with connect() as c:
        cur = c.execute("INSERT INTO eval_runs (label, status, settings) VALUES (?, 'running', ?)",
                        (label, json.dumps(settings)))
        return cur.lastrowid


def finish_eval_run(run_id):
    with connect() as c:
        c.execute("UPDATE eval_runs SET status = 'done', finished_at = CURRENT_TIMESTAMP WHERE id = ?", (run_id,))


def save_eval_doc(run_id, doc, data):
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO eval_docs VALUES (?,?,?)", (run_id, doc, json.dumps(data)))


def save_eval_result(run_id, case_id, data):
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO eval_results VALUES (?,?,?)", (run_id, case_id, json.dumps(data)))


def list_eval_runs():
    with connect() as c:
        rows = c.execute("SELECT * FROM eval_runs ORDER BY id DESC").fetchall()
    return [{**dict(r), "settings": json.loads(r["settings"] or "{}")} for r in rows]


def get_eval_run(run_id):
    """(test documents, test questions) recorded for one run."""
    with connect() as c:
        docs = c.execute("SELECT data FROM eval_docs WHERE run_id = ? ORDER BY doc", (run_id,)).fetchall()
        results = c.execute("SELECT data FROM eval_results WHERE run_id = ? ORDER BY rowid", (run_id,)).fetchall()
    return [json.loads(r["data"]) for r in docs], [json.loads(r["data"]) for r in results]
