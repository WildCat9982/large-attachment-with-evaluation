"""The three things the app does: ingest a file, summarise it, answer a question."""
import contextvars
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import context
import llm
import parsers
import persistence as db

TABLE_EXT = {"csv", "xlsx", "json"}


def upload_path(doc_id, ext):
    return os.path.join(config.UPLOAD_DIR, f"{doc_id}.{ext}")


def _upload_path(doc):
    return upload_path(doc["id"], doc["ext"])


def table_db(doc_id):
    return os.path.join(config.TABLE_DIR, f"{doc_id}.db")


# --- ingest ------------------------------------------------------------------

def ingest(r, doc_id):
    doc = db.get_document(doc_id)
    if doc is None:
        return
    db.update_document(doc_id, status="processing", error=None)
    try:
        if doc["ext"] in TABLE_EXT and _ingest_tables(r, doc):
            return
        _ingest_text(r, doc)
    except Exception as e:
        db.update_document(doc_id, status="failed", error=f"{type(e).__name__}: {e}"[:500])
        raise


def _ingest_tables(r, doc):
    context.set_progress(r, doc["id"], "Loading rows into SQLite")
    out_db = table_db(doc["id"])
    if os.path.exists(out_db):
        os.remove(out_db)
    try:
        tables = parsers.load_tables(_upload_path(doc), doc["ext"], out_db)
    except parsers.NoTables:
        os.remove(out_db)
        return False
    schema = parsers.describe_tables(out_db, tables)
    db.update_document(doc["id"], kind="table", status="ready",
                       meta={"tables": tables, "schema": schema})
    return True


def _ingest_text(r, doc):
    doc_id = doc["id"]
    context.ensure_index(r)
    context.drop_document(r, doc_id)
    db.delete_chunks(doc_id)
    context.set_progress(r, doc_id, "Extracting text")

    seq = n_chars = 0
    batch = []

    def flush():
        db.add_chunks([(doc_id, s, loc, t) for s, loc, t in batch])
        context.index_chunks(r, doc_id, batch)
        context.set_progress(r, doc_id, "Indexing", seq, 0)
        batch.clear()

    for loc, text in parsers.extract_text(_upload_path(doc), doc["ext"]):
        for piece in parsers.chunk_text(text, config.CHUNK_CHARS):
            batch.append((seq, loc, piece))
            seq += 1
            n_chars += len(piece)
            if len(batch) >= 500:
                flush()
    if batch:
        flush()

    if seq == 0:
        raise ValueError("No text could be extracted. If this is a scanned PDF, it needs OCR, "
                         "which this POC does not include.")
    context.mark_indexed(r, doc_id, seq)
    db.update_document(doc_id, kind="text", status="ready", n_chunks=seq, n_chars=n_chars)


def delete(r, doc_id):
    """Remove a document everywhere: Redis, SQLite, and its files."""
    doc = db.get_document(doc_id)
    context.drop_document(r, doc_id)
    db.delete_document(doc_id)
    paths = [table_db(doc_id)] + ([_upload_path(doc)] if doc else [])
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def ensure_context(r, doc):
    """Rebuild a document's Redis chunks from SQLite if Redis was emptied."""
    if doc["kind"] != "text" or context.is_indexed(r, doc["id"]):
        return
    context.ensure_index(r)
    batch = []
    for row in db.iter_chunks(doc["id"]):
        batch.append(row)
        if len(batch) >= 500:
            context.index_chunks(r, doc["id"], batch)
            batch = []
    if batch:
        context.index_chunks(r, doc["id"], batch)
    context.mark_indexed(r, doc["id"], doc["n_chunks"])


# --- summarise ---------------------------------------------------------------

MAP_SYSTEM = (
    "You are taking notes on one section of a longer document so that a full summary can be "
    "written later from the notes alone. Record the main points, key facts, figures, names, "
    "dates and conclusions in this section, in under 200 words. Notes only, no preamble."
)
REDUCE_SYSTEM = (
    "You are condensing notes taken in order from consecutive sections of one document. Merge "
    "them into a single set of notes under 400 words, keeping the order, the key facts and "
    "figures. Notes only, no preamble."
)
FINAL_SYSTEM = (
    "Write a summary of the document for someone who has not read it. Start with a two or three "
    "sentence overview, then cover the main points in the order the document presents them, "
    "including the important facts, figures and conclusions. Use only the provided material. "
    "Aim for 250 to 450 words in plain Markdown."
)
TABLE_SUMMARY_SYSTEM = (
    "You are describing a dataset to someone who has not opened it, using its schema and "
    "column statistics. Say what the data appears to record, how large it is, what the key "
    "columns are and their ranges, and note anything notable such as gaps or dominant values. "
    "Use only the provided material. Aim for 200 to 350 words in plain Markdown."
)


def _windows(doc_id):
    """Group consecutive chunks into windows of about SUMMARY_WINDOW_CHARS."""
    buf, size = [], 0
    for _seq, loc, text in db.iter_chunks(doc_id):
        buf.append(text)
        size += len(text)
        if size >= config.SUMMARY_WINDOW_CHARS:
            yield "\n".join(buf)
            buf, size = [], 0
    if buf:
        yield "\n".join(buf)


def _pack(notes, limit):
    """Group notes, in order, into batches that each fit in one prompt."""
    batch, size = [], 0
    for n in notes:
        if batch and size + len(n) > limit:
            yield batch
            batch, size = [], 0
        batch.append(n)
        size += len(n)
    if batch:
        yield batch


def summarize(r, doc_id):
    doc = db.get_document(doc_id)
    if doc is None:
        return
    db.set_summary(doc_id, "running")
    try:
        with llm.tagged(doc_id=doc_id, ref=f"summary:{doc_id}"):
            if doc["kind"] == "table":
                summary, coverage = _summarize_table(r, doc)
            else:
                summary, coverage = _summarize_text(r, doc)
        db.set_summary(doc_id, "done", summary=summary, coverage=coverage)
    except Exception as e:
        db.set_summary(doc_id, "failed", error=f"{type(e).__name__}: {e}"[:500])
        raise


def _summarize_table(r, doc):
    context.set_progress(r, doc["id"], "Profiling columns")
    profile = parsers.profile_tables(table_db(doc["id"]), doc["meta"]["tables"])
    context.set_progress(r, doc["id"], "Writing summary")
    prompt = f"File: {doc['filename']}\n\nSchema and samples:\n{doc['meta']['schema']}\n\nColumn statistics:\n{profile}"
    summary = llm.complete(TABLE_SUMMARY_SYSTEM, [{"role": "user", "content": prompt}], purpose="summary_final")
    return summary, "Based on statistics computed over every row."


def _summarize_text(r, doc):
    doc_id = doc["id"]
    windows = list(_windows(doc_id))
    total = len(windows)
    cap = config.SUMMARY_MAX_WINDOWS
    if cap and total > cap:
        picked = sorted({round(i * (total - 1) / (cap - 1)) for i in range(cap)}) if cap > 1 else [0]
        windows = [windows[i] for i in picked]
        coverage = (f"Sampled: read {len(windows)} of {total} sections, evenly spaced "
                    f"(about {100 * len(windows) // total}% of the text). "
                    f"Raise SUMMARY_MAX_WINDOWS to read more.")
    else:
        coverage = "Read the full text."

    if len(windows) == 1:
        context.set_progress(r, doc_id, "Writing summary")
        return llm.complete(FINAL_SYSTEM, [{"role": "user", "content": windows[0]}], purpose="summary_final"), coverage

    # Map: notes per window, a few at a time.
    notes = [None] * len(windows)
    context.set_progress(r, doc_id, "Reading sections", 0, len(windows))
    with ThreadPoolExecutor(max_workers=config.SUMMARY_CONCURRENCY) as pool:
        futures = {
            # copy_context carries the usage tags into the pool thread
            pool.submit(contextvars.copy_context().run, llm.complete, MAP_SYSTEM,
                        [{"role": "user", "content": w}], True, 500, "summary_map"): i
            for i, w in enumerate(windows)
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            notes[futures[fut]] = fut.result()
            context.set_progress(r, doc_id, "Reading sections", done, len(windows))

    # Reduce: merge notes until they fit in one prompt.
    while sum(len(n) for n in notes) > config.SUMMARY_WINDOW_CHARS:
        context.set_progress(r, doc_id, "Merging notes")
        batches = list(_pack(notes, config.SUMMARY_WINDOW_CHARS))
        notes = [llm.complete(REDUCE_SYSTEM, [{"role": "user", "content": "\n\n---\n\n".join(b)}], True, 1000, "summary_reduce")
                 for b in batches]

    context.set_progress(r, doc_id, "Writing summary")
    joined = "\n\n---\n\n".join(notes)
    summary = llm.complete(FINAL_SYSTEM, [{"role": "user", "content": f"Notes from the document, in order:\n\n{joined}"}],
                           purpose="summary_final")
    return summary, coverage


# --- answer ------------------------------------------------------------------

KEYWORD_SYSTEM = (
    "You write search keywords for a keyword search engine. Given a conversation and the "
    "latest question about a document, reply with a JSON array of 4 to 10 single words likely "
    "to appear in the passages that answer the question. Resolve references such as 'it' or "
    "'that' using the conversation, and include likely synonyms. Reply with the JSON array only."
)
ANSWER_SYSTEM = (
    "You answer questions about one document the user uploaded. Use only the excerpts and the "
    "summary provided with the question. Cite the excerpts you rely on by number, like [2]. "
    "The excerpts are the passages a keyword search found, not the whole document, so if they "
    "do not contain the answer, say that plainly and suggest how the user could rephrase, "
    "rather than guessing."
)
SQL_SYSTEM = (
    "You answer questions about a dataset stored in SQLite by writing a query. Reply with "
    "exactly one SQLite SELECT statement in a ```sql block and nothing else. Quote identifiers "
    "with double quotes, aggregate rather than listing raw rows where possible, and add a "
    "LIMIT when listing rows. If the question can be answered from the schema alone or is "
    "not about the data, reply with the answer in plain text and no code block."
)
RESULT_SYSTEM = (
    "You answer questions about a dataset. You are given the SQL that was run and its result. "
    "Answer the question directly from the result, stating figures exactly. If the result was "
    "truncated or does not answer the question, say so."
)


def _history(r, doc_id):
    """Recent turns from Redis; refilled from SQLite if the key has expired."""
    hist = context.get_history(r, doc_id)
    if hist is None:
        hist = [{"role": m["role"], "content": m["content"]}
                for m in db.get_messages(doc_id, last=2 * config.HISTORY_TURNS)]
        if hist:
            context.set_history(r, doc_id, hist)
    while hist and hist[0]["role"] != "user":
        hist = hist[1:]
    return hist


def answer(r, doc_id, question):
    """Answer a question, record the turn, and return (answer, meta)."""
    doc = db.get_document(doc_id)
    history = _history(r, doc_id)
    ref = uuid.uuid4().hex  # ties this answer to its model calls in the usage log
    started = time.perf_counter()
    with llm.tagged(doc_id=doc_id, ref=ref):
        if doc["kind"] == "table":
            text, meta = _answer_table(doc, history, question)
        else:
            ensure_context(r, doc)
            text, meta = _answer_text(r, doc, history, question)
    meta = {**meta, "ref": ref, "elapsed_ms": int((time.perf_counter() - started) * 1000)}
    db.add_message(doc_id, "user", question)
    db.add_message(doc_id, "assistant", text, meta)
    context.append_turn(r, doc_id, question, text)
    return text, meta


def _keywords(history, question):
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-4:])
    prompt = f"Conversation so far:\n{convo or '(none)'}\n\nLatest question: {question}"
    try:
        raw = llm.complete(KEYWORD_SYSTEM, [{"role": "user", "content": prompt}], True, 200, "keywords")
        words = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
    except Exception:
        words = []  # fall back to the words of the question itself
    return context.to_terms([*words, question])


def _answer_text(r, doc, history, question):
    terms = _keywords(history, question)
    passages, matched = context.search(r, doc["id"], terms)
    excerpts = "\n\n".join(
        f"[{i}]{' (' + p['loc'] + ')' if p['loc'] else ''}\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    summ = db.get_summary(doc["id"])
    parts = [f"Document: {doc['filename']}"]
    if summ and summ["status"] == "done":
        parts.append(f"Summary of the whole document:\n{summ['summary']}")
    if not matched:
        parts.append("Note: the search found no matching passages; the excerpts below are "
                     "just the opening of the document.")
    parts.append(f"Excerpts:\n{excerpts}")
    parts.append(f"Question: {question}")
    text = llm.complete(ANSWER_SYSTEM, [*history, {"role": "user", "content": "\n\n".join(parts)}],
                        purpose="answer")
    meta = {"terms": terms, "matched": matched,
            "sources": [{"n": i, "loc": p["loc"], "seq_from": p["seq_from"], "seq_to": p["seq_to"],
                         "text": p["text"][:700]}
                        for i, p in enumerate(passages, start=1)]}
    return text, meta


def _extract_sql(reply):
    m = re.search(r"```(?:sql)?\s*(.+?)```", reply, re.S | re.I)
    return m.group(1).strip() if m else None


def _answer_table(doc, history, question):
    schema = doc["meta"]["schema"]
    ask = f"Dataset: {doc['filename']}\n\n{schema}\n\nQuestion: {question}"
    msgs = [*history, {"role": "user", "content": ask}]
    reply = llm.complete(SQL_SYSTEM, msgs, max_tokens=800, purpose="sql")
    sql = _extract_sql(reply)
    if sql is None:
        return reply, {}

    db_path = table_db(doc["id"])
    limits = (config.SQL_MAX_ROWS, config.SQL_TIMEOUT_SECONDS)
    repaired = False
    try:
        cols, rows, truncated = parsers.run_sql(db_path, sql, *limits)
    except Exception as e:
        # One repair attempt, showing the model its error.
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"That query failed: {e}\nReply with a corrected query."}]
        repaired = True
        reply = llm.complete(SQL_SYSTEM, msgs, max_tokens=800, purpose="sql_repair")
        sql = _extract_sql(reply) or sql
        try:
            cols, rows, truncated = parsers.run_sql(db_path, sql, *limits)
        except Exception as e2:
            return (f"I couldn't run a query for that question ({e2}). Try rephrasing it.",
                    {"sql": sql, "error": str(e2), "repaired": True})

    lines = [" | ".join(cols)] + [" | ".join("" if v is None else str(v)[:200] for v in row) for row in rows]
    result = "\n".join(lines)
    if truncated:
        result += f"\n(truncated to the first {config.SQL_MAX_ROWS} rows)"
    prompt = f"Question: {question}\n\nSQL run:\n{sql}\n\nResult ({len(rows)} rows):\n{result}"
    text = llm.complete(RESULT_SYSTEM, [*history, {"role": "user", "content": prompt}], purpose="result")
    return text, {"sql": sql, "columns": cols, "rows": [list(r) for r in rows[:50]],
                  "truncated": truncated, "repaired": repaired}
