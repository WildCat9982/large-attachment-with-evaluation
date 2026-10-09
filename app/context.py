"""Redis 8: the working context.

- chunk:{doc}:{seq}   hashes, full-text indexed by the built-in query engine
- chat:{doc}          recent conversation turns (list, with TTL)
- progress:{doc}      live job progress for the UI
- jobs                stream consumed by the worker

Nothing here is the source of truth; SQLite is.
"""
import json
import re

import redis

import config

INDEX = "idx:chunks"
STREAM = "jobs"
GROUP = "workers"

# Redis' default stopwords; a query made only of these would match nothing.
STOPWORDS = set(
    "a is the an and are as at be but by for if in into it no not of on or such "
    "that their then there these they this to was will with what which who how "
    "why when where does do did can could would should about from have has had".split()
)


def client():
    # RESP2 keeps the FT.SEARCH reply a plain list: [total, key, key, ...]
    return redis.Redis.from_url(config.REDIS_URL, decode_responses=True, protocol=2, socket_timeout=30)


def ensure_index(r):
    try:
        r.execute_command(
            "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", "chunk:",
            "SCHEMA", "doc_id", "TAG", "seq", "NUMERIC", "text", "TEXT",
        )
    except redis.ResponseError as e:
        if "already exists" not in str(e).lower():
            raise


# --- job queue and progress --------------------------------------------------

def ensure_group(r):
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def enqueue(r, job_type, doc_id):
    r.xadd(STREAM, {"type": job_type, "doc_id": doc_id})


def set_progress(r, doc_id, stage, done=0, total=0):
    key = f"progress:{doc_id}"
    r.hset(key, mapping={"stage": stage, "done": done, "total": total})
    r.expire(key, 3600)


def get_progress(r, doc_id):
    p = r.hgetall(f"progress:{doc_id}")
    if not p:
        return None
    return {"stage": p.get("stage", ""), "done": int(p.get("done", 0)), "total": int(p.get("total", 0))}


# --- chunks ------------------------------------------------------------------

def index_chunks(r, doc_id, rows):
    """rows: iterable of (seq, loc, text)."""
    pipe = r.pipeline(transaction=False)
    for seq, loc, text in rows:
        pipe.hset(f"chunk:{doc_id}:{seq}",
                  mapping={"doc_id": doc_id, "seq": seq, "loc": loc or "", "text": text})
    pipe.execute()


def mark_indexed(r, doc_id, n_chunks):
    r.set(f"indexed:{doc_id}", n_chunks)


def is_indexed(r, doc_id):
    return bool(r.exists(f"indexed:{doc_id}"))


def drop_document(r, doc_id):
    batch = []
    for key in r.scan_iter(match=f"chunk:{doc_id}:*", count=1000):
        batch.append(key)
        if len(batch) >= 500:
            r.unlink(*batch)
            batch = []
    if batch:
        r.unlink(*batch)
    r.delete(f"indexed:{doc_id}", f"chat:{doc_id}", f"progress:{doc_id}")


def to_terms(texts, limit=16):
    """Turn free text into safe, de-duplicated search terms."""
    terms = []
    for t in texts:
        for w in re.findall(r"[^\W_]+", str(t).lower()):
            if len(w) > 1 and w not in STOPWORDS and w not in terms:
                terms.append(w)
    return terms[:limit]


def _get_chunks(r, doc_id, seqs):
    pipe = r.pipeline(transaction=False)
    for s in seqs:
        pipe.hgetall(f"chunk:{doc_id}:{s}")
    return [(s, h) for s, h in zip(seqs, pipe.execute()) if h]


def _hits(r, doc_id, terms, k):
    """Chunk numbers of the top-k BM25 matches for any of the terms."""
    if not terms:
        return []
    query = f"@doc_id:{{{doc_id}}} @text:({'|'.join(terms)})"
    reply = r.execute_command(
        "FT.SEARCH", INDEX, query, "NOCONTENT", "LIMIT", "0", str(k), "DIALECT", "2"
    )
    return [int(key.rsplit(":", 1)[1]) for key in reply[1:]]


def search(r, doc_id, terms, k=None):
    """BM25 search within one document.

    Returns (passages, matched). Each hit is widened with its neighbouring
    chunks and adjacent hits are merged, so the model sees continuous text.
    If nothing matches, the opening of the document is returned instead and
    matched is False.
    """
    k = k or config.TOP_K
    hit_seqs = _hits(r, doc_id, terms, k)
    matched = bool(hit_seqs)
    if matched:
        wanted = sorted({n for s in hit_seqs for n in (s - 1, s, s + 1) if n >= 0})
    else:
        wanted = list(range(k))

    passages = []
    for seq, h in _get_chunks(r, doc_id, wanted):
        if passages and passages[-1]["seq_to"] == seq - 1:
            passages[-1]["seq_to"] = seq
            passages[-1]["text"] += "\n" + h["text"]
        else:
            passages.append({"seq_from": seq, "seq_to": seq, "loc": h.get("loc", ""), "text": h["text"]})
    return passages, matched


# --- conversation ------------------------------------------------------------

def get_history(r, doc_id):
    """Recent turns as [{'role','content'}], or None if Redis has none."""
    items = r.lrange(f"chat:{doc_id}", 0, -1)
    return [json.loads(i) for i in items] if items else None


def set_history(r, doc_id, messages):
    key = f"chat:{doc_id}"
    keep = messages[-2 * config.HISTORY_TURNS:]
    pipe = r.pipeline()
    pipe.delete(key)
    if keep:
        pipe.rpush(key, *[json.dumps({"role": m["role"], "content": m["content"]}) for m in keep])
        pipe.expire(key, config.CHAT_TTL_SECONDS)
    pipe.execute()


def append_turn(r, doc_id, question, answer):
    key = f"chat:{doc_id}"
    pipe = r.pipeline()
    pipe.rpush(key, json.dumps({"role": "user", "content": question}),
               json.dumps({"role": "assistant", "content": answer}))
    pipe.ltrim(key, -2 * config.HISTORY_TURNS, -1)
    pipe.expire(key, config.CHAT_TTL_SECONDS)
    pipe.execute()
