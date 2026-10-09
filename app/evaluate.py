"""Evaluation: run the test set through the real pipeline and score it.

    docker compose run --rm eval --label "baseline"

For each test document this ingests the file, asks every question, writes a
summary, scores everything, and stores the results in SQLite as one run. The
Evaluation page in the app reads those runs.

Scoring uses exact checks where it can (was the right passage retrieved, does
the query result match the reference query) and a judge model for the rest
(is the answer correct, is it supported by the evidence, does the summary
cover the key points).
"""
import argparse
import json
import math
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import config
import context
import llm
import parsers
import persistence as db
import pipeline

NOT_IN_DOC = "The document does not contain this information."
SUMMARY_SOURCE_CHARS = 120_000  # most of a document the judge is shown when grading its summary
SIZE_ORDER = ["small", "medium", "large", "xlarge"]

CORRECT_SYSTEM = (
    "You grade an answer given by a document question-answering system against a reference "
    "answer. Reply with JSON only: {\"verdict\": \"correct\" | \"partial\" | \"incorrect\", "
    "\"reason\": \"<one sentence>\"}.\n"
    "correct: the answer conveys the same facts as the reference. Different wording, extra "
    "correct detail and rounded numbers are fine.\n"
    "partial: only some of the reference facts are present, or the answer is right but also "
    "contains a material error.\n"
    "incorrect: the answer is wrong, or does not answer the question.\n"
    "If the reference says the document does not contain the information, the answer is "
    "correct only if it says the information is not available and does not invent one."
)
GROUNDED_SYSTEM = (
    "You check whether an answer is supported by the evidence the system was given. Reply with "
    "JSON only: {\"grounded\": true | false, \"unsupported\": [\"<claim>\", ...]}.\n"
    "The answer is grounded if every factual claim in it is stated in the evidence or follows "
    "directly from it. List each claim that is not. Statements that information is unavailable, "
    "citation markers like [2] and suggestions to rephrase are not claims; ignore them."
)
SUMMARY_SYSTEM = (
    "You grade a summary of a document. You are given the source material, a numbered list of "
    "key points, and the summary. Reply with JSON only: {\"points\": [true | false, ...], "
    "\"unsupported\": [\"<statement>\", ...]}.\n"
    "points: one entry per key point, in order; true if the summary conveys that point.\n"
    "unsupported: each statement in the summary that the source material does not support. "
    "Use an empty list if there are none."
)


# --- scoring helpers ---------------------------------------------------------

def _norm(text):
    return re.sub(r"\s+", " ", str(text)).casefold().strip()


def _judge(system, prompt):
    """Ask the judge model for a JSON verdict. Returns a dict, or None if unusable."""
    try:
        with llm.tagged(ref="judge"):
            raw = llm.complete(system, [{"role": "user", "content": prompt}],
                               max_tokens=700, purpose="judge", model=config.EVAL_JUDGE_MODEL)
        return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:
        return None  # counted as a judge error; the case is left unscored


def _same(a, b):
    """Do two result cells hold the same value? Numbers within 0.1%, text ignoring case."""
    try:
        x, y = float(a), float(b)
        return abs(x - y) <= max(0.01, 0.001 * abs(y))
    except (TypeError, ValueError):
        return _norm(a) == _norm(b)


def _result_matches(gold_rows, got_rows):
    """True if every value the reference query returned appears in the system's result."""
    got = [v for row in got_rows for v in row]
    return all(any(_same(v, g) for v in got) for row in gold_rows for g in row)


def _usage(ref):
    calls = db.llm_calls(ref=ref)
    return {
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
        "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
        "cost": sum(llm.cost(c["model"], c["input_tokens"] or 0, c["output_tokens"] or 0) for c in calls),
    }


def _passages(doc_id, sources):
    """Full text and page labels of the passages an answer used (stored previews are cut short)."""
    wanted = set()
    for s in sources:
        wanted.update(range(s["seq_from"], s["seq_to"] + 1))
    rows = [(loc, text) for seq, loc, text in db.iter_chunks(doc_id) if seq in wanted]
    return "\n".join(text for _loc, text in rows), {loc for loc, _text in rows}


# --- one question ------------------------------------------------------------

def run_case(r, doc, case, size=None):
    doc_id = doc["id"]
    is_table = doc["kind"] == "table"
    out = {"case_id": case["id"], "doc": doc["filename"], "ext": doc["ext"], "size": size,
           "kind": doc["kind"], "type": case["type"], "question": case["question"]}

    # Each question starts from a clean conversation; follow-ups get their one prior turn.
    db.delete_messages(doc_id)
    prior = []
    if case.get("previous"):
        prior = [{"role": "user", "content": case["previous"]["question"]},
                 {"role": "assistant", "content": case["previous"]["answer"]}]
        for m in prior:
            db.add_message(doc_id, m["role"], m["content"])
        out["previous"] = case["previous"]
    context.set_history(r, doc_id, prior)

    try:
        answer, meta = pipeline.answer(r, doc_id, case["question"])
    except Exception as e:
        return {**out, "error": f"{type(e).__name__}: {e}"[:300], "correctness": 0.0}
    out.update(answer=answer, elapsed_ms=meta["elapsed_ms"], **_usage(meta["ref"]))

    # Reference answer. For data questions it is whatever the reference query returns now.
    expected = case.get("expected", NOT_IN_DOC)
    gold_rows = None
    if case.get("gold_sql"):
        _cols, gold_rows, _ = parsers.run_sql(pipeline.table_db(doc_id), case["gold_sql"], 50, 30)
        expected += " Reference result: " + "; ".join(", ".join(str(v) for v in row) for row in gold_rows)
    out["expected"] = expected

    # Exact checks.
    if is_table:
        out["sql"] = meta.get("sql")
        out["sql_failed"] = bool(meta.get("error"))
        out["sql_repaired"] = bool(meta.get("repaired"))
        out["rows"] = meta.get("rows", [])[:20]
        out["columns"] = meta.get("columns", [])
        if gold_rows is not None:
            out["sql_match"] = bool(meta.get("rows")) and _result_matches(gold_rows, meta["rows"])
        if meta.get("sql"):
            evidence = f"SQL run:\n{meta['sql']}\n\nResult:\n" + "\n".join(
                " | ".join(str(v) for v in row) for row in [meta.get("columns", []), *meta.get("rows", [])])
        else:
            evidence = doc["meta"]["schema"]
    else:
        sources = meta.get("sources", [])
        evidence, locs = _passages(doc_id, sources)
        out["matched"] = meta.get("matched")
        out["terms"] = meta.get("terms", [])
        out["sources"] = [{"n": s["n"], "loc": s["loc"], "text": s["text"]} for s in sources]
        if case.get("evidence"):   # a phrase from the passage that holds the answer
            out["retrieval_hit"] = _norm(case["evidence"]) in _norm(evidence)
        elif case.get("page"):     # or, for PDFs, the page(s) that hold it
            pages = case["page"] if isinstance(case["page"], list) else [case["page"]]
            out["retrieval_hit"] = any(f"p.{p}" in locs for p in pages)
        if prior:
            evidence = f"Earlier in the conversation:\n{prior[0]['content']}\n{prior[1]['content']}\n\n{evidence}"

    # Judge: correctness, then groundedness.
    verdict = _judge(CORRECT_SYSTEM, f"Question: {case['question']}\n\nReference answer: {expected}\n\n"
                                     f"System's answer: {answer}")
    if verdict and verdict.get("verdict") in ("correct", "partial", "incorrect"):
        out["verdict"] = verdict["verdict"]
        out["correctness"] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[verdict["verdict"]]
        out["correct_reason"] = verdict.get("reason", "")
    else:
        out["judge_error"] = True

    grounded = _judge(GROUNDED_SYSTEM, f"Evidence:\n{evidence[:60000]}\n\nAnswer to check:\n{answer}")
    if grounded and isinstance(grounded.get("grounded"), bool):
        out["grounded"] = grounded["grounded"]
        out["unsupported"] = grounded.get("unsupported", [])
    else:
        out["judge_error"] = True
    return out


# --- one document ------------------------------------------------------------

def run_document(run_id, spec, docs_dir, skip_summary=False):
    r = context.client()
    filename = spec["file"]
    ext = filename.rsplit(".", 1)[-1].lower()
    src = os.path.join(docs_dir, filename)
    doc_id = uuid.uuid4().hex
    out = {"doc": filename, "ext": ext, "size": spec.get("size"), "size_bytes": os.path.getsize(src),
           "questions": len(spec["cases"])}

    with llm.tagged(source="eval"):
        shutil.copy(src, pipeline.upload_path(doc_id, ext))
        db.add_document(doc_id, filename, ext, out["size_bytes"], is_eval=True)
        try:
            # Ingest
            started = time.perf_counter()
            try:
                pipeline.ingest(r, doc_id)
                out["ingest_ok"] = True
            except Exception as e:
                out.update(ingest_ok=False, ingest_error=f"{type(e).__name__}: {e}"[:300])
                db.save_eval_doc(run_id, filename, out)
                for case in spec["cases"]:
                    db.save_eval_result(run_id, case["id"], {
                        "case_id": case["id"], "doc": filename, "ext": ext, "size": spec.get("size"), "type": case["type"],
                        "question": case["question"], "error": "Document failed to ingest", "correctness": 0.0})
                return
            out["ingest_seconds"] = round(time.perf_counter() - started, 2)
            doc = db.get_document(doc_id)
            out.update(kind=doc["kind"], n_chunks=doc["n_chunks"])

            # Questions (before the summary exists, so answers rest on retrieval alone)
            for case in spec["cases"]:
                db.save_eval_result(run_id, case["id"], run_case(r, doc, case, spec.get("size")))
                print(f"  {case['id']} done", flush=True)

            # Summary
            if skip_summary:
                db.save_eval_doc(run_id, filename, out)
                return
            started = time.perf_counter()
            try:
                pipeline.summarize(r, doc_id)
                summ = db.get_summary(doc_id)
                out.update(summary=summ["summary"], summary_note=summ["coverage"],
                           summary_seconds=round(time.perf_counter() - started, 2),
                           summary_usage=_usage(f"summary:{doc_id}"))
                if doc["kind"] == "table":
                    source = doc["meta"]["schema"] + "\n\n" + parsers.profile_tables(
                        pipeline.table_db(doc_id), doc["meta"]["tables"])
                else:
                    source = "\n".join(text for _s, _l, text in db.iter_chunks(doc_id))
                points = spec.get("summary_points", [])
                numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1))
                whole = len(source) <= SUMMARY_SOURCE_CHARS
                graded = _judge(SUMMARY_SYSTEM, f"Source material:\n{source[:SUMMARY_SOURCE_CHARS]}\n\n"
                                                f"Key points:\n{numbered}\n\nSummary:\n{summ['summary']}")
                if graded and isinstance(graded.get("points"), list) and len(graded["points"]) == len(points):
                    out["points"] = [{"point": p, "covered": bool(c)} for p, c in zip(points, graded["points"])]
                    out["coverage"] = sum(bool(c) for c in graded["points"]) / len(points) if points else None
                    if whole:  # the judge can only spot unsupported statements if it saw the whole source
                        out["unsupported"] = graded.get("unsupported", [])
                else:
                    out["judge_error"] = True
            except Exception as e:
                out["summary_error"] = f"{type(e).__name__}: {e}"[:300]
            db.save_eval_doc(run_id, filename, out)
            print(f"  {filename} summary done", flush=True)
        finally:
            pipeline.delete(r, doc_id)  # test documents never appear in the app's document list


# --- aggregate ---------------------------------------------------------------

def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _pct(values, q):
    values = sorted(v for v in values if v is not None)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


def metrics(docs, results):
    """Headline numbers for one run. None means 'nothing to measure'."""
    get = lambda key, rows=results: [x.get(key) for x in rows]
    of_type = lambda t: [x for x in results if x.get("type") == t]
    tables = [x for x in results if x.get("kind") == "table"]
    q_cost, q_tokens = _mean(get("cost")), _mean([(x.get("input_tokens") or 0) + (x.get("output_tokens") or 0)
                                                  for x in results if "calls" in x])
    s_cost = _mean([d.get("summary_usage", {}).get("cost") for d in docs])
    return {
        "questions": len(results),
        "correctness": _mean(get("correctness")),
        "groundedness": _mean(get("grounded")),
        "retrieval_hit": _mean(get("retrieval_hit")),
        "no_match": _mean([not x["matched"] for x in results if x.get("matched") is not None]),
        "sql_accuracy": _mean(get("sql_match")),
        "sql_first_try": _mean([not x["sql_repaired"] for x in tables if x.get("sql")]),
        "sql_failed": _mean([x.get("sql_failed") for x in tables if "sql_failed" in x]),
        "refusal": _mean(get("correctness", of_type("unanswerable"))),
        "followup": _mean(get("correctness", of_type("followup"))),
        "p50_ms": _pct(get("elapsed_ms"), 0.5),
        "p95_ms": _pct(get("elapsed_ms"), 0.95),
        "calls_per_question": _mean(get("calls")),
        "tokens_per_question": q_tokens,
        "cost_per_question": q_cost,
        "summary_coverage": _mean([d.get("coverage") for d in docs]),
        "summary_unsupported": _mean([len(d["unsupported"]) for d in docs if "unsupported" in d]),
        "summary_seconds": _mean([d.get("summary_seconds") for d in docs]),
        "cost_per_summary": s_cost,
        # One summary plus ten questions, at this run's averages.
        "cost_per_document": (s_cost + 10 * q_cost) if s_cost is not None and q_cost is not None else None,
        "ingest_success": _mean([d.get("ingest_ok") for d in docs]),
        "judge_errors": sum(bool(x.get("judge_error")) for x in [*results, *docs]),
        "errors": sum(bool(x.get("error")) for x in results),
    }


def breakdown(results, key):
    """Correctness, groundedness and speed per value of `key` (e.g. file type)."""
    groups = {}
    for x in results:
        groups.setdefault(x.get(key) or "unknown", []).append(x)
    order = lambda name: (SIZE_ORDER.index(name) if name in SIZE_ORDER else len(SIZE_ORDER), name)
    return [{key: name, "questions": len(rows),
             "correctness": _mean([x.get("correctness") for x in rows]),
             "groundedness": _mean([x.get("grounded") for x in rows]),
             "retrieval_hit": _mean([x.get("retrieval_hit") for x in rows]),
             "sql_accuracy": _mean([x.get("sql_match") for x in rows]),
             "median_seconds": (_pct([x.get("elapsed_ms") for x in rows], 0.5) or 0) / 1000}
            for name, rows in sorted(groups.items(), key=lambda g: order(g[0]))]


def fmt_pct(v):
    return "n/a" if v is None else f"{100 * v:.0f}%"


# --- command line ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run the evaluation test set.")
    ap.add_argument("--label", default="", help="Name shown for this run on the Evaluation page")
    ap.add_argument("--cases", default=os.path.join(config.EVAL_DIR, "cases.json"))
    ap.add_argument("--only", default="", help="Run a single test document, by file name")
    ap.add_argument("--skip-summaries", action="store_true", help="Ask the questions but do not write or grade summaries")
    args = ap.parse_args()

    with open(args.cases) as f:
        specs = json.load(f)["documents"]
    if args.only:
        specs = [s for s in specs if s["file"] == args.only]
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(args.cases)), "docs")

    db.init()
    context.ensure_index(context.client())
    settings = {"model": config.LLM_MODEL, "fast_model": config.LLM_FAST_MODEL,
                "judge_model": config.EVAL_JUDGE_MODEL, "top_k": config.TOP_K,
                "chunk_chars": config.CHUNK_CHARS, "prices_set": config.PRICES_SET}
    run_id = db.start_eval_run(args.label or f"run {time.strftime('%Y-%m-%d %H:%M')}", settings)
    n_cases = sum(len(s["cases"]) for s in specs)
    print(f"Run {run_id}: {len(specs)} documents, {n_cases} questions. Settings: {settings}", flush=True)

    with ThreadPoolExecutor(max_workers=config.EVAL_CONCURRENCY) as pool:
        for fut in [pool.submit(run_document, run_id, s, docs_dir, args.skip_summaries) for s in specs]:
            fut.result()
    db.finish_eval_run(run_id)

    docs, results = db.get_eval_run(run_id)
    m = metrics(docs, results)
    print(f"\nRun {run_id} finished")
    for label, value in [
        ("Answer correctness", fmt_pct(m["correctness"])), ("Groundedness", fmt_pct(m["groundedness"])),
        ("Retrieval hit rate", fmt_pct(m["retrieval_hit"])), ("SQL result accuracy", fmt_pct(m["sql_accuracy"])),
        ("Correct refusals", fmt_pct(m["refusal"])), ("Follow-up correctness", fmt_pct(m["followup"])),
        ("Summary key-point coverage", fmt_pct(m["summary_coverage"])),
        ("Time to answer, median / 95th", f"{(m['p50_ms'] or 0) / 1000:.1f}s / {(m['p95_ms'] or 0) / 1000:.1f}s"),
        ("Tokens per question", f"{m['tokens_per_question'] or 0:,.0f}"),
        ("Cost per document", f"${m['cost_per_document']:.4f}" if config.PRICES_SET and m["cost_per_document"] is not None
         else "set prices in .env"),
        ("Judge errors", m["judge_errors"]),
    ]:
        print(f"  {label:<32}{value}")
    print("Open the Evaluation page in the app for the breakdown and individual cases.")


if __name__ == "__main__":
    main()
