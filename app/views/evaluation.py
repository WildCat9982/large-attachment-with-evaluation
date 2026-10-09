"""Evaluation page: test-set results per run, and usage from real sessions."""
import pandas as pd
import streamlit as st

import config
import evaluate
import llm
import persistence as db

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")
db.init()

pct = evaluate.fmt_pct
ANSWER_PURPOSES = {"keywords", "answer", "sql", "sql_repair", "result"}


def secs(ms):
    return "n/a" if ms is None else f"{ms / 1000:.1f}s"


def tile(col, label, value, shown, base=None, unit="pts", lower_is_better=False, help=None):
    """One headline number, with the change against the comparison run if there is one."""
    delta = None
    if value is not None and base is not None:
        diff = (value - base) * (100 if unit == "pts" else 1)
        text = f"{diff:+.0f} pts" if unit == "pts" else f"{diff:+.3f}{unit}" if "USD" in unit else f"{diff:+.1f}{unit}"
        delta = text if any(ch in "123456789" for ch in text) else None  # hide changes that round to zero
    col.metric(label, shown, delta=delta, delta_color="inverse" if lower_is_better else "normal", help=help)


def needs_attention(x):
    return bool(x.get("error") or (x.get("correctness") is not None and x["correctness"] < 1)
                or x.get("grounded") is False or x.get("retrieval_hit") is False or x.get("sql_match") is False)


def mark(x):
    if x.get("error") or x.get("correctness") == 0:
        return "❌"
    return "⚠️" if needs_attention(x) else "✅"


def yes_no(v):
    return "n/a" if v is None else ("Yes" if v else "No")


def show_tests():
    runs = db.list_eval_runs()
    if not runs:
        st.info("No evaluation runs yet. Start one from the project folder with:")
        st.code('docker compose run --rm eval --label "baseline"', language="bash")
        return

    name = lambda run: f"#{run['id']} · {run['label']} · {run['started_at'][:16]}"
    by_id = {run["id"]: run for run in runs}
    left, right = st.columns(2)
    run_id = left.selectbox("Run", list(by_id), format_func=lambda i: name(by_id[i]))
    others = [i for i in by_id if i != run_id]
    base_id = right.selectbox("Compare with", [None, *others],
                              format_func=lambda i: "No comparison" if i is None else name(by_id[i]))
    run = by_id[run_id]
    docs, results = db.get_eval_run(run_id)
    m = evaluate.metrics(docs, results)
    b = evaluate.metrics(*db.get_eval_run(base_id)) if base_id else {}

    cfg = run["settings"]
    st.caption(f"Answers: {cfg.get('model')} · bulk steps: {cfg.get('fast_model')} · judge: {cfg.get('judge_model')} · "
               f"passages per question: {cfg.get('top_k')} · passage size: {cfg.get('chunk_chars')} characters · "
               f"{m['questions']} questions over {len(docs)} documents")
    if run["status"] != "done":
        st.warning("This run has not finished, so the numbers below are partial.")
    if m["judge_errors"]:
        st.warning(f"The judge model gave no usable verdict for {m['judge_errors']} item(s); they are left out of the scores.")

    # Headline numbers
    c = st.columns(5)
    tile(c[0], "Answer correctness", m["correctness"], pct(m["correctness"]), b.get("correctness"),
         help="Share of answers the judge model marked as matching the reference answer. Partly right counts as half.")
    tile(c[1], "Groundedness", m["groundedness"], pct(m["groundedness"]), b.get("groundedness"),
         help="Share of answers in which every claim is supported by the passages or query result the system used.")
    tile(c[2], "Retrieval hit rate", m["retrieval_hit"], pct(m["retrieval_hit"]), b.get("retrieval_hit"),
         help="Prose documents only: share of questions where the passage holding the answer was among those retrieved.")
    p95, base_p95 = m["p95_ms"], b.get("p95_ms")
    tile(c[3], "Time to answer (95th percentile)", p95 and p95 / 1000, secs(p95), base_p95 and base_p95 / 1000,
         unit="s", lower_is_better=True, help="95% of questions were answered at least this fast.")
    if cfg.get("prices_set") and m["cost_per_document"] is not None:
        tile(c[4], "Cost per document", m["cost_per_document"], f"${m['cost_per_document']:.3f}",
             b.get("cost_per_document"), unit=" USD", lower_is_better=True,
             help="One summary plus ten questions at this run's average token use, at the prices set in .env.")
    else:
        tok, base_tok = m["tokens_per_question"], b.get("tokens_per_question")
        tile(c[4], "Tokens per question", tok and tok / 1000, "n/a" if tok is None else f"{tok:,.0f}",
             base_tok and base_tok / 1000, unit="k", lower_is_better=True,
             help="Average input plus output tokens per question. Set token prices in .env to see cost instead.")

    c = st.columns(5)
    tile(c[0], "Query result accuracy", m["sql_accuracy"], pct(m["sql_accuracy"]), b.get("sql_accuracy"),
         help="Data files only: share of questions where the generated query returned the same values as the reference query.")
    tile(c[1], "Correct refusals", m["refusal"], pct(m["refusal"]), b.get("refusal"),
         help="Questions the document cannot answer: share where the system said so and did not guess.")
    tile(c[2], "Follow-up correctness", m["followup"], pct(m["followup"]), b.get("followup"),
         help="Questions that only make sense with the previous turn, such as 'who approved it?'.")
    tile(c[3], "Summary key-point coverage", m["summary_coverage"], pct(m["summary_coverage"]), b.get("summary_coverage"),
         help="Share of each document's listed key points that its summary conveys.")
    tile(c[4], "Ingestion success", m["ingest_success"], pct(m["ingest_success"]), b.get("ingest_success"),
         help="Share of test documents that were parsed and loaded without error.")

    # Breakdowns
    def table(key, title):
        rows = [{title: r[key], "Questions": r["questions"], "Correctness": pct(r["correctness"]),
                 "Groundedness": pct(r["groundedness"]), "Retrieval hit": pct(r["retrieval_hit"]),
                 "Query accuracy": pct(r["sql_accuracy"]), "Median time": f"{r['median_seconds']:.1f}s"}
                for r in evaluate.breakdown(results, key)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("By file type")
        table("ext", "File type")
    with right:
        st.subheader("By question type")
        table("type", "Question type")
    if any(x.get("size") for x in results):
        st.subheader("By file size")
        table("size", "Size tier")

    st.subheader("Documents")
    st.dataframe(pd.DataFrame([{
        "Document": d["doc"], "Size (MB)": round(d["size_bytes"] / 1024 / 1024, 2),
        "Ingested": yes_no(d.get("ingest_ok")), "Ingest time": f"{d['ingest_seconds']:.2f}s" if "ingest_seconds" in d else "n/a",
        "Summary coverage": pct(d.get("coverage")),
        "Unsupported statements": len(d["unsupported"]) if "unsupported" in d else "n/a",
        "Summary time": f"{d['summary_seconds']:.1f}s" if "summary_seconds" in d else "n/a",
    } for d in sorted(docs, key=lambda d: d["size_bytes"])]), hide_index=True, width="stretch")

    with st.expander("Summaries and their key points"):
        for d in docs:
            st.markdown(f"**{d['doc']}**")
            if d.get("summary_error") or d.get("ingest_error"):
                st.error(d.get("summary_error") or d.get("ingest_error"))
                continue
            for p in d.get("points", []):
                st.markdown(f"{'✅' if p['covered'] else '❌'} {p['point']}")
            for u in d.get("unsupported", []):
                st.markdown(f"⚠️ Not supported by the source: {u}")
            st.caption(d.get("summary_note", ""))
            st.markdown(d.get("summary", "").replace("$", "\\$"))
            st.divider()

    # Individual questions
    st.subheader("Individual questions")
    flagged = [x for x in results if needs_attention(x)]
    only = st.toggle(f"Show only the {len(flagged)} that need attention", value=False)
    shown = flagged if only else results
    if not shown:
        st.write("Nothing to show.")
        return
    cases = {x["case_id"]: x for x in shown}
    case_id = st.selectbox("Question", list(cases),
                           format_func=lambda i: f"{mark(cases[i])} {i} · {cases[i]['question']}")
    x = cases[case_id]

    if x.get("previous"):
        st.caption(f"Previous turn: “{x['previous']['question']}” → “{x['previous']['answer']}”")
    left, right = st.columns(2)
    with left:
        st.markdown("**Reference answer**")
        st.write(x.get("expected", "n/a").replace("$", "\\$"))
    with right:
        st.markdown("**System's answer**")
        st.write((x.get("answer") or x.get("error") or "n/a").replace("$", "\\$"))

    c = st.columns(4)
    c[0].metric("Correctness", (x.get("verdict") or "not scored").capitalize(), help=x.get("correct_reason"))
    c[1].metric("Grounded", yes_no(x.get("grounded")))
    if x.get("kind") == "table":
        c[2].metric("Query result matches reference", yes_no(x.get("sql_match")))
        c[3].metric("Time", secs(x.get("elapsed_ms")))
    else:
        c[2].metric("Right passage retrieved", yes_no(x.get("retrieval_hit")))
        c[3].metric("Time", secs(x.get("elapsed_ms")))
    if x.get("correct_reason"):
        st.caption(f"Judge on correctness: {x['correct_reason']}")
    for claim in x.get("unsupported") or []:
        st.caption(f"Judge found this unsupported: {claim}")

    if x.get("sql"):
        with st.expander("Query used", expanded=x.get("sql_match") is False):
            st.code(x["sql"], language="sql")
            if x.get("sql_repaired"):
                st.caption("The first query failed and was corrected on the second attempt.")
            if x.get("rows"):
                st.dataframe(pd.DataFrame(x["rows"], columns=x["columns"]), hide_index=True)
    if x.get("sources"):
        with st.expander("Passages retrieved", expanded=x.get("retrieval_hit") is False):
            st.caption("Search terms: " + ", ".join(x.get("terms", [])))
            if x.get("matched") is False:
                st.caption("No passages matched; the opening of the document was used.")
            for s in x["sources"]:
                st.markdown(f"**[{s['n']}]** {s['loc']}")
                st.text(s["text"])


def show_live():
    answers = db.live_answers()
    calls = db.llm_calls(source="live")
    if not answers and not calls:
        st.info("No usage recorded yet. Ask some questions on the Documents page.")
        return
    times = [a["elapsed_ms"] for a in answers if a["elapsed_ms"] is not None]
    rated = [a["feedback"] for a in answers if a["feedback"] is not None]
    q_calls = [c for c in calls if c["purpose"] in ANSWER_PURPOSES]
    q_refs = {c["ref"] for c in q_calls}
    q_tokens = sum((c["input_tokens"] or 0) + (c["output_tokens"] or 0) for c in q_calls)

    c = st.columns(5)
    c[0].metric("Questions answered", len(answers))
    c[1].metric("Time to answer (median)", secs(evaluate._pct(times, 0.5)))
    c[2].metric("Time to answer (95th percentile)", secs(evaluate._pct(times, 0.95)))
    c[3].metric("Tokens per question", f"{q_tokens / len(q_refs):,.0f}" if q_refs else "n/a")
    c[4].metric("Thumbs up", f"{100 * sum(rated) / len(rated):.0f}%" if rated else "n/a",
                help=f"{len(rated)} of {len(answers)} answers rated")

    st.subheader("Model calls by step")
    groups = {}
    for call in calls:
        groups.setdefault((call["purpose"] or "other", call["model"]), []).append(call)
    rows = []
    for (purpose, model), items in sorted(groups.items()):
        tin = sum(i["input_tokens"] or 0 for i in items)
        tout = sum(i["output_tokens"] or 0 for i in items)
        row = {"Step": purpose, "Model": model, "Calls": len(items),
               "Average time": f"{sum(i['duration_ms'] or 0 for i in items) / len(items) / 1000:.1f}s",
               "Input tokens": tin, "Output tokens": tout}
        if config.PRICES_SET:
            row["Cost (USD)"] = round(llm.cost(model, tin, tout), 4)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if not config.PRICES_SET:
        st.caption("Set token prices in .env to see cost.")


st.title("Evaluation")
tests, live = st.tabs(["Test results", "Live usage"])
with tests:
    show_tests()
with live:
    show_live()
