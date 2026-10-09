"""Documents page: upload a document, summarise it, ask follow-up questions."""
import uuid

import pandas as pd
import streamlit as st

import config
import context
import persistence as db
import pipeline

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")


@st.cache_resource
def setup():
    db.init()
    r = context.client()
    context.ensure_group(r)
    return r


r = setup()

STATUS_ICON = {"queued": "⏳", "processing": "⏳", "ready": "✅", "failed": "⚠️"}


def md(text):
    """Render model text; escape $ so amounts are not read as LaTeX."""
    st.markdown(text.replace("$", "\\$"))


def show_progress(doc_id, fallback):
    p = context.get_progress(r, doc_id)
    if p and p["total"]:
        st.progress(min(p["done"] / p["total"], 1.0), text=f"{p['stage']} ({p['done']}/{p['total']})")
    elif p and p["done"]:
        st.info(f"{p['stage']}… {p['done']:,} passages so far")
    else:
        st.info(f"{p['stage'] if p else fallback}…")


@st.fragment(run_every=2)
def watch_ingest(doc_id):
    doc = db.get_document(doc_id)
    if doc is None or doc["status"] not in ("queued", "processing"):
        st.rerun()
    show_progress(doc_id, "Waiting for the worker")


@st.fragment(run_every=2)
def watch_summary(doc_id):
    summ = db.get_summary(doc_id)
    if summ is None or summ["status"] not in ("queued", "running"):
        st.rerun()
    show_progress(doc_id, "Waiting for the worker")


def save_feedback(message_id):
    db.set_feedback(message_id, st.session_state.get(f"fb_{message_id}"))


def show_answer_details(meta):
    if meta.get("sql"):
        with st.expander("Query used"):
            st.code(meta["sql"], language="sql")
            if meta.get("rows"):
                st.dataframe(pd.DataFrame(meta["rows"], columns=meta["columns"]), hide_index=True)
                if meta.get("truncated") or len(meta["rows"]) == 50:
                    st.caption("Showing the first rows only.")
    if meta.get("sources"):
        with st.expander("Passages used"):
            if not meta.get("matched", True):
                st.caption("No passages matched the question; the opening of the document was used.")
            for s in meta["sources"]:
                st.markdown(f"**[{s['n']}]** {s['loc']}")
                st.text(s["text"])


# --- sidebar: upload and pick a document ------------------------------------

with st.sidebar:
    st.header("Documents")
    with st.form("upload", clear_on_submit=True):
        file = st.file_uploader(f"Upload a file (up to {config.MAX_UPLOAD_MB} MB)", type=config.ALLOWED_EXT)
        if st.form_submit_button("Upload", type="primary") and file is not None:
            if file.size > config.MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"That file is over {config.MAX_UPLOAD_MB} MB.")
            else:
                doc_id = uuid.uuid4().hex
                ext = file.name.rsplit(".", 1)[-1].lower()
                with open(pipeline.upload_path(doc_id, ext), "wb") as out:
                    out.write(file.getbuffer())
                db.add_document(doc_id, file.name, ext, file.size)
                context.enqueue(r, "ingest", doc_id)
                st.session_state["doc_id"] = doc_id

    docs = {d["id"]: d for d in db.list_documents()}
    if docs:
        if st.session_state.get("doc_id") not in docs:
            st.session_state["doc_id"] = next(iter(docs))
        st.radio(
            "Your documents", list(docs), key="doc_id",
            format_func=lambda i: f"{STATUS_ICON.get(docs[i]['status'], '')} {docs[i]['filename']}",
        )
        if st.button("Delete this document"):
            pipeline.delete(r, st.session_state["doc_id"])
            del st.session_state["doc_id"]
            st.rerun()

# --- main panel --------------------------------------------------------------

doc = docs.get(st.session_state.get("doc_id")) if docs else None
if doc is None:
    st.title("Document Q&A")
    st.write("Upload a document in the sidebar to summarise it and ask questions about it.")
    st.stop()

st.title(doc["filename"])
doc_id = doc["id"]

if doc["status"] in ("queued", "processing"):
    watch_ingest(doc_id)
    st.stop()
if doc["status"] == "failed":
    st.error(f"This file could not be processed. {doc['error']}")
    st.stop()

size_mb = doc["size_bytes"] / 1024 / 1024
if doc["kind"] == "table":
    tables = doc["meta"]["tables"]
    st.caption(f"{size_mb:.1f} MB · data file · {len(tables)} table{'s' if len(tables) != 1 else ''} · "
               "questions are answered by querying the data")
else:
    st.caption(f"{size_mb:.1f} MB · {doc['n_chars']:,} characters in {doc['n_chunks']:,} passages · "
               "questions are answered from the best-matching passages")

st.subheader("Summary")
summ = db.get_summary(doc_id)
if summ and summ["status"] in ("queued", "running"):
    watch_summary(doc_id)
elif summ and summ["status"] == "done":
    md(summ["summary"])
    st.caption(summ["coverage"])
else:
    if summ and summ["status"] == "failed":
        st.error(f"The summary failed. {summ['error']}")
    if st.button("Summarize", type="primary"):
        db.set_summary(doc_id, "queued")
        context.enqueue(r, "summarize", doc_id)
        st.rerun()

st.subheader("Questions")
for m in db.get_messages(doc_id):
    with st.chat_message(m["role"]):
        md(m["content"])
        show_answer_details(m["meta"])
        if m["role"] == "assistant":
            st.feedback("thumbs", key=f"fb_{m['id']}", default=m["feedback"],
                        on_change=save_feedback, args=(m["id"],))

if question := st.chat_input("Ask a question about this document"):
    with st.chat_message("user"):
        md(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Working on it…"):
                pipeline.answer(r, doc_id, question)
        except Exception as e:
            st.error(f"That question could not be answered. {type(e).__name__}: {e}")
            st.stop()
    st.rerun()  # redraw from history, which adds the feedback buttons
