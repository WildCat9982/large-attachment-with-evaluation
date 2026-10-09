# Document Q&A POC

Upload a file of up to 50 MB, get a summary, and ask follow-up questions.
Streamlit for the UI, Redis 8.0 for working context, SQLite for persistence.

## Run it

```bash
cp .env.example .env      # then add your API key
docker compose up --build
```

Open http://localhost:8501. Supported files: PDF, DOCX, TXT, HTML, CSV, JSON, XLSX.

## How it works

Three containers: `app` (Streamlit), `worker` (background jobs), `redis`.

An upload is saved to the shared volume and queued on a Redis stream. The
worker then takes one of two paths, depending on what the file contains:

| | Prose: PDF, DOCX, TXT, HTML, free-form JSON | Data: CSV, XLSX, JSON records |
|---|---|---|
| Ingest | Extract text, split into ~1,200-character passages, store in SQLite, index in Redis | Load every row into a per-document SQLite database |
| Summarise | Read the text in ~40,000-character sections, take notes on each, merge the notes | Compute statistics for every column, describe them |
| Answer | Find the best-matching passages with Redis full-text search (BM25), answer from those | Have the model write a SELECT, run it read-only, answer from the result |

So the model never receives the whole file. A question costs two or three
model calls regardless of file size.

### What lives where

| Redis (context, disposable) | SQLite (persistence, on the `data` volume) |
|---|---|
| Passages and their search index | Documents and their status |
| Last few conversation turns, 24h TTL | All passages |
| Job queue and live progress | Summaries, full chat history and feedback |
| | Usage log (every model call) and evaluation runs |

Redis runs without persistence on purpose. If it is restarted, passages are
re-indexed from SQLite on the next question, and recent turns are reloaded.

### What is stored in Redis

Six kinds of keys. `{doc}` is the document's ID.

| Key | Type | Contents | Lifetime |
|---|---|---|---|
| `chunk:{doc}:{n}` | Hash | One passage of a prose document: `doc_id`, `seq` (its position), `loc` (such as `p.12` for PDFs), `text` | Until the document is deleted |
| `idx:chunks` | Search index | Full-text index over every `chunk:` hash: `text` for keyword search, `doc_id` to restrict a search to one document, `seq` as a number | Permanent |
| `indexed:{doc}` | String | The passage count, used as a marker that this document's passages are loaded | Until the document is deleted |
| `chat:{doc}` | List | The last 6 question-and-answer turns, each a JSON object with `role` and `content` | Expires 24 hours after the last question |
| `progress:{doc}` | Hash | Live job status for the UI: `stage`, `done`, `total` | Expires after 1 hour |
| `jobs` | Stream | One entry per background job: `type` (`ingest` or `summarize`) and `doc_id`, read by the `workers` consumer group | Entries are never trimmed |

- Data files (CSV, XLSX, record-style JSON) are queried in SQLite, so they have
  no `chunk:` keys; they only ever have `chat:`, `progress:` and job entries.
- The `chunk:` hashes are a second copy of the document text and account for
  nearly all of Redis' memory use.
- Finished jobs are acknowledged but not removed from the `jobs` stream. That is
  harmless for a POC; for longer use, cap the stream with `MAXLEN`.
- The uploaded file, summaries, full chat history and answer details (sources
  and SQL) are in SQLite only.

To inspect a running instance:

```bash
docker compose exec redis redis-cli --scan --pattern '*' | head -20
docker compose exec redis redis-cli FT.INFO idx:chunks
```

## Evaluation

The app has a second page, **Evaluation**, that shows how well the system
performs on a fixed test set, plus usage figures from real sessions.

```bash
docker compose run --rm eval --label "baseline"
```

This ingests each test document, asks every test question, writes a summary,
scores the results and stores them as a run. With the sample set (7 documents,
57 questions) that is roughly 250 model calls. Open the Evaluation page to see it.

### What is measured

| Metric | How it is scored |
|---|---|
| Answer correctness | A judge model compares each answer with the reference answer: correct, partly correct (counts half) or incorrect |
| Groundedness | The judge checks that every claim in the answer is supported by the passages or query result the system used |
| Retrieval hit rate | Exact check, prose only: the passage containing the answer was among those retrieved |
| Query result accuracy | Exact check, data files only: the generated query returned the same values as a reference query |
| Correct refusals | Correctness on questions the document cannot answer |
| Follow-up correctness | Correctness on questions that depend on the previous turn |
| Summary key-point coverage | The judge checks each document's listed key points against its summary, and lists unsupported statements |
| Time to answer | Median and 95th percentile, measured around the whole answer |
| Tokens and cost | From the usage log; cost appears once token prices are set in `.env` |
| Ingestion success and time | Per test document |

"Cost per document" means one summary plus ten questions at the run's averages.

### Comparing runs

Every run records the models and retrieval settings it used. To see the effect
of a change, run again with the setting overridden and pick both runs on the
page:

```bash
docker compose run --rm -e TOP_K=2 eval --label "2 passages per question"
```

### The test set

`eval/cases.json` lists the documents in `eval/docs/`, and for each one its
questions and the key points a summary should cover. A question looks like this:

```json
{"id": "handbook-02", "type": "answerable",
 "question": "If I don't use all my holiday, how much of it can I move into next year?",
 "expected": "Up to 5 days can be carried over, and they must be taken by 31 March.",
 "evidence": "carry over up to 5 days"}
```

- `type` is `answerable`, `unanswerable` (no `expected` needed) or `followup`
  (add `"previous": {"question": ..., "answer": ...}` for the earlier turn).
- `evidence`, for prose, is a short phrase copied from the document where the
  answer sits; it drives the retrieval hit rate.
- `gold_sql`, for data files, is a reference query; its result is the expected
  answer and drives query result accuracy. Tables are named `data` for CSV and
  JSON, and after the sheet names for XLSX; column names are lower-cased with
  underscores.

The folder is mounted into the container, so edits take effect on the next run
without rebuilding.

The shipped set is a synthetic sample about a fictional freight company. Most
of its prose documents are short, which makes retrieval easy; only the incident
review (about 90 KB) and the handbook really test it. Scores on your own
documents, with questions written by people who know them, are the ones that
count.

### Size-tiered test files

To test how the system behaves as files grow, generate a second test set with
every file type at four sizes:

```bash
docker compose run --rm testdata                       # writes ./eval/sizes, about 400 MB
docker compose run --rm -e EVAL_CONCURRENCY=1 eval \
    --cases eval/sizes/cases.json --label "size tiers"
```

| Tier | Target size | Prose files (TXT, HTML) | Rich files (DOCX, PDF) | Data files (CSV, XLSX, JSON) |
|---|---|---|---|---|
| small | 60 KB | about 45 sections | 30 sections | about 300 to 1,100 rows |
| medium | 1 MB | about 800 sections | 150 sections plus images | about 5,000 to 19,000 rows |
| large | 10 MB | about 8,000 sections | 800 sections plus images | about 50,000 to 190,000 rows |
| xlarge | 45 MB | about 35,000 sections | 2,500 sections plus images | about 220,000 to 860,000 rows |

That is 28 files and 168 questions. The Evaluation page adds a "By file size"
table when a run has size tiers.

- Prose files are a synthetic depot operations manual with one section per
  depot. Each question targets one fact in one section (at the start, middle
  and end of the file, one by depot name only, one follow-up, one unanswerable),
  so the expected answer and the phrase that must be retrieved are exact.
- DOCX and PDF files get their size from embedded images, as real ones do; the
  images are random noise and are ignored by the app.
- Data files are scored by reference queries, so nothing is hard-coded.
- Files and answers are seeded and identical on every run. Use
  `--sizes small,medium` to generate only some tiers.

A full run is roughly 1,800 model calls and about 10 million input tokens, most
of it summary notes on the large prose files. Start with the small and medium
tiers, and add `--skip-summaries` to the eval command to test ingestion,
retrieval and answers only (roughly 800 calls). Use `EVAL_CONCURRENCY=1` so
that large files are not processed side by side, and watch memory with
`docker stats` while it runs.

Ingestion measured while building this, on one CPU with Redis 7 and no search
index, so treat it as a rough guide only:

| File at 45 MB | Ingest time | Peak memory |
|---|---|---|
| TXT (45,000 passages) | 3 s | 240 MB |
| HTML (42,000 passages) | 10 s | 590 MB |
| DOCX (3,100 passages) | 1 s | 185 MB |
| PDF (3,000 passages) | 6 s | 220 MB |
| CSV (705,000 rows) | 3 s | 170 MB |
| JSON (220,000 records) | 7 s | 360 MB |
| XLSX (860,000 rows) | 149 s | 200 MB |

**Adding your own documents.** Copy a file into `eval/sizes/docs/`, describe it
in `eval/sizes/extra_cases.json`, and rerun the generator; it is merged into
`cases.json`. For PDFs a question can give `"page": 532` in place of an
`evidence` phrase, meaning the page that holds the answer. The shipped
`extra_cases.json` 20% direct retrieval, 25% contextual reasoning, 20% comparison/scenario, 15% multi-hop, 10% conversational follow-up, and 10% unanswerable/adversarial. That will tell you far more about whether the document-Q&A architecture actually works than 20–30 trivia questions. questions for `AWSCertifiedGenerativeAIDeveloper.pdf`,
which is not included; copy your own copy in. `eval/sizes/docs/` is excluded
from git and from the Docker image, so test documents are not published with
the code.

### Reading the scores

- Correctness, groundedness and summary coverage come from a model's judgement.
  Set `EVAL_JUDGE_MODEL` to a different model from the one answering, and check
  a sample of verdicts by hand on the page (each shows the judge's reason).
- Questions are asked before the summary is written, so answers rest on
  retrieval alone. Each question starts from a clean conversation.
- The **Live usage** tab shows real sessions only: answer times, tokens per
  question, thumbs up/down, and model calls by step.

## Settings

Set in `.env`; all are optional except the API key.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, or `openai` for any OpenAI-compatible endpoint |
| `LLM_MODEL` | `claude-sonnet-5-5` | Answers and final summaries |
| `LLM_FAST_MODEL` | `claude-haiku-4-5-20251001` | Section notes and search keywords |
| `SUMMARY_MAX_WINDOWS` | `150` | Most sections read per summary; `0` reads everything |
| `SUMMARY_CONCURRENCY` | `4` | Parallel model calls while summarising |
| `TOP_K` | `6` | Passages retrieved per question |
| `CHUNK_CHARS` | `1200` | Passage size |
| `LLM_PRICE_IN`, `LLM_PRICE_OUT` | blank | USD per million tokens for `LLM_MODEL` |
| `LLM_FAST_PRICE_IN`, `LLM_FAST_PRICE_OUT` | blank | USD per million tokens for `LLM_FAST_MODEL` |
| `EVAL_JUDGE_MODEL` | same as `LLM_MODEL` | Model that scores answers in evaluation runs |

`SUMMARY_MAX_WINDOWS` is a cost guard. A 50 MB plain-text file is about 1,300
sections, so a full read is about 1,300 model calls. Above the cap, an evenly
spaced sample is read and the UI says what share of the text was covered.

## Limits of this POC

- Retrieval is keyword-based. The model expands each question into search
  terms, which helps, but a question worded very differently from the text
  can miss. Adding embeddings and Redis vector search is the natural next step.
- Scanned PDFs have no text layer and are rejected; there is no OCR.
- Images inside documents are ignored.
- One conversation per document, no user accounts.
- A job that is running when the worker is killed stays marked as in
  progress; delete the document and upload it again.

## Layout

```
app/ui.py                Entry point and page navigation
app/views/documents.py   Documents page: upload, summary, chat, feedback
app/views/evaluation.py  Evaluation page
app/worker.py            Job loop
app/pipeline.py          Ingest, summarise, answer (start here)
app/evaluate.py          Evaluation runner and scoring
app/make_testdata.py     Generator for size-tiered test files
app/parsers.py           File readers, chunking, guarded SQL
app/context.py           Redis
app/persistence.py       SQLite
app/llm.py               Model call and usage logging (swap providers here)
app/config.py            Settings
eval/cases.json          Test questions
eval/docs/               Test documents
eval/sizes/              Size-tiered test set (generated) and extra_cases.json
```


