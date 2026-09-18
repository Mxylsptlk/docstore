# docstore

Local PDF document-store built for **accurately extracting statistics from image-heavy
PDFs** and retrieving them later with citations. It ingests PDFs, uses a layout-aware
**vision** pass to pull the information you ask for (keeping every number tied to its
label), embeds the results locally, and stores them in a local vector database. When you
ask a question, it retrieves the most relevant passages and writes them to `ANSWER.md`
for an assistant (e.g. Claude) to read and answer.

## How it works

```
PDF → layout extract (blocks + bounding boxes)
    → per page: vision needed? ──no──> use the free text layer
                    │ yes
                    ▼
              render page → vision extract (curated text + statistics)
                          → verify each statistic (re-read its cropped region)
    → layout-aware chunking (a statistic stays glued to its label)
    → local embeddings (Ollama)
    → vector store (LanceDB)

ask("question") → embed → search → ANSWER.md (cited passages) → assistant answers
```

Two models, two jobs:
- **Embedding model** (local, via Ollama) turns text into vectors for semantic search.
  It runs at ingest (per chunk) and at query time (the question). It never writes prose.
- **Answering model** is the assistant (Claude) reading `ANSWER.md`. The vector store
  only *retrieves* passages; it does not compose answers.

### Statistics accuracy

Accurate numbers are the whole point, so:
- Numbers are read from the **rendered page image** (vision), avoiding scrambled or
  reordered PDF text layers.
- Each extracted statistic is **verified** by cropping its region and re-reading it; a
  mismatch is flagged `⚠ verification FAILED` and surfaced all the way into `ANSWER.md`.
- Stats without a reliable location are marked `⚠ unverified` rather than silently
  trusted. The extractor is instructed to transcribe numbers verbatim — never compute or
  round.

### Cost control (the vision gate)

Extraction defaults to Claude vision, but a per-page gate sends only pages that need it
(images, tables, or numeral-heavy pages with a weak text layer) to Claude. Clean text
pages use the free text layer. `force_vision: true` sends every page through vision.
The ingest summary reports `pages_vision` / `pages_textonly` so you can see the spend.

## Prerequisites

- **Python 3.11+**
- **Ollama** running locally with the embedding model pulled:
  ```bash
  ollama serve            # if not already running
  ollama pull nomic-embed-text
  ```
- **Anthropic API key** for the default Claude extraction backend, in the environment
  (never in config or committed):
  ```bash
  export ANTHROPIC_API_KEY=sk-...
  ```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Check your environment is ready:

```bash
python scripts/preflight.py
```

## Usage

### Python API (primary)

```python
import docstore
from docstore.config import load_config

cfg = load_config("config.yaml")

# Ingest a PDF, telling it what to extract.
summary = docstore.ingest(
    "reports/q3.pdf",
    instruction="Extract every statistic and its label, exactly as shown.",
    cfg=cfg,
)
print(summary)
# {'doc_id': 'q3-1a2b3c4d', 'page_count': 8, 'pages_vision': 3, 'pages_textonly': 5,
#  'chunk_count': 41, 'stat_count': 12, 'unverified_count': 1}

# Ask a question -> writes ANSWER.md with cited passages for the assistant to read.
path = docstore.ask("What was the on-time completion rate?", k=5, cfg=cfg)
print("Read", path)

# Or get results directly:
for r in docstore.retrieve("on-time completion rate", k=5, cfg=cfg):
    print(r.citation, r.is_stat, r.stat_verified, "-", r.text[:60])
```

### CLI (thin wrapper)

```bash
docstore ingest reports/q3.pdf --instruction "Extract every statistic and its label"
docstore ingest reports/q3.pdf --backend ollama      # local vision instead of Claude
docstore ask "What was the on-time completion rate?"  # writes ANSWER.md
```

### The ANSWER.md workflow

`ask()` retrieves the top-k passages and writes `ANSWER.md`: the question, each passage
under its `source.pdf:pN` citation heading (with a verification badge on statistics), and
an instruction block telling the assistant to cite passages, transcribe numbers exactly,
warn on unverified stats, and say when the context is insufficient. Point your assistant
at that file to get the grounded answer.

## Configuration

Non-secret settings live in `config.yaml` (see the file for all options):

| Key                  | Default                   | Meaning                                       |
|----------------------|---------------------------|-----------------------------------------------|
| `embed_model`        | `nomic-embed-text`        | Local Ollama embedding model                  |
| `extraction_backend` | `claude`                  | `claude` (vision) or `ollama` (local vision)  |
| `vision_model`       | `claude-sonnet-5`    | Vision model for extraction                   |
| `render_dpi`         | `220`                     | Page render resolution for vision             |
| `chunk_size`/`overlap`| `1200` / `150`           | Chunking parameters                           |
| `verify_stats`       | `true`                    | Re-read each statistic to verify it           |
| `force_vision`       | `false`                   | Send every page through vision (skip the gate)|

The Anthropic API key is read from `ANTHROPIC_API_KEY`, never from config.

## Testing

```bash
pytest -m "not live"     # fast unit tests, no Ollama/key needed (mocks external calls)
pytest -m live           # end-to-end; requires Ollama + ANTHROPIC_API_KEY
```

## Knowledge graph

Alongside the vector store, docstore builds a local **Kùzu** knowledge graph. The vector
store answers *"what text is similar to my question?"*; the graph answers *"what is
connected?"* — the same entity across many PDFs, multi-hop links, explainable paths.

Schema: `(Document)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity)`, plus a
`(Document)-[:REFERENCES {count}]->(Entity)` rollup. Entities unify on a normalized key,
so "Acme Corp." and "acme corp" become **one node** — that unification is the whole point.

Entity extraction is pluggable via `entity_backend`: **`regex`** (default, zero-dependency —
MONEY / DATE / ORG), `ollama` (a local instruct model, richer), or `claude` (best, costs
tokens). The richer backends fall back to regex on any failure, so ingest never breaks.

**Turning the graph off per document** — pass `graph=False` to exclude a single document
from the graph even when it's otherwise enabled:

```python
docstore.ingest("throwaway.pdf", cfg=cfg, graph=False)   # ingested for search, NOT graphed
```
```bash
docstore ingest throwaway.pdf --no-graph
```

Globally, set `build_graph: false` in `config.yaml` (then `graph=True` re-enables per call).

**Querying the graph:**

```python
docstore.documents_mentioning("Acme Corp")   # -> ['alpha-1a2b', 'beta-3c4d']  (cross-document)
docstore.related_documents("alpha-1a2b")      # -> docs sharing >=1 entity
docstore.entities_in_document("alpha-1a2b")   # -> entity keys this doc references
docstore.chunks_mentioning("Acme Corp")       # -> chunk ids (join back to the vector store)
```

The graph joins to the vector store on `chunk_id`, so a future GraphRAG step can seed from
semantic hits and walk the graph for connected context.

## Roadmap

- GraphRAG retrieval: seed from vector hits, expand along graph edges before answering.
- Richer relationship edges (co-occurrence weights, `REFERENCES` between documents).
- Dedicated table-parsing pass for dense numeric tables.

### Page routing

Every page is routed one of three ways during ingest (reported in the summary as
`pages_textonly` / `pages_vision` / `pages_skipped`):

- **Skip** — sparse decorative pages (cover, section divider) that carry no statistic are
  dropped so they don't pollute retrieval. A page is skipped *only* when there is nothing
  statistical to lose; any stat-like number keeps it.
- **Vision** — pages whose numbers would scramble when flattened to a text stream: tables,
  scattered stat callouts (e.g. `42%`, `$4.2M` in their own blocks), or number grids.
- **Text layer** — clean prose, including numbers inside sentences (which read fine in
  order), uses the free, accurate text layer.

`force_vision: true` bypasses skip and routes every page through vision.
