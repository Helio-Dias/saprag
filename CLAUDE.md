# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

This repository is a knowledge and tooling workspace for database operations, currently focused on:

- **MySQL 8 / GCP Cloud SQL** — diagnostic playbook for temporary tablespace issues (`temporary_tbs.txt`), covering root-cause analysis, query identification scripts, remediation, monitoring, and alerting.
- **PostgreSQL (local Docker)** — schema `cortador` with tables `documentos` and `chunks`, secondary store.
- **GCP Firestore** (`helio-test-db`) — primary store for documents, chunks, and embeddings.

> Note: Firestore collection/field names and the PostgreSQL schema are kept in Portuguese because they map to data already persisted in the live stores.

## PostgreSQL access

The local Postgres instance runs in Docker via Colima.

| Parameter | Value |
|-----------|-------|
| Host | localhost |
| Port | 5432 |
| User | postgres |
| Password | postgres |
| Database | postgres |

```bash
# Start the runtime (if not running)
colima start

# Start the container (if stopped)
docker start postgres

# Open a psql shell
docker exec postgres psql -U postgres

# Connect to a specific database
docker exec postgres psql -U postgres -d <db_name>
```

## Database schema

Schema: `cortador`

```sql
-- Documents
CREATE TABLE cortador.documentos (
    id SERIAL PRIMARY KEY,
    descricao TEXT,
    caminho_arquivo TEXT
);

-- Text chunks (FK to documentos, composite PK)
CREATE TABLE cortador.chunks (
    id_documento INTEGER NOT NULL REFERENCES cortador.documentos(id),
    sequencia_chunk INTEGER NOT NULL,
    texto VARCHAR(1000),
    PRIMARY KEY (id_documento, sequencia_chunk)
);
```

## GCP Firestore

| Parameter | Value |
|-----------|-------|
| Database | helio-test-db |
| Project | ems-data-platform-dev |
| Region | europe-west1 |
| Type | FIRESTORE_NATIVE |

Authentication: Application Default Credentials (ADC)
```bash
gcloud auth application-default login
```

### Firestore collection structure

```
documentos/{id}
  ├── id, descricao, caminho_arquivo, total_chunks
  ├── data_onboard: timestamp UTC   # added by process_new_files.py (absent on migrated docs)
  └── chunks/{sequencia_chunk}
        ├── id_documento, sequencia_chunk, texto
        └── embedding: [768 floats]   # null until embed_chunks.py runs
```

- The `data_onboard` field is used by `search.py` to resolve contradictions between chunks (most recent = source of truth).
- The document `id` is sequential (`max(id) + 1`), computed by `get_next_id()` in `process_new_files.py`.

### Result cache (collection `cache_resultado`)

```
cache_resultado/{auto_id}
  ├── query_text: str               # original question
  ├── query_embedding: [768 floats] # question vector
  ├── resposta: str                  # final pipeline answer
  ├── data_cache: timestamp UTC      # when it was cached
  ├── data_ultimo_uso: timestamp UTC # last time it was used
  ├── uso_exclusivo: int             # perfect/near-perfect hits (direct return)
  └── uso_auxiliar: int              # auxiliary hits (used as extra context)
```

`search.py` checks the cache before the full pipeline. Configurable thresholds:

| Level | Threshold | Behavior |
|-------|-----------|----------|
| Perfect | >= 0.98 | Returns cached answer directly |
| Near-perfect | >= 0.92 | Returns cached answer directly |
| Auxiliary | >= 0.85 | Uses the answer as additional context during synthesis |
| Miss | < 0.85 | Full pipeline |

## saprag.py — central TUI (main entry point)

`saprag.py` is the single entry point that dispatches to every module from one
`simple-term-menu`. Prefer running it over invoking the scripts individually.

```bash
python3 saprag.py
```

| Menu option | Dispatches to |
|-------------|---------------|
| Insert document (interactive chunking) | `cutter.main()` |
| Process new files (arquivos_novos/) | `process_new_files.process()` |
| Compute embeddings | `embed_chunks.embed_chunks()` |
| Search (RAG pipeline) | prompts for a question → `search.enhanced_search(query)` |
| Evolve cached answers | `evolution.main()` |
| Migrate PostgreSQL -> Firestore | `migrate_to_firestore.migrate()` |
| Exit | quits |

- **Lazy imports** — each module is imported only when its option is selected, so
  a missing optional dependency only affects that
  action, not the whole menu.
- **Error isolation** — each action runs in a try/except; an error or Ctrl-C
  returns to the menu instead of crashing the program.
- Reuses the existing entry points without duplicating logic; `cutter` and
  `evolution` keep their own sub-prompts.
- Requires a real TTY (same as `cutter.py`); it will not run with piped stdin.

## cutter.py program

Python TUI to split text documents into chunks and store them in PostgreSQL and Firestore.

### Dependencies

```bash
pip3 install -r requirements.txt
```

- `simple-term-menu` — interactive terminal menu
- `psycopg2-binary` — PostgreSQL connection
- `google-cloud-firestore` — Firestore connection
- `google-cloud-aiplatform` — Vertex AI embeddings
- `pymupdf` — text extraction from PDFs
- **Claude Code CLI** (`claude`) — `search.py` and `evolution.py` call Claude Haiku (`claude-haiku-4-5`) for query decomposition and RAG synthesis by shelling out to `claude -p --model claude-haiku-4-5`. The CLI must be installed and authenticated, and on `PATH`. Each LLM call is a separate CLI invocation.

### Running

```bash
python3 cutter.py
```

### Features

| Menu | Description |
|------|-------------|
| Choose file | Lists files in the current directory (`.txt`, `.doc`, `.docx`, `.md`) for selection |
| Choose description | Descriptive text for the document |
| Choose chunk size | Chunk size in characters (default: 528) |
| Choose overlap | Overlap between chunks (default: 128) |
| Run chunking | Reads the file, chunks it, and inserts into PostgreSQL and Firestore |

### Chunking algorithm

- Splits the text into pieces of `chunk_size` characters with `overlap` of overlap
- If the cut point falls in the middle of a word, advances to the next boundary (whitespace)
- Atomic transaction in PostgreSQL; batch write to Firestore

## Helper scripts

| Script | Description |
|--------|-------------|
| `process_new_files.py` | Batch-processes `.doc`, `.docx`, `.txt`, `.md`, `.pdf` from the `arquivos_novos/` folder — extracts text, chunks it, validates the absence of HTML, and writes to Firestore with deduplication by `caminho_arquivo`. Writes batches of 100 chunks. Auto-generates `descricao` from the file name (URL-decoded). |
| `migrate_to_firestore.py` | Migrates existing documents and chunks from PostgreSQL to Firestore. **Note:** does not populate `data_onboard` — migrated documents are left without that field. |
| `embed_chunks.py` | Computes embeddings (text-embedding-004, 768d) for chunks without an embedding and writes them to Firestore |
| `search.py` | Multi-query semantic search with two-round RAG refinement |
| `evolution.py` | Offline job that re-answers the most recent cached questions by reading the stored documents **whole** (no RAG / no vector search), aiming to beat the cached RAG answer. Resumable across runs; writes to the `evolution` collection. |

### search.py — Multi-Query RAG Pipeline

`search.py` implements an advanced search pipeline with query decomposition and automatic refinement:

1. **Decomposition** — Claude Haiku (via the Claude Code CLI) generates 2-4 sub-queries while preserving the central context of the original question
2. **Multi-query search** — batch embeddings (original query + sub-queries), Firestore loaded once in memory, UNION + dedup by best score per chunk
3. **Round 1 synthesis** — the LLM answers and generates follow-up questions (marker `<<<FOLLOW_UP>>>`) for gaps or incomplete passages
4. **Second round (automatic)** — if follow-ups exist, searches for new chunks and performs a final consolidated synthesis

#### Tuning parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `TOP_K_PER_QUERY` | 5 | Chunks returned per sub-query |
| `TOP_K_MERGED` | 15 | Maximum chunks in round 1 (after dedup) |
| `TOP_K_SECOND_ROUND` | 10 | Maximum new chunks in round 2 |
| `MAX_SUB_QUERIES` | 4 | Limit of generated sub-queries |

#### Contradiction resolution

When chunks contradict each other, the pipeline uses `data_onboard` (ingestion date) as the criterion: the most recent chunk is the source of truth.

Characteristics:
- **Automatic language** — the answer is always in the same language as the user's question
- **Graceful degradation** — a failure at any stage falls back to the previous stage's answer
- **Chunks loaded once** — avoids multiple scans in Firestore
- **Backward compatible** — legacy functions `search(query, top_k=10)` and `synthesize_with_claude(query, resultados)` remain available

```bash
# Search with the full pipeline
python3 search.py "your question here"
```

### evolution.py — Offline answer improvement (no RAG)

`evolution.py` tries to produce answers that are *better* than what the RAG
pipeline already cached. It assumes the cached answer may have lost context by
retrieving only a few generic chunks, so instead of vector search it reads the
**entire corpus** document-by-document with Claude Haiku and re-answers.

Interactive inputs:

- **Minute budget (total)** — total time to spend this run.
- **Number of cached questions to evolve** — the N most recent `cache_resultado`
  entries (newest first by `data_cache`). The budget is split evenly:
  `per_question_seconds = (minutes * 60) / N`. Example: 10 min, 5 questions →
  2 min each.

Pipeline per question (map-reduce, **not** RAG):

1. **Map** — for each stored document, join its chunks into full text (batched by
   `MAP_CHUNK_BATCH = 40` chunks per model call) and extract only what is
   relevant to the question. Segments with nothing relevant return `NONE`.
2. **Reduce** — once *all* documents are read, synthesize a final answer from the
   collected evidence, instructed to surpass the previous (cached) answer and
   stay grounded in the evidence only.

Key behaviors:

- **No RAG / no embeddings** — reads documents whole; `search.py`'s vector search
  is bypassed entirely.
- **Soft deadline** — the per-question time slice is checked *before* reading each
  document. A model call already running is always allowed to finish, so wall
  time can slightly exceed the slice.
- **Resumable** — reading all documents is slow (each map call is a CLI round-trip
  to Haiku) and rarely completes in one slice. Progress (next document index +
  evidence gathered so far) is persisted per question in the `evolution_progress`
  collection. Re-run to continue; the reduce step and the write to `evolution`
  happen only after the whole corpus has been read, then the progress record is
  cleared.

Stored output (collection `evolution`):

```
evolution/{auto_id}
  ├── cache_doc_id: str          # source cache_resultado document id
  ├── query_text: str            # the question
  ├── resposta_anterior: str     # the cached (RAG) answer
  ├── resposta_evoluida: str     # the improved answer
  └── data_evolucao: timestamp UTC
```

Resume state (collection `evolution_progress`, transient — deleted on completion):

```
evolution_progress/{cache_doc_id}
  ├── next_index: int            # next document to read (into the id-sorted corpus)
  ├── evidence_parts: [str]      # map extractions gathered so far
  └── data_atualizacao: timestamp UTC
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MAP_CHUNK_BATCH` | 40 | Chunks joined per map-step model call |

```bash
# Improve the 3 most recent cached answers over a 6-minute budget (2 min each)
python3 evolution.py
# then enter: 6  and  3
# Re-run with the same N to continue questions that didn't finish.
```

> **Note:** each map step is a separate `claude -p` call to Haiku, so a large
> corpus may not be fully read in one short run — this is exactly why progress is
> persisted and the job is resumable.

### Supported formats in process_new_files.py

| Format | Handling |
|--------|----------|
| `.doc` / `.docx` | Multipart MIME (Confluence export) — `email.parser` + `quopri` + `HTMLParser` |
| `.txt` | MIME with fallback to plain text |
| `.md` | Plain text — only `_normalize()` |
| `.pdf` | PyMuPDF (`fitz`) — page-by-page extraction |

### Embeddings

- Model: `text-embedding-004` (Vertex AI, region `europe-west1`)
- Dimensions: 768
- Batch size: 20 chunks per call (API limit of ~20k tokens)
- Idempotent: skips chunks that already have a computed embedding

```bash
# After inserting new documents, compute pending embeddings
python3 embed_chunks.py
```

## End-to-end flow for new files

The simplest path is the central TUI, which exposes every step as a menu option:

```bash
python3 saprag.py
```

Equivalent manual invocation:

```bash
# 1. Copy files into arquivos_novos/
# 2. Process (chunk + Firestore)
python3 process_new_files.py
# 3. Compute embeddings
python3 embed_chunks.py
# 4. Search
python3 search.py "your question here"
```
