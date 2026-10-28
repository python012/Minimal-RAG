# Copilot Instructions — Minimal RAG

A minimal Chinese-language RAG (Retrieval-Augmented Generation) Q&A system built on the **Qwen AI Platform (千问AI平台)** via the OpenAI-compatible API, **ChromaDB** for vector storage, and **Streamlit** for the web UI.

## Build, test, lint

There is **no test suite, no linter, and no CI workflow** in this repo. Verification is manual and CLI-driven.

```bash
# Always work inside the venv — `python` does not exist on the system PATH (only `python3`)
source venv/bin/activate
pip install -r requirements.txt

# Start the web app (http://localhost:8501)
streamlit run app.py

# Ingest the bundled JSON recipes (one record per recipe)
python data_loader.py --input ./data/recipes/ --pattern "*.json"

# Ingest text/markdown — this is the CLI's default pattern set, which excludes JSON
python data_loader.py --input ./data/ --pattern "*.txt" "*.md"

# Wipe, then re-ingest
python data_loader.py --input ./data/recipes/ --pattern "*.json" --clear

# Inspect the vector DB (fast, no API calls, no API key needed)
python data_loader.py --stats
python query_chunk.py "煤球冰淇淋"                   # exact metadata.name match
python query_chunk.py --like "煤球" --max 10         # substring match, case-insensitive
```

`--pattern` is `nargs="+"`, so pass several globs in one invocation. Directory loads search recursively and dispatch per file extension (`.json` → recipe flattener, everything else → text chunker). **A single-file `--input` always goes through the text chunker — even for `.json`** — so pointing `--input` at one recipe file stores raw JSON text instead of parsed recipe metadata.

Platform wrappers live in [scripts/](../scripts) (`clear.sh`, `load-json.sh`, `stats.sh` plus `.bat` equivalents). They assume the venv is already active and end with a `read -p`/`pause` prompt, so they block — don't call them from automation. They also predate the `scripts/` move in [README.md](../README.md), which still shows them at the repo root.

The fastest end-to-end check after a change is: `python data_loader.py --stats` → `python query_chunk.py --like "<keyword>"` → restart the app and ask the same question.

If startup fails with a `pyarrow requires NumPy 2.0` error, follow the upgrade note in the README troubleshooting section (`streamlit>=1.40`, `numpy>=2.0`, `opentelemetry-proto>=1.45`); `requirements.txt` already encodes the resolution.

## Architecture

Three CLI/app entry points share one ChromaDB store. Data flows one direction only:

```
data/**/*.txt, *.md, *.json
        │
        │ data_loader.py  (embed → chunk/flatten → collection.add)
        ▼
ChromaDB collection "knowledge_base"  (./vector_db)
        ▲
        │ rag_engine.py  (search + boost → generate_answer)
        │
app.py (Streamlit UI)
```

- **[data_loader.py](../data_loader.py)** — `DataLoader` class. Chunks text, flattens JSON recipes into semantic text, calls the embedding API, and `collection.add()`s each record. Also owns `--stats` / `--clear`.
- **[rag_engine.py](../rag_engine.py)** — `RAGEngine` class. Owns retrieval (embed → Chroma query → exact-name boost → relevance filter) and generation (prompt build → Qwen chat completion, streaming or not).
- **[app.py](../app.py)** — Streamlit UI. Caches one `RAGEngine` via `@st.cache_resource`; the sidebar's "重新加载引擎" button clears that cache, which is how the app picks up newly ingested documents (the collection handle is created once per process). It calls `RAGEngine.search()` then `RAGEngine.generate_answer(stream=True)` **directly**; it does *not* use `RAGEngine.query()`. The `query()` method is a non-streaming convenience wrapper that is currently unused by the app — see "Distances are L2-squared" below.
- **[query_chunk.py](../query_chunk.py)** — read-only CLI for inspecting what actually landed in ChromaDB. Its exact/fuzzy matching is on `metadata.name`, **not** on vector similarity.
- **[aliyun-qwen-api-demo/](../aliyun-qwen-api-demo/)** — minimal standalone snippets showing the raw embedding and chat calls. Useful reference for the exact request shapes.

## Key conventions

**The ChromaDB `ids` field is always returned, regardless of `include`.** `collection.query(include=["documents", "metadatas", "distances"])` still returns `ids`, and `collection.get()` rejects `"ids"` in `include` with `ValueError` while returning ids anyway. This bit the codebase before: when `_boost_exact_matches()` re-sorts results, `ids` must be carried through the same reordering and the same `[:n_results]` truncation as `documents`/`metadatas`/`distances`, or the four lists desync and `zip()` silently pairs wrong metadata with wrong documents. Any code that re-slices or reorders search results **must** move all four lists together.

**All four result lists are index-aligned and doubly nested.** The inner lists of `ids`/`documents`/`metadatas`/`distances` from `collection.query()` correspond 1:1 by index, and the outer list is the query batch — this engine always sends one embedding, so consume `results["documents"][0]`, not `results["documents"]`. Preserve both invariants in any transformation.

**Distances are L2-squared, not cosine.** The collection is created without `hnsw:space`, so ChromaDB's default `l2` applies (`collection.configuration_json["hnsw"]["space"] == "l2"`; returned distances match `sum((q - d)**2)` exactly). Distances are unbounded — they can exceed 1 and shrink after boosting — so there is no 0–1 range. `RAGEngine.query()` and `app.py` both nonetheless compute `max_relevance = max(1 - dist)` and compare it against `min_relevance`; that score is a display heuristic, can go negative, and the `SearchResults` docstring explicitly warns against using it as a threshold. Keep the two computations in agreement: changing threshold semantics in one without the other makes the library API and the UI disagree about what "relevant" means. The UI slider defaults to `0.35`.

**Retrieval expands candidates, then truncates.** `search()` over-fetches `search_n = min(max(n_results * 5, 30), 100)` records, applies the name boost and re-sorts, and only then truncates all four lists to `n_results`. Boosting therefore affects which records survive the cut, not just their order.

**Types describe the happy path, and there is exactly one cast.** `SearchResults` / `RecipeMeta` in `rag_engine.py` model the nested Chroma shapes; `cast(SearchResults, raw)` at the single `collection.query()` call is the only loose boundary, so everything downstream is precisely typed. Keep `ids` non-`Optional` — empty results return empty inner lists, never missing keys — and let the defensive length-mismatch branch keep its `# type: ignore[list-item]` instead of widening the type. There is no mypy/ruff dependency or config in the repo (`.mypy_cache/` is a leftover from an ad-hoc editor run), so annotations serve Pylance and were tightened in the `refactor(typing)` commit.

**Record IDs are path-derived, and re-ingesting the same file is a no-op.** Text chunks get `f"{file_path}_{i}"` (with `i` the chunk index, so skipped empty chunks leave gaps) and metadata `source`/`file_path`/`file_name`/`chunk_id`/`total_chunks`/`chunk_size`; JSON recipes get `f"{file_path}"` (no chunk suffix — one record per recipe, not chunked). Because the id embeds the path exactly as passed on the CLI, ingesting through a different spelling (`./data/...` vs `data/...`) creates a second copy rather than updating. ChromaDB's `add()` silently ignores duplicate IDs rather than upserting, so re-running the loader over an unchanged file will **not** update the stored document or embedding. To pick up changed content, run `--clear` first or delete the affected records.

**JSON recipes are flattened into a single semantic text blob**, not chunked. The blob is assembled in a fixed order: `名称` → `描述` → `类型` → `关键词` → `配料` → `步骤`. The embedded `document` is that blob; the individual fields are also stored in metadata (`name`, `type`, `tags`, `instructions`, `image_path`, `recipe_source`).

**Chinese field names are primary; English are fallbacks.** The loader reads `名称`/`name`, `描述`/`description`, `类型`/`type`, `关键词`/`keywords`/`tags`, `配料`/`ingredients`, `步骤`/`directions`/`instructions`, `图片`/`image`, `来源`/`source`. Add new fields following the same "Chinese key first, English fallback" order.

**`metadata.name` drives the exact-match boost and the `query_chunk.py` lookups.** It is only set for JSON recipes — text files get `file_name`/`file_path` instead, so the name-boost in `_boost_exact_matches()` silently does nothing for them, and `query_chunk.py "..."` can never find text chunks.

**The boost is tiered and score-based.** `_normalize_text()` lowercases, replaces everything except CJK/ASCII letters, digits and whitespace with a space, then collapses whitespace — so `"芒果-慕斯！"` and `"芒果 慕斯"` match. `_boost_exact_matches()` scores each record: normalized equality → 1000, mutual substring → 500, otherwise `int(300 * overlap / union)` over word sets. The factor is `min(0.99, score / 1000)` and the new distance is `dist * (1 - factor)`. It rewrites `distances` in place (no separate score column) and re-sorts by the new distance; records without `name` keep their original distance.

**Images are matched by convention, not by path in the JSON.** `_match_image_for_recipe()` first tries the JSON's explicit `图片`/`image` filename, then `data/images/<recipe_basename>.{jpg,jpeg,png}` (plus a legacy `data2/images/`), and returns the first hit. The resolved path is stored relative as `metadata.image_path` (e.g. `data/images/煤球冰淇淋.jpg`) and rendered by the app, so paths are cwd-relative — the app and loader must run from the repo root.

**Text chunking is 1000 chars with 200 overlap**, breaking preferentially at `.` or `\n` when that break point falls past the halfway mark of the chunk.

**Config is read from the environment at import time**, not per call. `load_dotenv()` runs at module import in every module (`app.py`, `rag_engine.py`, `data_loader.py`, `query_chunk.py`), and the `ALIYUN_*` constants are module-level. Changing `.env` requires restarting the process. `.env` is gitignored; `.env.example` documents every supported key.

**`ALIYUN_API_TIMEOUT` is parsed but never applied.** Both `rag_engine.py` and `data_loader.py` read it into a module constant, but the `OpenAI(...)` client is constructed without a `timeout=` argument. If you touch client construction, wire the constant up rather than adding a second timeout source.

**An `OpenAI` client is constructed per call** in `get_embedding()` and `generate_answer()`. Fine at this scale, but it means API errors surface at call time, not construction time.

**`chat_history` is accepted but ignored.** `generate_answer()` and `query()` take the parameter, but retrieval and prompting are single-turn — the app keeps its own `st.session_state.chat_history` purely for rendering. Wiring history into the prompt means threading it through `generate_answer()` explicitly; passing it does nothing today.

**Comments, docstrings, and console output are in Chinese.** `rag_engine.py` uses the `logging` module (`logger.exception` / `logger.debug`); `data_loader.py`, `query_chunk.py`, and `app.py` print emoji-prefixed status lines to stdout (`✓`, `❌`, `📦`, `📊`) and have no logger. Match the file you are editing rather than unifying them.

**Error handling is deliberately asymmetric.** `search()` and `generate_answer()` log and re-raise; `query()` catches everything and returns `{"answer": "错误：...", "sources": [], "raw_results": None}` instead of raising. Both loaders swallow per-file exceptions and return `0`, so an embedding API failure mid-ingest looks like "0 chunks loaded" rather than a crash — check the printed `❌` line, not just the exit code.

**The two `get_stats()` methods return different keys.** `DataLoader.get_stats()` → `total_chunks`; `RAGEngine.get_stats()` → `total_documents`. Both report `collection.count()`.

**`query_chunk.py --max` is a local filter, not a query limit.** The fuzzy path pulls a `max(max_results * 20, 200)`-row metadata snapshot into memory and filters on `name`, so matches beyond that snapshot are invisible on larger databases.

**`source` is the metadata field for provenance filtering** (`file` by default; `git`/`jira`/`confluence` are selectable in the UI). It is a top-level metadata key, so `where={"source": ...}` filters on it.

**Switching the embedding model or dimension invalidates the existing vector DB.** `ALIYUN_EMBED_DIM` is sent on every request (default 1024), and `text-embedding-v4` supports 64–2048. Changing the model or dimension requires re-ingesting all documents with `--clear`, otherwise old and new vectors coexist in incompatible spaces.
