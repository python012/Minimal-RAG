# Copilot Instructions — Minimal RAG

A minimal Chinese-language RAG (Retrieval-Augmented Generation) Q&A system built on the **Qwen AI Platform (千问AI平台)** via the OpenAI-compatible API, **ChromaDB** for vector storage, and **Streamlit** for the web UI.

## Build, test, lint

There is **no test suite, no linter, and no CI workflow** in this repo. Verification is manual and CLI-driven.

```bash
# Always work inside the venv — `python` does not exist on the system PATH
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the web app (http://localhost:8501)
streamlit run app.py

# Ingest documents
python data_loader.py --input ./data/ --pattern "*.json"  # JSON recipes
python data_loader.py --input ./data/sample.txt           # single text file
python data_loader.py --input ./data/ --clear              # wipe, then re-ingest

# Inspect the vector DB (fast, no API calls, no API key needed)
python data_loader.py --stats
python query_chunk.py "煤球冰淇淋"                         # exact metadata.name match
python query_chunk.py --like "煤球" --max 10              # substring match, case-insensitive
```

Convenience wrappers exist for both platforms: `./clear.sh`, `./load-json.sh`, `./stats.sh` (macOS/Linux) and `clear.bat`, `load-json.bat`, `stats.bat` (Windows). They just call `data_loader.py` and end with a `read -p`/`pause` prompt.

To verify a change end-to-end, the fastest loop is: `python data_loader.py --stats` → `python query_chunk.py --like "<keyword>"` → restart the app and ask the same question.

## Architecture

Three CLI/app entry points share one ChromaDB store. Data flows one direction only:

```
data/recipes/*.json, *.txt
        │
        │ data_loader.py  (embed → chunk/flatten → collection.add)
        ▼
ChromaDB collection "knowledge_base"  (./vector_db)
        ▲
        │ rag_engine.py  (search + boost → generate_answer)
        │
app.py (Streamlit UI)
```

- **[data_loader.py](/Users/Reed/ai-dev/Minimal-RAG/data_loader.py)** — `DataLoader` class. Chunks text, builds semantic text from JSON recipes, calls the embedding API, and `collection.add()`s each record. Also owns `--stats` / `--clear`.
- **[rag_engine.py](/Users/Reed/ai-dev/Minimal-RAG/rag_engine.py)** — `RAGEngine` class. Owns retrieval (embed → Chroma query → exact-name boost → relevance filter) and generation (prompt build → Qwen chat completion, streaming or not).
- **[app.py](/Users/Reed/ai-dev/Minimal-RAG/app.py)** — Streamlit UI. Calls `RAGEngine.search()` then `RAGEngine.generate_answer(stream=True)` **directly**; it does *not* use `RAGEngine.query()`. The `query()` method is a non-streaming convenience wrapper that is currently unused by the app — see "Two parallel code paths" below.
- **[query_chunk.py](/Users/Reed/ai-dev/Minimal-RAG/query_chunk.py)** — read-only CLI for inspecting what actually landed in ChromaDB. Its exact/fuzzy matching is on `metadata.name`, **not** on vector similarity.
- **[aliyun-qwen-api-demo/](/Users/Reed/ai-dev/Minimal-RAG/aliyun-qwen-api-demo/)** — minimal standalone snippets showing the raw embedding and chat calls. Useful reference for the exact request shapes.

## Key conventions

**The ChromaDB `ids` field is always returned, regardless of `include`.** `collection.query(include=["documents", "metadatas", "distances"])` still returns `ids`. This bit the codebase before: when `_boost_exact_matches()` re-sorts results, `ids` must be carried through the same reordering and the same `[:n_results]` truncation as `documents`/`metadatas`/`distances`, or the four lists desync and `zip()` silently pairs wrong metadata with wrong documents. Any code that re-slices or reorders search results **must** move all four lists together.

**All four result lists are index-aligned.** The inner lists of `ids`/`documents`/`metadatas`/`distances` from `collection.query()` correspond 1:1 by index. Preserve that invariant in any transformation.

**Two parallel code paths for relevance filtering.** `RAGEngine.query()` and `app.py` each independently compute `max_relevance = max(1 - d for d in distances)` and apply the `min_relevance` threshold. They must agree — if you change the threshold semantics or the relevance score in one, change it in the other, or the app and the library API will disagree about what "relevant" means.

**Distance is cosine distance; relevance is `1 - distance`.** Lower distance = more relevant. ChromaDB's `include` whitelist does not apply to `ids`.

**Record IDs are path-derived, and re-ingesting the same file is a no-op.** Text chunks get `f"{file_path}_{i}"`; JSON recipes get `f"{file_path}"` (no chunk suffix — one record per recipe, not chunked). ChromaDB's `add()` silently ignores duplicate IDs rather than upserting, so re-running the loader over an unchanged file will **not** update the stored document or embedding. To pick up changed content, run `--clear` first or delete the affected records.

**JSON recipes are flattened into a single semantic text blob**, not chunked. The blob is assembled in a fixed order: `名称` → `描述` → `类型` → `关键词` → `配料` → `步骤`. The embedded `document` is that blob; the individual fields are also stored in metadata (`name`, `type`, `tags`, `instructions`, `image_path`, `recipe_source`).

**Chinese field names are primary; English are fallbacks.** The loader reads `名称`/`name`, `描述`/`description`, `类型`/`type`, `关键词`/`keywords`/`tags`, `配料`/`ingredients`, `步骤`/`directions`/`instructions`, `图片`/`image`, `来源`/`source`. Add new fields following the same "Chinese key first, English fallback" order.

**`metadata.name` drives the exact-match boost and the `query_chunk.py` lookups.** It is only set for JSON recipes — text files get `file_name`/`file_path` instead, so the name-boost in `_boost_exact_matches()` silently does nothing for them. The boost scales the distance down (`dist * (1 - boost_factor)`), it does not add a separate score column, so relevance scores downstream remain comparable to `1 - distance`.

**Images are matched by convention, not by path in the JSON.** `_match_image_for_recipe()` looks for `data/images/<recipe_basename>.{jpg,jpeg,png}` (and a legacy `data2/images/`), falling back to the JSON's explicit `图片`/`image` filename. The resolved relative path lands in `metadata.image_path` and is rendered by the app.

**Text chunking is 1000 chars with 200 overlap**, breaking preferentially at `.` or `\n` when that break point falls past the halfway mark of the chunk.

**Config is read from the environment at import time**, not per call. `load_dotenv()` runs at module import in all three entry-point files, and the `ALIYUN_*` constants are module-level. Changing `.env` requires restarting the process. `.env` is gitignored; `.env.example` documents every supported key.

**`ALIYUN_API_TIMEOUT` is parsed but never applied.** Both `rag_engine.py` and `data_loader.py` read it into a module constant, but the `OpenAI(...)` client is constructed without a `timeout=` argument. If you touch client construction, wire the constant up rather than adding a second timeout source.

**An `OpenAI` client is constructed per call** in `get_embedding()` and `generate_answer()`. Fine at this scale, but it means API errors surface at call time, not construction time.

**Comments, docstrings, and console output are in Chinese**, and console output uses emoji-prefixed status lines (`✓`, `❌`, `📦`, `📊`) printed to stdout — not the `logging` module. Keep this style when editing.

**`source` is the metadata field for provenance filtering** (`file` by default; `git`/`jira`/`confluence` are selectable in the UI). It is a top-level metadata key, so `where={"source": ...}` filters on it.

**Switching the embedding model or dimension invalidates the existing vector DB.** `ALIYUN_EMBED_DIM` is sent on every request (default 1024), and `text-embedding-v4` supports 64–2048. Changing the model or dimension requires re-ingesting all documents with `--clear`, otherwise old and new vectors coexist in incompatible spaces.
