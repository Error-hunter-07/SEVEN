# SEVEN — Setup

## Requirements

- Python 3.13+ (developed on 3.13.5)
- llama.cpp `llama-server` binary — download from https://github.com/ggerganov/llama.cpp/releases
- A GGUF model file (main chat model + optional smaller background model)

---

## 1. Create a virtual environment

```bash
# From the project root (Seven/)
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

You should see `(.venv)` in your prompt. **Always activate the venv before running or installing anything.**

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

First run will also download the `all-MiniLM-L6-v2` embedding model (~90MB) into `.cache/st/`. Every subsequent startup loads it from that local cache with no network calls.

---

## 3. Configure `.env`

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required fields:

```env
# Main chat model (GPU-accelored)
LLM_MODEL=your-model-name
LLM_MODEL_PATH=C:/path/to/model.gguf
LLM_CLI_PATH=C:/path/to/llama-server.exe
MMPROJ_PATH=                         # set but unused unless multimodal

# ChromaDB persistence
DEFAULT_PERSIST_DIR=./data/chroma
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# SQLite (embedded — these fields exist for compatibility, any value works)
DB_USER=ignored
DB_PASSWORD=ignored
DB_NAME=ignored

# Background mini-LLM (optional — for semantic extraction + summarization)
# A smaller CPU-only model that runs alongside the main GPU model.
# If not set, background work falls back to the main model.
BACKGROUND_LLM_MODEL_PATH=C:/path/to/small-model.gguf
BACKGROUND_LLM_MODEL=your-background-model-name
# BACKGROUND_LLM_CLI_PATH=C:/path/to/llama-server.exe  # only if different binary
```

### Background model (optional)

The background model is a smaller, CPU-only LLM that handles:
- Semantic memory extraction (every turn)
- Episodic/chunk summarization (every 5 turns)
- Reflection generation (every 5 turns + session end)

This lets background work run **concurrently** with the main chat turn instead of queuing behind it. If not configured, all background work uses the main GPU model (slower, blocks chat turns).

Recommended: a 1-3B parameter GGUF quantized model with `gpu_layers=0` (CPU-only).

---

## 4. Run

```bash
python -m LLMEngine.cli
```

### CLI commands

| Command | Description |
|---------|-------------|
| `/stop` | Flush pending work, write episodic memory, exit |
| `/sleep` | Run KG sleep pipeline (up to 20 sessions) |
| `/sleep N` | Run KG sleep pipeline for up to N sessions |
| `/sleep status` | Show how many sessions are pending in the queue |

---

## 5. First run behavior

On the first startup:
1. The embedding model downloads from HuggingFace (~90MB) and caches locally
2. The SQLite database initializes at `data/seven_local.db`
3. ChromaDB initializes at `data/chroma/`
4. The main LLM server starts and blocks until ready
5. The background LLM server starts on a daemon thread (if configured)
6. Background workers (extraction, chunk summary, reflection) start

All subsequent startups skip the embedding download and load from cache.

---

## venv notes

- The `.venv` folder belongs in the project root and is already in `.gitignore` (or should be — never commit it)
- If you see `ModuleNotFoundError` after pulling new code, re-run `pip install -r requirements.txt` inside the activated venv
- If you ever need a clean slate: delete `.venv` and repeat steps 1–2
- Do **not** use `pip install` outside the venv — it installs into your system Python and will conflict with other projects
