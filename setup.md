# SEVEN — Setup

## Requirements

- Python 3.11+ (developed on 3.14)
- llama.cpp `llama-server` binary — download from https://github.com/ggerganov/llama.cpp/releases

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
LLM_MODEL=your-model-name           # model name passed to llama-server
LLM_MODEL_PATH=C:/path/to/model.gguf
LLM_CLI_PATH=C:/path/to/llama-server.exe
MMPROJ_PATH=C:/path/to/mmproj.gguf  # set but unused unless multimodal

DEFAULT_PERSIST_DIR=./data/chroma
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

DB_USER=ignored      # legacy field, SQLite is used — any value works
DB_PASSWORD=ignored
DB_NAME=ignored

# BACKGROUND_LLM_MODEL_PATH=C:/path/to/small-model.gguf
# BACKGROUND_LLM_CLI_PATH=C:/path/to/llama-server.exe  # if different binary
```

---

## 4. Run

```bash
python -m LLMEngine.cli
```

---

## venv notes

- The `.venv` folder belongs in the project root and is already in `.gitignore` (or should be — never commit it)
- If you see `ModuleNotFoundError` after pulling new code, re-run `pip install -r requirements.txt` inside the activated venv
- If you ever need a clean slate: delete `.venv` and repeat steps 1–2
- Do **not** use `pip install` outside the venv — it installs into your system Python and will conflict with other projects