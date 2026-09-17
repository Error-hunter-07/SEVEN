# SEVEN

SEVEN is a locally-run AI assistant with a six-layer cognitive memory architecture that persists knowledge across sessions. It runs entirely on your hardware — no cloud APIs, no data leaving your machine. Built on llama.cpp for inference and ChromaDB + SQLite for storage.

[Diagram placeholder]

---

## What it does

SEVEN is not a chat wrapper. It is a full agent loop with persistent memory that learns about you over time. Every conversation is automatically distilled into durable facts, session summaries, and structured relationships — all without manual intervention.

The assistant remembers who you are, what you've discussed, what decisions were made, and why. It proactively uses this context on future turns without being asked.

---

## How it works

SEVEN runs two LLM processes simultaneously:

- **Main model** (GPU-accelerated) — handles your conversation directly
- **Background model** (CPU-only, optional) — handles memory extraction, summarization, and reflection in parallel so chat turns are never blocked

Every turn, the system:
1. Retrieves relevant memories (semantic, episodic, reflections)
2. Injects them into the prompt as context
3. Generates a response using the main model
4. Extracts facts from the conversation in the background
5. Summarizes conversation chunks every 5 turns
6. Produces behavioral self-corrections every 5 turns

At session end, the full conversation is summarized into an episodic memory, and all pending work is flushed before shutdown.

---

## Memory architecture

Six distinct memory layers, each with a specific purpose and storage backend:

| Layer | Backend | Scope | Purpose |
|-------|---------|-------|---------|
| **Scratchpad** | In-memory | Per-turn | Planning state, tool outputs, reasoning notes |
| **Working Memory** | SQLite | Session (60-day TTL) | Facts needed again later in the same session |
| **Episodic Memory** | ChromaDB | Cross-session | Session summaries — what happened, what was decided |
| **Semantic Memory** | ChromaDB | Cross-session | Atomized facts about you — identity, preferences, goals |
| **Knowledge Graph** | SQLite (7 tables) | Cross-session | Structured relationships between concepts |
| **Active Sessions** | SQLite | Current session | Crash-durability marker + live scratch space |

The Knowledge Graph is populated by a sleep pipeline (`/sleep` command) that extracts entities and relationships from session histories using background LLM calls.

Full documentation: [MemoryArchitecture.md](MemoryArchitecture.md)

---

## Features

### Memory system
- **Automatic fact extraction** — every conversation turn is analyzed and 1-3 durable facts are stored without you asking
- **Deduplication** — near-identical facts are detected via cosine similarity and merged, not duplicated
- **Negation detection** — "User likes Python" and "User dislikes Python" are kept separate
- **Decay and pruning** — memories lose importance over time and are pruned when the store grows too large (500 max, pruned to 400)
- **Episodic decay-by-summarization** — old session summaries are merged into higher-level summaries at 6, 12, and 24 months

### Knowledge Graph
- **Entity extraction** — background LLM extracts named entities from session histories
- **Entity resolution** — cascade matching (exact name → alias → keyword → create new) prevents duplicate nodes
- **Relationship proposals** — background LLM proposes edges between entities based on conversation context
- **Deterministic validation** — all proposed graph operations are validated before writing (no blind LLM writes)
- **Query service** — 4 retrieval methods (exact name, alias, keyword, semantic) fused via Reciprocal Rank Fusion
- **Audit trail** — every graph mutation is logged in an append-only table

### Self-improvement
- **Reflection worker** — produces behavioral directives every 5 turns (e.g. "be more concise about technical topics")
- **5-criteria scoring** — each directive is scored on scope, specificity, confidence, actionability, and novelty
- **Reflection consolidation** — at session end, an LLM pass judges which reflections to keep, delete, or promote to semantic memory
- **Cross-session persistence** — reflection directives survive across sessions and influence every future prompt

### Crash recovery
- **Durable session markers** — active sessions are persisted to SQLite before every turn
- **Rolling chunk summaries** — every 5 turns, the last 5 turns are summarized and stored
- **Full conversation backup** — raw messages are saved every turn as a last resort
- **Automatic recovery** — on startup, any session still marked "in_progress" is recovered from whatever data survived

### Prompt engineering
- **Dynamic context injection** — semantic memory, episodic context, and reflections are assembled per-turn based on the current query
- **Deterministic episodic trigger** — regex patterns ("last time", "we discussed") automatically inject episodic context before the LLM sees the message
- **Token budget management** — 12,000 token context limit with graceful degradation (episodic → reflections → semantic → drop all)
- **KV cache optimization** — system prompt is set once per session for llama-server prompt caching

### Dual-model architecture
- **Main model** — GPU-accelerated, handles conversation, tool calls, and knowledge graph pipeline
- **Background model** — CPU-only, handles memory extraction, summarization, and reflection generation
- **Independent locks** — both models run genuinely concurrently (no shared lock or GPU slot)
- **Graceful fallback** — if background model is not configured, all work falls back to the main model

---

## Tech stack

- **Inference**: llama.cpp (`llama-server` with `--parallel 1`)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store**: ChromaDB (semantic + episodic memory)
- **Structured storage**: SQLite (working memory, active sessions, knowledge graph)
- **Language**: Python 3.13+

---

## Getting started

See [setup.md](setup.md) for installation and configuration.

```bash
python -m LLMEngine.cli
```

---

## Project structure

```
Seven/
├── LLMEngine/              Core LLM interaction (client, workers, history)
├── MemoryManagement/       All memory layers (short-term, working, episodic, semantic)
├── KnowledgeGraph/         Sleep pipeline + query service
├── SessionManager/         Session lifecycle + crash recovery
├── PromptBuilder/          Per-turn context assembly
├── Database/               SQLite + ChromaDB clients
├── Tools/                  LLM-facing tool bridges
├── ToolCalling/            Tool parsing, execution, registration
├── Runtime/                llama-server process management
├── GlobalHelpers/          Config, logging, token counting
├── data/                   SQLite DB + ChromaDB persistence
└── logs/                   Application + session logs
```
