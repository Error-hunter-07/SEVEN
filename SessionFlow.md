# SEVEN — Session Flow

A session in SEVEN spans from the moment the user starts the application to the moment they exit. Every session has a unique ID, persists across crashes, and produces durable memory artifacts that survive into future sessions.

This document traces the complete lifecycle: startup, session start, each turn, background worker activity, clean shutdown, and crash recovery.

[Session lifecycle diagram placeholder]

---

## 1. Application startup

When `python -m LLMEngine.cli` runs, `LLMEngine/llm_client.py` executes at module level:

```
configure_logging()
  ↓
bootstrap_all_models()
  ├─ main_process.start_blocking()
  │    → ProcessManager(role="main") — starts llama-server on port 8081
  │    → wait_until_ready() — blocks until HTTP health check passes
  │    → generates session_id (UUID)
  │
  └─ background_process.start_nonblocking()
       → ProcessManager(role="background") — starts llama-server on port 8082
       → daemon thread — does NOT block startup
  ↓
wait_for_chroma(timeout=120)
  → ChromaDB init on daemon thread (model download on first run)
  ↓
on_session_start(session_id)
  → crash recovery sweep
  → register session in active_sessions
  → seed scratchpad with semantic context
  ↓
extraction_worker.start()          → daemon thread for semantic fact extraction
chunk_summary_worker.start()       → daemon thread for rolling chunk summaries
working_memory_lifecycle.start()   → TTL expiry pruning (one-shot)
episodic_memory_lifecycle.start()  → decay-by-summarization (one-shot)
```

After this, the REPL loop in `cli.py` begins accepting user input.

[Startup sequence diagram placeholder]

---

## 2. Session start

`on_session_start(session_id)` in `SessionManager/session_lifecycle.py`:

| Step | What happens | File:Line |
|------|-------------|-----------|
| 1 | Set session_id in contextvars (for log lines) | `session_lifecycle.py:688` |
| 2 | Attach per-session file handler (full debug logs) | `session_lifecycle.py:690` |
| 3 | **Crash recovery**: sweep active_sessions for stale `in_progress` rows from previous process | `session_lifecycle.py:699` |
| 4 | Register this session in active_sessions (`status='in_progress'`) | `session_lifecycle.py:704` |
| 5 | Retrieve user goals from semantic memory (k=2) | `session_lifecycle.py:709` |
| 6 | Retrieve recent experience from semantic memory (k=2) | `session_lifecycle.py:712` |
| 7 | Fetch last 2 episodic memories (passive seed) | `session_lifecycle.py:723` |
| 8 | Seed scratchpad with combined context | `session_lifecycle.py:732` |

The crash recovery at step 3 runs BEFORE this session registers itself, so stale sessions from a previous crash are recovered first.

---

## 3. A single turn

When the user types a message, `cli.py` calls `ask_llm(query)` in `llm_client.py`:

```
User types message
  ↓
ask_llm(query)
  ↓
┌─ Set system message (once per session, for KV cache) ──────────┐
│  history_manager.set_system_message(SYSTEM_PROMPT)              │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Build dynamic context ────────────────────────────────────────┐
│  prompt_builder.build_dynamic_context(query)                    │
│    → memory_retriever.get_retrieved_context(query)              │
│       → scratchpad.get_compiled_memory()  (always)              │
│       → semantic_memory.retrieve_as_text(query, k=5)            │
│    → episodic_trigger.maybe_get_episodic_context(query)          │
│    → working_memory_db_client.get_active_reflections(limit=5)   │
│    → fallback chain if over 12K token budget                    │
│  user_turn_content = f"{dynamic_context}\n\n{query}"            │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Call LLM ─────────────────────────────────────────────────────┐
│  request_completion(trimmed_history, use_tools=True)             │
│    → llm_request_lock.post_completion(payload)                   │
│    → returns {text, native_calls, finish_reason}                 │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Execute tool calls ───────────────────────────────────────────┐
│  tool_executor.execute_tool_calls(text, native_calls)            │
│    → parse tag-based calls from text                             │
│    → execute tag calls (fire-and-forget)                         │
│    → execute native calls → {call_id: result_string}             │
│    → truncate results to 2000 chars                              │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Follow-up text reply (if pure tool-call response) ────────────┐
│  request_completion(history + tool_results + "give your text")   │
│  use_tools=False to avoid re-triggering tool calls               │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Persist turn data ────────────────────────────────────────────┐
│  1. Append assistant message to history_manager                  │
│  2. Queue turn for semantic extraction (extraction_worker)       │
│  3. Heartbeat active_sessions (turn_count++, last_turn_at)       │
│  4. Accumulate turn in _current_chunk_turns                      │
│  5. Every 5 turns → queue chunk to chunk_summary_worker          │
│  6. Overwrite full_conversation backup (crash recovery)          │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Return ───────────────────────────────────────────────────────┐
│  format_for_display(parsed_response) → ANSI for terminal only   │
│  (all stored data uses plain text, never ANSI)                  │
└────────────────────────────────────────────────────────────────┘
```

[Single turn flow diagram placeholder]

---

## 4. Background workers during a turn

While the user's turn is being processed, several background workers may be active:

### Extraction worker (`LLMEngine/extraction_worker.py`)

- **Trigger**: every turn, queued via `extraction_worker.queue_turn()`
- **Batching**: accumulates turns, flushes when either:
  - `MIN_EXTRACTION_INTERVAL = 30s` since last flush, OR
  - `MAX_BATCH_WAIT = 90s` since oldest queued turn
- **LLM call**: `memory_extractor.extract_and_store_batch()` — extracts 1-3 facts
- **Output**: stored in ChromaDB semantic memory (with deduplication)

### Chunk summary worker (`LLMEngine/chunk_summary_worker.py`)

- **Trigger**: every 5 turns, queued via `chunk_summary_worker.queue_chunk()`
- **LLM call**: `summarizer.summarize_chunk()` — produces a rough narrative note
- **Output**: appended to `active_sessions.chunk_summaries` (used at session end for episodic summary)

### Reflection worker (`LLMEngine/reflection_worker.py`)

- **Trigger**: every 5 turns + session end
- **LLM call**: produces up to 5 behavioral directives, each scored on 5 criteria
- **Output**: written to `working_memory` as `memory_type='reflection'` with computed `expires_at`
- **Visibility**: appears in the prompt on the VERY NEXT turn via PromptBuilder's live SQLite read

### Memory lifecycle workers (one-shot at startup)

- **Working memory TTL pruning**: deletes expired rows from SQLite
- **Episodic decay-by-summarization**: merges old episodes at 6/12/24 month thresholds

[Background workers diagram placeholder]

---

## 5. Clean session end

When the user types `/stop` or Ctrl+C, `cli.py` calls `on_session_end(session_id)`:

```
/stop typed or Ctrl+C
  ↓
┌─ Flush pending work ───────────────────────────────────────────┐
│  extraction_worker.flush_and_wait(timeout=120)                  │
│  (forces pending turns to extract immediately)                  │
└────────────────────────────────────────────────────────────────┘
  ↓
on_session_end(session_id)
  ↓
┌─ Idempotency check ────────────────────────────────────────────┐
│  if not is_session_active(session_id): return (already closed)  │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Promote scratchpad to semantic memory ─────────────────────────┐
│  1. Current goal → semantic_memory.store(category='goals')       │
│  2. Completed subtasks[:3] → store(category='experience')        │
│  3. Memory updates → store(category='other')                     │
│  4. Last error → store(category='experience')                    │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Write session summary to working memory ──────────────────────┐
│  working_memory_tool.insert_working_memory(                      │
│    memory_type='session_summary', ...)                           │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Write episodic memory ────────────────────────────────────────┐
│  Skip if turn_count < 2 and no goal/completed subtasks          │
│  Otherwise:                                                     │
│    1. episodic_summarizer.summarize_session(                    │
│         chunk_summaries, goal, subtasks, ...)                   │
│    2. episodic_memory_store.insert_episode(...)                  │
│    3. kg_sleep_queue_client.enqueue_session(...)                 │
│       (before close_session deletes active_sessions)             │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Close session marker ─────────────────────────────────────────┐
│  active_sessions_db_client.close_session(session_id)             │
│  (deletes the row entirely)                                      │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Reflection consolidation ─────────────────────────────────────┐
│  _consolidate_reflections(session_id)                            │
│    → fetch all memory_type='reflection' rows for this session    │
│    → skip if < 2 rows                                            │
│    → ONE background LLM call → classify each as:                 │
│       "keep" → leave in working_memory                           │
│       "delete" → hard-delete                                     │
│       "promote_to_semantic" → store in semantic_memory            │
│    → fully non-fatal                                             │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Reset scratchpad ─────────────────────────────────────────────┐
│  scratchpad.reset()                                              │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Stop servers ─────────────────────────────────────────────────┐
│  process_manager.stop_from_cli()                                 │
│  background_process.stop_if_running()                            │
└────────────────────────────────────────────────────────────────┘
```

[Clean shutdown diagram placeholder]

---

## 6. Crash recovery

If the process is killed (Ctrl+C, power loss, `kill -9`), `on_session_end` never runs. The active_sessions row remains `in_progress`. On next startup:

```
Application starts
  ↓
on_session_start(new_session_id)
  ↓
┌─ Crash recovery sweep ─────────────────────────────────────────┐
│  _recover_stale_sessions(current_session_id)                     │
│    → active_sessions WHERE status='in_progress'                  │
│      AND session_id != current_session_id                        │
│    → for each stale row:                                         │
│       _finalize_crashed_session(stale_row)                       │
└────────────────────────────────────────────────────────────────┘
  ↓
┌─ Finalize each crashed session ─────────────────────────────────┐
│  Data sources (in priority order):                               │
│    1. chunk_summaries — rolling 5-turn notes (preferred)         │
│    2. full_conversation — raw message backup (if no chunks)      │
│    3. related_semantic_memory_ids — linked fact IDs              │
│                                                                  │
│  Skip if turn_count < 1 AND no chunks AND no conversation        │
│                                                                  │
│  Otherwise:                                                      │
│    1. episodic_summarizer.summarize_crashed(...)                  │
│    2. episodic_memory_store.insert_episode(outcome='interrupted') │
│    3. kg_sleep_queue_client.enqueue_session(...)                  │
│    4. active_sessions_db_client.close_session(...)                │
└────────────────────────────────────────────────────────────────┘
```

The key insight: `chunk_summaries` and `full_conversation` are written live during the session (every 5 turns and every turn respectively), so a crash mid-session still has real data to recover. The older design lost everything because it only wrote at session end.

[Crash recovery diagram placeholder]

---

## 7. The /sleep command

The Knowledge Graph sleep pipeline is triggered manually by the user typing `/sleep`:

```
User types /sleep [N]
  ↓
cli.py → _handle_sleep(arg)
  ↓
sleep_scheduler.run_sleep_cycle(max_batches=N)
  ↓
┌─ For each pending session (up to N) ───────────────────────────┐
│                                                                  │
│  memory_selector.get_next_batch()                                │
│    → kg_sleep_queue WHERE processed_at IS NULL                   │
│    → hydrate with ChromaDB texts (episode + semantic)            │
│                                                                  │
│  entity_extractor.extract_entities_from_bundle(bundle)            │
│    → ONE background LLM call → entities + candidate relations    │
│                                                                  │
│  entity_resolver.resolve_entities(extraction)                     │
│    → cascade: exact name → alias → keyword → create new          │
│                                                                  │
│  subgraph_retriever.fetch_subgraph(resolved)                      │
│    → BFS from resolved nodes (1 hop, 15 nodes max)               │
│                                                                  │
│  operation_proposer.propose_operations(resolved, subgraph)        │
│    → ONE background LLM call → minimum graph operations          │
│                                                                  │
│  validator.validate_operations(proposed)                          │
│    → deterministic checks (node exists, no duplicate, etc.)      │
│                                                                  │
│  _execute_valid_ops(validation_results)                           │
│    → insert edges, update confidence, deactivate, add aliases    │
│                                                                  │
│  _link_memories_to_nodes(resolved, bundle)                        │
│    → kg_memory_nodes (links ChromaDB IDs → graph nodes)          │
│                                                                  │
│  mark_processed(session_id)                                       │
│    → stamps queue row (not deleted, just timestamped)             │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
  ↓
Print SleepReport summary
```

Each session goes through the full pipeline independently. If any step fails, the session stays pending for the next `/sleep`. The retry loop allows up to `MAX_RETRIES = 3` attempts per session before skipping.

[Sleep pipeline diagram placeholder]

---

## 8. Data flow between memory layers

A single turn touches multiple memory layers. Here's the complete data flow:

```
User message arrives
  │
  ├─→ Scratchpad (read: compiled memory for prompt)
  ├─→ Semantic Memory (read: k=5 relevant facts for prompt)
  ├─→ Episodic Memory (read: triggered by regex patterns in query)
  ├─→ Working Memory (read: top 5 reflections for prompt)
  │
  ├─→ LLM generates response
  │
  ├─→ Tool calls execute
  │    ├─→ Scratchpad (write: state updates, tool outputs)
  │    ├─→ Working Memory (write: session facts via bridge tool)
  │    ├─→ Semantic Memory (write: durable facts via bridge tool)
  │    └─→ Knowledge Graph (read: entity queries via bridge tool)
  │
  ├─→ Extraction Worker (queued: turn → semantic memory extraction)
  ├─→ Chunk Summary Worker (queued: every 5 turns → episodic chunk)
  ├─→ Reflection Worker (queued: every 5 turns → working memory reflections)
  ├─→ Active Sessions (write: heartbeat, full_conversation backup)
  └─→ History Manager (write: conversation history)
```

At session end:

```
Session ends
  │
  ├─→ Scratchpad (read: promote goals, subtasks, errors → semantic memory)
  ├─→ Semantic Memory (write: promoted scratchpad state)
  ├─→ Working Memory (write: session_summary entry)
  ├─→ Episodic Memory (write: summarize_session → insert_episode)
  ├─→ KG Sleep Queue (write: enqueue for /sleep processing)
  ├─→ Active Sessions (delete: close_session)
  ├─→ Reflection Consolidation (LLM: classify keep/delete/promote)
  │    ├─→ Working Memory (delete: noise reflections)
  │    └─→ Semantic Memory (write: promoted directives)
  └─→ Scratchpad (reset: clear all state)
```

---

## 9. Conversation history management

`LLMEngine/history_manager.py` maintains an in-memory list of messages:

- **System message**: set once per session (position 0), never changed — enables KV cache reuse
- **User/assistant pairs**: last `MAX_HISTORY_TURNS = 4` pairs kept (8 messages + system = 9 total)
- **Trimming**: on every `get_trimmed_history()`, the list is sliced to the last 8 messages

The system message contains the full `SYSTEM_PROMPT` (tool descriptions, memory type guidance, behavioral rules). Dynamic context (semantic memory, episodic recall, reflections) is attached to the current user turn instead, so only the small per-turn tail needs reprocessing.

---

## 10. Token budget management

`PromptBuilder/prompt_builder.py` enforces a `LOCAL_CTX_LIMIT = 12000` token budget for the dynamic context portion of each turn. When the budget is exceeded:

| Priority | What gets dropped |
|----------|-------------------|
| 1st | Episodic context |
| 2nd | Reflection directives |
| 3rd | Semantic retrieval (replaced with empty-query fallback) |
| 4th | All context dropped |

The system prompt itself is never trimmed — it's a fixed constant set once per session.

---

## File index

| File | Purpose |
|------|---------|
| `LLMEngine/cli.py` | REPL loop, /stop and /sleep handlers |
| `LLMEngine/llm_client.py` | `ask_llm()`, `request_completion()`, bootstrap |
| `LLMEngine/history_manager.py` | Conversation history + trimming |
| `LLMEngine/extraction_worker.py` | Background semantic extraction queue |
| `LLMEngine/chunk_summary_worker.py` | Background chunk summarization |
| `LLMEngine/reflection_worker.py` | Background reflection generation |
| `LLMEngine/llm_request_lock.py` | Per-role LLM request locking |
| `SessionManager/session_lifecycle.py` | `on_session_start`, `on_session_end`, crash recovery |
| `SessionManager/session_generator.py` | UUID session ID generation |
| `SessionManager/session_memory_tracker.py` | Semantic memory ID tracking per session |
| `Database/active_sessions_db_client.py` | Session CRUD + crash marker |
| `PromptBuilder/prompt_builder.py` | Per-turn context assembly + token budget |
| `MemoryManagement/memory_retriever.py` | Scratchpad + semantic memory assembly |
| `Runtime/process_manager.py` | llama-server process lifecycle |
| `Runtime/main_process.py` | Main model launch config |
| `Runtime/background_process.py` | Background model launch config |
