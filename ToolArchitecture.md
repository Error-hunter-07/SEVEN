# SEVEN — Tool Architecture

SEVEN's tool system lets the LLM interact with its memory layers, scratchpad, and knowledge graph without touching any storage backend directly. The LLM emits tool calls in its response, the system executes them, and feeds results back — all within a single turn.

The system supports two tool-call formats simultaneously: native OpenAI-style function calling (preferred) and tag-based `<tool_call>` parsing (fallback for local models that don't support native tool calling).

[Tool call flow diagram placeholder]

---

## Overview

Every tool follows the same pattern:

```
LLM emits tool call → ToolCalling parses it → ToolCalling dispatches to registered function
  → Function executes (reads/writes SQLite or ChromaDB) → Result returned to LLM
```

The LLM never touches a database, file system, or API directly. All storage access goes through bridge modules in `Tools/` that translate tool calls into safe database operations.

---

## Tool registration

All tools are registered once at module load time in `ToolCalling/register.py`. The `ToolRegistry` class maps tool names to `Tool` dataclass instances:

```python
@dataclass
class Tool:
    name: str           # e.g. "store_semantic_memory"
    description: str    # shown to the LLM in the function schema
    parameters: dict    # {param_name: description} — inferred into JSON Schema
    func: Callable      # the Python function to execute
```

At startup, `llm_client.py` reads the registry and builds OpenAI-compatible function schemas via `tool_schema.build_tool_schema()`, which infers JSON Schema types from the leading word of each parameter's description string (e.g. `"bool - ..."` → `{"type": "boolean"}`).

[Tool registration diagram placeholder]

---

## Tool call formats

### Native (OpenAI function calling)

The primary format. The LLM server returns tool calls as a structured list in the response:

```json
{
  "message": {
    "content": "",
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "store_semantic_memory",
          "arguments": "{\"text\": \"User prefers Python\", \"importance\": 0.7}"
        }
      }
    ]
  },
  "finish_reason": "tool_calls"
}
```

Parsed directly from the API response in `llm_client.py:149`.

### Tag-based (fallback)

For local models that don't support native function calling. The LLM emits tool calls as text tags in its content:

```xml
<tool_call>
{"tool": "store_semantic_memory", "arguments": {"text": "User prefers Python"}}
</tool_call>
```

Parsed by `ToolCalling/parser.py` using regex. Supports both canonical `<tool_call>` and pipe-delimited `<|tool_call|>` variants that some local model chat templates produce.

**Priority**: native calls are always processed first. Tag-based calls are processed from the text content afterward.

---

## Tool call lifecycle

### Step 1: LLM generates response

`llm_client.request_completion()` sends the conversation history + tool schemas to the LLM server. Returns:
- `text` — content field (may be empty for pure tool-call responses)
- `native_calls` — list of tool calls from the API response
- `finish_reason` — `stop`, `tool_calls`, `length`, or `null`

If `finish_reason == "length"` and there are native calls, the arguments are likely truncated mid-JSON — they are discarded (`llm_client.py:156-161`).

### Step 2: Tool execution

`ToolCalling/executor.execute_tool_calls()` receives both the raw text and the native calls:

1. Parse tag-based calls from `text` via `parser.parse_tool_calls()`
2. Parse native calls from the structured list
3. Execute tag calls first (fire-and-forget, results not returned to LLM)
4. Execute native calls, collecting `{tool_call_id: result_string}` for each
5. Truncate each result to 2000 chars to prevent prompt overflow

### Step 3: Follow-up text reply (if needed)

If the model replied with tool calls only and no text (`parsed_response` is empty), `llm_client` sends a follow-up request:

```
[conversation history] + [assistant tool_calls] + [tool results as "tool" messages]
+ "Now give your text reply."
```

This follow-up uses `use_tools=False` to avoid re-triggering tool calls.

### Step 4: History and memory updates

The final response (with tool call tags stripped) is:
- Appended to conversation history (`history_manager`)
- Queued for semantic memory extraction (`extraction_worker`)
- Saved as a crash backup (`active_sessions_db_client.save_full_conversation`)
- Accumulated for chunk summarization (`_current_chunk_turns`)
- ANSI-formatted only at the final return point for terminal display

---

## Tool catalog

### Scratchpad tools

Bridge to the in-memory scratchpad (`MemoryManagement/shortterm_memory/scratchpad.py`).

| Tool | Parameters | Description |
|------|-----------|-------------|
| `update_scratchpad_state` | `section`, `key`, `value` | Update planning/execution/reflection state |
| `get_scratchpad_state` | — | Return full scratchpad state dict |
| `update_scratchpad_summary` | `summary_text` | Append to conversation summary |
| `get_scratchpad_retrieved_context` | `working_memory_only`, `include_tool_outputs`, `include_all_working_memory` | Pull context into scratchpad |

**Allowed sections and keys:**

| Section | Keys |
|---------|------|
| `planning` | `current_goal`, `subtasks`, `completed_subtasks`, `current_step`, `next_action` |
| `execution` | `active_tool`, `retry_count`, `last_error` |
| `reflection` | `seven_notes` |
| `tool_outputs` | any string key (dynamic tool names) |

### Working memory tools

Bridge to SQLite working memory (`Database/working_memory_db_client.py`).

| Tool | Parameters | Description |
|------|-----------|-------------|
| `add_scratchpad_memory_update` | `memory_type`, `key`, `value`, `priority`, `relevance`, `source`, `tags`, `memory_id`, `update` | Insert or update a working memory entry |

The LLM-facing name is historical — this writes to working_memory, not the scratchpad. The scratchpad just tracks the audit trail.

Internal functions (`insert_working_memory`, `update_working_memory`, `get_working_memory`) are also called by `session_lifecycle` and `reflection_worker` but are NOT directly exposed to the LLM via the registry.

### Semantic memory tools

Bridge to ChromaDB semantic memory (`MemoryManagement/semantic_memory/semantic_memory.py`).

| Tool | Parameters | Description |
|------|-----------|-------------|
| `store_semantic_memory` | `text`, `importance`, `category`, `polarity` | Save a durable fact about the user |
| `search_semantic_memory` | `query`, `k` | Search for standing facts by similarity |

Deduplication happens inside `semantic_memory.store()` — near-identical facts (cosine distance ≤ 0.08) are merged, not duplicated. Negation detection prevents merging "likes X" with "dislikes X".

### Episodic memory tools

Bridge to ChromaDB episodic memory (`MemoryManagement/episodic_memory/episodic_memory_store.py`).

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_episodic_memory` | `query`, `k` | Semantic search over past session summaries |
| `browse_episodic_memory` | `mode`, `query`, `limit`, `session_id`, `within_days` | Browse by access pattern (recent/oldest/semantic/by_session) |

A deterministic trigger (`LLMEngine/episodic_trigger.py`) also calls the same search logic automatically for recall-shaped phrasing ("last time", "we discussed") before the LLM ever sees the message.

### Knowledge Graph tools

Bridge to the KG query service (`KnowledgeGraph/kg_query_service.py`).

| Tool | Parameters | Description |
|------|-----------|-------------|
| `query_knowledge_graph` | `query` | Look up entities and their relationships |

Uses 4 candidate generation methods (exact name, alias, keyword, semantic) fused via Reciprocal Rank Fusion.

---

## Tool schema generation

`LLMEngine/tool_schema.py` converts the flat `{param_name: description}` dict from each `Tool` into OpenAI-compatible JSON Schema:

```python
{"param_name": "str - A self-contained fact"}  →
{"type": "object", "properties": {"param_name": {"type": "string", "description": "A self-contained fact"}}}
```

Type inference is based on the leading word of the description:
- `"str - ..."` → `{"type": "string"}`
- `"int - ..."` → `{"type": "integer"}`
- `"float - ..."` → `{"type": "number"}`
- `"bool - ..."` → `{"type": "boolean"}`

All parameters are treated as optional in the schema (no `required` field) — the tool execution layer handles missing arguments gracefully.

[Tool schema generation diagram placeholder]

---

## Response parsing

`LLMEngine/response_parser.py` provides two distinct transformations:

| Function | Purpose | When to use |
|----------|---------|-------------|
| `strip_tool_call_tags()` | Remove tool-call wrapper tags from text | Storage, history, memory extraction — anything persisted |
| `format_for_display()` | Convert markdown to ANSI terminal formatting | Printing to console only |

**Critical rule**: `format_for_display()` output must NEVER be stored or re-fed into another LLM call. Only `strip_tool_call_tags()` output is storage-safe. This separation prevents control characters from leaking into stored data and degrading background model quality.

A backwards-compat alias `parse_response()` points to `strip_tool_call_tags()` so code importing the old name gets storage-safe behavior.

---

## Result truncation

All tool results are truncated to 2000 characters (`ToolCalling/executor.py:70-71`):

```python
if isinstance(result, (dict, list)):
    return json.dumps(result, default=str)[:2000]
return str(result)[:2000]
```

This prevents any single tool call from overwhelming the context window. Large results (e.g. a knowledge graph query returning many nodes) are silently clipped — the LLM sees the first 2000 chars and can follow up with a more targeted query if needed.

---

## Error handling

- **Unknown tool name**: logged and returned as `"Unknown tool: {name}"` — never raises
- **Missing arguments**: tool functions use default values or return `"Done."` for None results
- **TypeError on call**: tries calling with no args as fallback, returns `"Failed to call {name}."` if that also fails
- **JSON decode error on tag payload**: logged and dropped — no best-effort salvage
- **Truncated tool calls** (`finish_reason == "length"`): native calls discarded entirely

All errors are non-fatal — a failed tool call never crashes the turn or blocks the REPL.

---

## Shutdown behavior

`extraction_worker.flush_and_wait(timeout=120)` is called during orderly shutdown (`/stop` or `finally` block) to force any pending extraction work to complete immediately. The worker's queue may hold turns that haven't hit the batching threshold yet — without `flush_and_wait()`, those would be silently lost when the daemon thread dies with the process.

---

## File index

| File | Purpose |
|------|---------|
| `Tools/tool.py` | Base `Tool` dataclass |
| `ToolCalling/register.py` | Central tool registry + all tool registrations |
| `ToolCalling/parser.py` | Tag-based `<tool_call>` regex parser |
| `ToolCalling/executor.py` | Tool call dispatch + result collection |
| `LLMEngine/tool_schema.py` | `{param: description}` → JSON Schema conversion |
| `LLMEngine/response_parser.py` | Storage-safe tag stripping + ANSI display formatting |
| `Tools/scratchpad_tool.py` | Scratchpad bridge (planning, execution, reflection state) |
| `Tools/working_memory_tool.py` | Working memory bridge (session-scoped facts) |
| `Tools/semantic_memory_tool.py` | Semantic memory bridge (durable user facts) |
| `Tools/episodic_memory_tool.py` | Episodic memory bridge (past session recall) |
| `Tools/kg_query_tool.py` | Knowledge Graph bridge (entity relationships) |
