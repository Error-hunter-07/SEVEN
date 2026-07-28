"""
LLMEngine/llm_client.py

Core library: process bootstrap, the raw completion request, and the
ask_llm() orchestration (tool execution, follow-up handling, memory
extraction queuing, rolling episodic chunk summarization). No REPL loop
here — see LLMEngine/cli.py.

MODULARITY: this used to be one 383-line file doing five separate jobs.
Now split into:
    LLMEngine/history_manager.py       conversation history state + trimming
    LLMEngine/tool_schema.py           tool parameter dict -> JSON Schema
    LLMEngine/extraction_worker.py     background memory-extraction queue
    LLMEngine/chunk_summary_worker.py  background rolling episodic chunk summarizer
    LLMEngine/llm_request_lock.py      shared lock around every LLM HTTP call
    LLMEngine/cli.py                   the __main__ REPL loop
llm_client.py itself is now importable as a pure library (e.g. by tests,
or by a future non-CLI frontend) without triggering an input() loop.
"""

import os
import sys

from ToolCalling.register import registry

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Tools.scratchpad_tool import get_scratchpad_memory

try:
    from .response_parser import parse_response
except ImportError:
    from response_parser import parse_response

import requests
import PromptBuilder.prompt_builder as prompt_builder
from Runtime.process_manager import ProcessManager
import ToolCalling.executor as tool_executor

from Database.chroma_db import wait_for_chroma
from GlobalHelpers.logger import configure_logging, get_logger
from GlobalHelpers.config import settings

import LLMEngine.history_manager as history_manager
import LLMEngine.tool_schema as tool_schema
import LLMEngine.extraction_worker as extraction_worker
import LLMEngine.chunk_summary_worker as chunk_summary_worker
import LLMEngine.llm_request_lock as llm_request_lock
import MemoryManagement.working_memory.memory_lifecycle as working_memory_lifecycle
import MemoryManagement.episodic_memory.memory_lifecycle as episodic_memory_lifecycle
import Database.active_sessions_db_client as active_sessions_db_client

# Configure logging once for the process before other components initialize
configure_logging()
log = get_logger(__name__)

process_manager = ProcessManager(
    model_path=settings.llm_model_path,
    llama_cli_path=settings.llm_cli_path,
    mmproj_path=settings.mmproj_path
)

try:
    process_manager.start_for_client()

    # Wait for ChromaDB to finish loading in parallel with llama-server
    if not wait_for_chroma(timeout=120):
        log.warning("ChromaDB did not initialize in time.")

    from SessionManager.session_lifecycle import on_session_start
    session_id = process_manager.session_id
    on_session_start(session_id)
    extraction_worker.start()
    chunk_summary_worker.start()  # rolling episodic chunk summarizer, every 5 turns
    working_memory_lifecycle.start()   # working memory TTL pruning at startup
    episodic_memory_lifecycle.start()  # episodic memory decay-by-summarization at startup
except Exception:
    log.exception("Failed to start LLM process")


# max_tokens tuned to 2048 — large enough for code-bearing tool calls,
# well within the 16384 context window, combined with the truncation
# guard in request_completion below (see review notes: 500 errors were
# previously confirmed to occur at exactly max_tokens decoded, mid
# tool-call JSON, when this was set too low).
MAX_RESPONSE_TOKENS = 2048

# Rolling episodic chunk summarization: every 5 turns, the last 5
# (query, assistant_activity) pairs get handed to
# chunk_summary_worker.queue_chunk() and this local accumulator resets.
# Deliberately a plain list, not a queue — ask_llm() is only ever called
# from the single-threaded CLI REPL loop, same as history_manager's own
# message list, so no lock is needed here.
CHUNK_INTERVAL_TURNS = 5
_current_chunk_turns: list[tuple[str, str]] = []


def request_completion(request_messages: list[dict], use_tools: bool = True) -> dict:
    """
    Returns a dict with:
        text            str   — content field from the model (may be empty)
        native_calls    list  — tool_calls list if finish_reason=tool_calls, else []
        finish_reason   str   — stop | tool_calls | length | null

    use_tools=False skips sending the "tools" schema entirely — use this
    for text-only follow-up requests (e.g. "now give your text reply")
    where forcing tool-call grammar serves no purpose and increases the
    chance of a truncated/invalid generation, especially for code-heavy
    replies.

    CHANGED: now goes through LLMEngine.llm_request_lock.post_completion
    instead of calling requests.post directly — the local server runs
    --parallel 1 and can only process one request at a time, so this
    shared lock keeps the main chat turn from racing the background
    extraction worker or the episodic chunk/session summarizers for the
    server's single processing slot.
    """
    payload = {
        "model":       settings.llm_model,
        "messages":    request_messages,
        "temperature": 0.7,
        "max_tokens":  MAX_RESPONSE_TOKENS,
    }
    if use_tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name":        tool.name,
                    "description": tool.description,
                    "parameters":  tool_schema.build_tool_schema(tool.parameters),
                }
            }
            for tool in registry.list_tools()
        ]

    response = llm_request_lock.post_completion(payload, timeout=120)
    response.raise_for_status()
    data    = response.json()
    choice  = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "")

    native_calls = message.get("tool_calls") or []

    # Detect truncated tool-call arguments: confirmed via llama-server
    # logs to occur at exactly max_tokens decoded, mid tool-call JSON
    # ("unexpected end of input", "missing closing quote"). If truncated,
    # the arguments are unreliable — discard rather than risk a malformed
    # downstream call.
    if finish_reason == "length" and native_calls:
        log.warning(
            "Response truncated at max_tokens mid tool-call — "
            "discarding likely-incomplete tool_calls."
        )
        native_calls = []

    return {
        "text":          message.get("content") or "",
        "native_calls":  native_calls,
        "finish_reason": finish_reason,
    }


def ask_llm(query: str) -> str | None:
    # CHANGED: the system message is now set ONCE to the literal
    # SYSTEM_PROMPT constant and never touched again for the rest of the
    # session — previously it was rebuilt every turn with per-query
    # semantic/episodic context spliced in, which meant message[0] (the
    # start of the prompt) changed almost every request and defeated
    # llama-server's --cache-prompt/--cache-reuse (see
    # PromptBuilder.prompt_builder.build_dynamic_context's docstring for
    # the full explanation). Calling this every turn with the same
    # constant string is harmless either way — identical content means
    # identical tokens, so it doesn't cost anything even though it's
    # redundant after the first call.
    history_manager.set_system_message(prompt_builder.SYSTEM_PROMPT)

    # The dynamic part (retrieved semantic memory + any triggered
    # episodic recall) is now glued onto THIS turn's user message instead
    # of the system message, so only this small, always-new tail needs
    # reprocessing each turn rather than the whole prompt.
    dynamic_context = prompt_builder.build_dynamic_context(query)
    user_turn_content = f"{dynamic_context}\n\n{query}" if dynamic_context.strip() else query
    history_manager.append_user_message(user_turn_content)

    try:
        trimmed = history_manager.get_trimmed_history()
        result  = request_completion(trimmed)

        text          = result["text"]
        native_calls  = result["native_calls"]
        finish_reason = result["finish_reason"]

        if not text and not native_calls:
            log.warning("Empty response from model.")
            return "I didn't get a response. Please try again."

        # Execute tools and capture what each one actually returned, keyed
        # by tool_call_id, so the follow-up "give your text reply" turn
        # can see real results instead of a hardcoded "Done." placeholder.
        tool_results = tool_executor.execute_tool_calls(
            text=text,
            native_tool_calls=native_calls,
        )

        history_message = {
            "role":    "assistant",
            "content": text or None,
        }
        if native_calls:
            history_message["tool_calls"] = native_calls

        parsed_response = parse_response(text) if text else ""

        # If there's no text (pure tool-call reply), ask for a follow-up text reply
        if not parsed_response.strip():
            follow_messages = history_manager.get_trimmed_history() + [
                history_message,
                *[
                    {
                        "role":         "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content":      tool_results.get(tc.get("id", ""), "Done."),
                    }
                    for tc in native_calls
                ],
                {"role": "user", "content": "Now give your text reply."},
            ]
            try:
                # use_tools=False: this is a text-only follow-up, forcing
                # tool-call grammar here served no purpose and was the
                # second half of a truncation -> 500 error chain seen in
                # production logs.
                follow_result = request_completion(follow_messages, use_tools=False)
                parsed_response = parse_response(follow_result["text"])
            except requests.exceptions.HTTPError as e:
                log.error("Follow-up completion failed: %s", e, exc_info=True)
                parsed_response = text or "I ran into an issue finishing that response — could you try rephrasing or asking again?"

        if not parsed_response.strip():
            parsed_response = "Done."

        history_manager.append_message(history_message)

        # Build a summary of any native tool calls so memory extraction
        # still has assistant context even when text == "".
        tool_summary = ""
        if native_calls:
            tool_summary = " ".join(
                f"[Called {tc['function']['name']}]" for tc in native_calls
            )
        full_assistant_activity = f"{text} {tool_summary}".strip()

        extraction_worker.queue_turn(query, full_assistant_activity)

        # Durable turn counter for episodic memory — see
        # Database/active_sessions_db_client.py. Written to SQLite (WAL
        # mode) so it survives a crash even though history_manager's
        # message list itself is only ever in-memory.
        try:
            active_sessions_db_client.heartbeat(process_manager.session_id)
        except Exception:
            log.exception("Failed to update active_sessions heartbeat (non-fatal).")

        # Rolling episodic chunk summarization: accumulate this turn,
        # and every CHUNK_INTERVAL_TURNS turns hand the accumulated
        # slice off to the background chunk summarizer, then reset for
        # the next window. Enqueuing is non-blocking — the actual LLM
        # call happens on chunk_summary_worker's own thread, guarded by
        # the same shared llm_request_lock request_completion() uses, so
        # it never competes with this turn or the extraction worker for
        # the server's single slot in an unbounded way.
        _current_chunk_turns.append((query, full_assistant_activity))
        try:
            turn_count = active_sessions_db_client.get_turn_count(process_manager.session_id)
            if turn_count > 0 and turn_count % CHUNK_INTERVAL_TURNS == 0 and _current_chunk_turns:
                chunk_summary_worker.queue_chunk(process_manager.session_id, list(_current_chunk_turns))
                _current_chunk_turns.clear()
        except Exception:
            log.exception("Failed to queue rolling chunk summary (non-fatal).")

        # Crash backup: overwrite the full conversation snapshot every
        # turn. Cheap at the sizes this app deals with, and the
        # last-resort recovery source for whatever hasn't been
        # chunk-summarized yet (e.g. a crash within the first 5 turns,
        # before the first chunk fires) — see
        # SessionManager/session_lifecycle.py's crash-recovery path.
        try:
            active_sessions_db_client.save_full_conversation(
                process_manager.session_id, history_manager.get_full_history()
            )
        except Exception:
            log.exception("Failed to save full_conversation backup (non-fatal).")

        log.debug(get_scratchpad_memory())
        return parsed_response

    except requests.exceptions.ConnectionError:
        log.error("Could not connect to local LLM server at :8081.")
    except requests.exceptions.Timeout:
        log.error("Model timed out.")
    except requests.exceptions.RequestException as e:
        log.error("Request failed: %s", e, exc_info=True)
    except Exception:
        log.exception("Unexpected error during ask_llm")