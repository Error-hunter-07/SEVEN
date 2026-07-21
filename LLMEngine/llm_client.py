"""
LLMEngine/llm_client.py

Core library: process bootstrap, the raw completion request, and the
ask_llm() orchestration (tool execution, follow-up handling, memory
extraction queuing). No REPL loop here see LLMEngine/cli.py.

MODULARITY: this used to be one 383-line file doing five separate jobs.
Now split into:
    LLMEngine/history_manager.py    conversation history state + trimming
    LLMEngine/tool_schema.py        tool parameter dict -> JSON Schema
    LLMEngine/extraction_worker.py  background memory-extraction queue
    LLMEngine/cli.py                the __main__ REPL loop
llm_client.py itself is now importable as a pure library (e.g. by tests,
or by a future non-CLI frontend) without triggering an input() loop.
"""

import os
import sys
import requests

from ToolCalling.register import registry

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Tools.scratchpad_tool import get_scratchpad_memory

try:
    from .response_parser import parse_response
except ImportError:
    from response_parser import parse_response

import PromptBuilder.prompt_builder as prompt_builder
from Runtime.process_manager import ProcessManager
import ToolCalling.executor as tool_executor

from Database.chroma_db import wait_for_chroma
from GlobalHelpers.logger import configure_logging, get_logger
from GlobalHelpers.config import settings

import LLMEngine.history_manager as history_manager
import LLMEngine.tool_schema as tool_schema
import LLMEngine.extraction_worker as extraction_worker
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
    working_memory_lifecycle.start() #Added working memory lifecycle pruning at startup
    episodic_memory_lifecycle.start() #Added episodic memory decay-by-summarization at startup
except Exception:
    log.exception("Failed to start LLM process")


# max_tokens tuned to 2048 — large enough for code-bearing tool calls,
# well within the 16384 context window, combined with the truncation
# guard in request_completion below (see review notes: 500 errors were
# previously confirmed to occur at exactly max_tokens decoded, mid
# tool-call JSON, when this was set too low).
MAX_RESPONSE_TOKENS = 2048


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

    response = requests.post(
        "http://127.0.0.1:8081/v1/chat/completions",
        json=payload,
        timeout=120,
    )
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
    history_manager.set_system_message(prompt_builder.build_prompt(query))
    history_manager.append_user_message(query)

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
