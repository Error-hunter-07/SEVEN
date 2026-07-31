"""
Runtime/background_process.py

Owns launch configuration for the BACKGROUND mini-LLM (role="background")
— a small, CPU-only model dedicated to semantic-memory extraction
(MemoryManagement/semantic_memory/memory_extractor.py) and episodic/
chunk summarization (MemoryManagement/episodic_memory/summarizer.py,
LLMEngine/chunk_summary_worker.py), so that work stops competing with
the main chat model for the same GPU-resident llama-server slot.

CHANGED (background mini-LLM): previously there was only ever one
llama-server process (role="main"), and every LLM call in the app —
including background memory-extraction and summarization work — shared
it via the single lock in LLMEngine/llm_request_lock.py. That lock made
the contention SAFE (no two requests ever hit the --parallel 1 server at
once) but did not remove it: a background extraction call still had to
wait its turn behind the main chat completion, and vice-versa. Running a
second, smaller model as its own process means the two roles now have
independent servers AND independent locks (see llm_request_lock.py's
_ENDPOINTS), so they can genuinely run at the same time.

MODULARITY: split out of Runtime/process_manager.py the same way
Runtime/main_process.py was — this module owns only the background
role's specific launch config and its non-blocking startup sequencing;
the generic subprocess/health-check machinery stays in process_manager.py.

Deliberately small footprint:
  - gpu_layers=0: the intent is to stay off the GPU entirely, so it never
    contends with the main model for VRAM.
  - device="none": CHANGED — gpu_layers=0 alone does NOT achieve that
    intent. Per llama.cpp's own docs, "The GPU may still be used to
    accelerate some parts of the computation even when using the -ngl 0
    option" — the KV cache and/or compute buffers can still land on the
    GPU even with zero layers offloaded, which is exactly what was
    showing up as an extra ~0.8GB of VRAM use (4.1GB -> 4.9GB) once this
    process started. --device none is the flag that actually disables
    GPU acceleration for a process; gpu_layers=0 is kept too since it's
    also correct/harmless, but device="none" is what's doing the real
    work now.
  - ctx_size=4096: background inputs (a handful of recent turns, or one
    short conversation snippet) are short; no reason to reserve the same
    32k-token KV cache budget the main model gets.
  - threads=4: a conservative default so this role doesn't grab every
    CPU thread on the machine — the main process still does real CPU-side
    work (tokenization, sampling, request handling) even while it's
    GPU-bound. Tune this against what you actually observe once it's
    running; there's nothing sacred about 4, it's a starting point.
"""

import threading
from typing import Optional

from GlobalHelpers.config import settings
from GlobalHelpers.logger import get_logger
from Runtime.process_manager import ProcessManager

log = get_logger(__name__)

PORT = 8082
CTX_SIZE = 4096
GPU_LAYERS = 0
THREADS = 4
DEVICE = "none"


def is_configured() -> bool:
    """
    The background model is opt-in. Existing deployments that haven't
    set BACKGROUND_LLM_MODEL_PATH yet keep running exactly as before —
    main model only, no .env changes forced on them. memory_extractor.py
    and summarizer.py's role="background" calls will fail (connection
    refused) until this is configured and start_nonblocking() below has
    actually been run — but those call sites already treat LLM failures
    as non-fatal (they log and fall back), so the app doesn't crash, it
    just doesn't get background extraction/summarization until it's set up.
    """
    return bool(settings.background_llm_model_path)


def get_or_create() -> ProcessManager:
    if not is_configured():
        raise RuntimeError(
            "Background model requested but not configured — set "
            "BACKGROUND_LLM_MODEL_PATH in .env (and BACKGROUND_LLM_CLI_PATH "
            "too, if the background model needs a different llama-server "
            "build than the main model's LLM_CLI_PATH)."
        )
    return ProcessManager(
        role="background",
        model_path=settings.background_llm_model_path,
        llama_cli_path=settings.background_llm_cli_path or settings.llm_cli_path,
        mmproj_path=None,
        ctx_size=CTX_SIZE,
        gpu_layers=GPU_LAYERS,
        port=PORT,
        threads=THREADS,
        device=DEVICE,
    )


def start_nonblocking() -> Optional[threading.Thread]:
    """
    Starts the background model on a daemon thread and returns
    immediately. Deliberately NOT synchronous like main_process's
    start_blocking(): the main chat model is what the user needs right
    away, and the background model isn't touched until the first
    extraction batch (5 turns in, per extraction_worker.py) or the first
    chunk summary (every 5 turns, per chunk_summary_worker.py) — plenty
    of time for it to finish loading in parallel without the user ever
    noticing it was still starting up.

    Returns the thread (daemon=True, so it won't block process exit) so
    callers COULD join() on it if they ever needed to, though nothing
    currently does. Returns None if the background model isn't
    configured — logs a warning and the app continues main-model-only.
    """
    if not is_configured():
        log.warning(
            "Background model not configured (BACKGROUND_LLM_MODEL_PATH unset) "
            "— skipping. Semantic-memory extraction and episodic/chunk "
            "summarization will keep failing non-fatally until it's set up."
        )
        return None

    def _run():
        try:
            pm = get_or_create()
            pm.start_for_client()
        except Exception:
            log.exception(
                "Background model failed to start — non-fatal, main chat "
                "continues without it."
            )

    thread = threading.Thread(target=_run, daemon=True, name="background-model-loader")
    thread.start()
    return thread


def stop_if_running() -> None:
    """
    Stops the background model's subprocess if one was ever started.
    Safe to call even when the background model was never configured
    (start_nonblocking() returned None and no ProcessManager for
    role="background" was ever registered) — becomes a no-op rather than
    raising, since callers (LLMEngine/cli.py's shutdown path) shouldn't
    need to know whether the background model was actually running.

    Without this, LLMEngine/cli.py's shutdown sequence — which only
    knows about the "main" role via LLMEngine.llm_client.process_manager
    — would stop the main llama-server but leave the background one
    running as an orphaned subprocess after the app exits.
    """
    try:
        pm = ProcessManager.get_instance(role="background")
    except RuntimeError:
        return  # never started — nothing to stop
    try:
        pm.stop()
    except Exception:
        log.exception("Failed to stop background model subprocess (non-fatal).")