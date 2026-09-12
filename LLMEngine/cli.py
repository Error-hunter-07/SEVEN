"""
LLMEngine/cli.py

The interactive REPL loop split out of llm_client.py so llm_client
can be imported as a pure library (by tests, or by a future non-CLI
frontend) without triggering an input() loop as a side effect of import.

Run with: python -m LLMEngine.cli

ADDED: /sleep and /sleep N commands that trigger the Knowledge Graph
sleep pipeline. The pipeline is imported lazily (inside the handler)
so a missing KnowledgeGraph package never blocks normal chat startup.

COMMANDS:
  /stop          — flush pending work, write episodic memory, exit.
  /sleep         — run the KG sleep pipeline (up to MAX_BATCHES_PER_SLEEP sessions).
  /sleep N       — run the KG sleep pipeline for up to N sessions.
  /sleep status  — show how many sessions are pending in the queue.
"""

from LLMEngine.llm_client import ask_llm, process_manager
import LLMEngine.extraction_worker as extraction_worker
import Runtime.background_process as background_process
from SessionManager.session_lifecycle import on_session_end
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# /sleep command handler
# ---------------------------------------------------------------------------

def _handle_sleep(arg: str) -> None:
    """
    Handle /sleep [N | status].

    /sleep          → run up to MAX_BATCHES_PER_SLEEP sessions
    /sleep N        → run up to N sessions (N must be a positive integer)
    /sleep status   → print pending session count, do not process anything
    """
    try:
        from KnowledgeGraph.sleep_scheduler import run_sleep_cycle
        from KnowledgeGraph.memory_selector import get_queue_status
        from KnowledgeGraph.constants import MAX_BATCHES_PER_SLEEP
    except ImportError as e:
        print(f"[Sleep] KnowledgeGraph package not available: {e}")
        return

    arg = arg.strip().lower()

    # /sleep status
    if arg == "status":
        try:
            total, pending = get_queue_status()
            if pending < 0:
                print("[Sleep] Could not read queue status (DB error).")
            elif pending == 0:
                print(f"[Sleep] Queue is empty — knowledge graph is up to date. ({total} sessions total)")
            else:
                print(f"[Sleep] {pending} session(s) pending / {total} total in queue.")
        except Exception as e:
            print(f"[Sleep] Error reading queue status: {e}")
        return

    # /sleep N
    max_batches = MAX_BATCHES_PER_SLEEP
    if arg:
        try:
            n = int(arg)
            if n <= 0:
                print(f"[Sleep] N must be a positive integer, got {n!r}.")
                return
            max_batches = n
        except ValueError:
            print(f"[Sleep] Unknown argument {arg!r}. Usage: /sleep | /sleep N | /sleep status")
            return

    # Run the pipeline
    try:
        run_sleep_cycle(max_batches=max_batches, batch_size=1, print_progress=True)
    except Exception as e:
        log.exception("_handle_sleep: unhandled error during sleep cycle.")
        print(f"[Sleep] Error during sleep cycle: {e}")


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def run() -> None:
    """
    The /stop path and the finally-block fallback below can both end up
    calling on_session_end for the same session — that's intentional,
    not a bug. on_session_end is idempotent (guarded by the
    active_sessions marker), so the finally block acts purely as a
    safety net: Ctrl+C during input(), an uncaught exception mid-turn,
    or any other unexpected exit still gets a clean episodic-memory
    write instead of silently losing that session's history.
    """
    try:
        while True:
            try:
                user_query = input("You: ")
            except EOFError:
                # stdin closed (e.g. piped input ran out) — treat like a
                # clean stop rather than falling through to the generic
                # exception handler below.
                break

            stripped = user_query.strip()

            # ── /stop ─────────────────────────────────────────────────
            if stripped.lower() == "/stop":
                extraction_worker.flush_and_wait(timeout=120)
                on_session_end(process_manager.session_id)
                process_manager.stop_from_cli()
                background_process.stop_if_running()
                break

            # ── /sleep [N | status] ───────────────────────────────────
            if stripped.lower().startswith("/sleep"):
                arg = stripped[len("/sleep"):].strip()
                _handle_sleep(arg)
                continue

            # ── normal turn ───────────────────────────────────────────
            answer = ask_llm(user_query)

            print("\nAssistant:")
            print(answer)

    except KeyboardInterrupt:
        print("\nInterrupted — closing session cleanly...")
        log.warning(
            "KeyboardInterrupt received — closing session for %s.",
            process_manager.session_id,
        )
    except Exception:
        log.exception("Unhandled exception in CLI loop — attempting a clean session close.")
    finally:
        # Order matters here: flush pending extraction work and write the
        # episodic summary WHILE the LLM server is still guaranteed to be
        # running, and only stop the server as the very last step.
        try:
            extraction_worker.flush_and_wait(timeout=120)
        except Exception:
            log.exception("Failed to flush extraction worker during shutdown (non-fatal).")
        try:
            on_session_end(process_manager.session_id)
        except Exception:
            log.exception("on_session_end failed during shutdown (non-fatal).")
        try:
            process_manager.stop_from_cli()
        except Exception:
            log.exception("process_manager.stop_from_cli failed during shutdown (non-fatal).")
        background_process.stop_if_running()


if __name__ == "__main__":
    run()