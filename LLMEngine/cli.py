"""
LLMEngine/cli.py

The interactive REPL loop split out of llm_client.py so llm_client
can be imported as a pure library (by tests, or by a future non-CLI
frontend) without triggering an input() loop as a side effect of import.

Run with: python -m LLMEngine.cli
"""

from LLMEngine.llm_client import ask_llm, process_manager
import LLMEngine.extraction_worker as extraction_worker
from SessionManager.session_lifecycle import on_session_end
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


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

            if user_query.strip().lower() == "/stop":
                extraction_worker.flush_and_wait(timeout=120)
                on_session_end(process_manager.session_id)
                process_manager.stop_from_cli()
                break

            answer = ask_llm(user_query)

            print("\nAssistant:")
            print(answer)
    except KeyboardInterrupt:
        print("\nInterrupted — closing session cleanly...")
        log.warning("KeyboardInterrupt received — closing session for %s.", process_manager.session_id)
    except Exception:
        log.exception("Unhandled exception in CLI loop — attempting a clean session close.")
    finally:
        # Order matters here: flush pending extraction work and write the
        # episodic summary WHILE the LLM server is still guaranteed to be
        # running (see Runtime/process_manager.py's process-group isolation
        # — the server no longer dies from the console's own Ctrl+C, so it's
        # still up at this point even after an interrupt), and only stop
        # the server as the very last step.
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


if __name__ == "__main__":
    run()
