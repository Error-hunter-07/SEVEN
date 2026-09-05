"""
Runtime/main_process.py

Owns launch configuration for the MAIN chat model (role="main") — the
model the user directly talks to on every turn.

MODULARITY: split out of Runtime/process_manager.py as part of adding the
background mini-LLM. process_manager.py keeps the generic, role-agnostic
ProcessManager class (start/stop/health-check/registry) — the mechanics
that are identical regardless of which model is being run. This module
owns only what's specific to the main role: which port it binds, how
much context/GPU it gets, and that its startup BLOCKS the caller (the
user is waiting directly on this one, unlike the background role — see
Runtime/background_process.py).
"""

from GlobalHelpers.config import settings
from GlobalHelpers.logger import get_logger
from Runtime.process_manager import ProcessManager

log = get_logger(__name__)

PORT = 8081
CTX_SIZE = 32768
GPU_LAYERS = 20


def get_or_create() -> ProcessManager:
    """
    Builds (on first call) or retrieves (on later calls) the ProcessManager
    for the main chat model. Safe to call more than once — ProcessManager's
    own registry returns the same cached instance for role="main" every
    time (see ProcessManager.__new__), so later calls' kwargs are ignored
    just like the old singleton's behavior.
    """
    return ProcessManager(
        role="main",
        model_path=settings.llm_model_path,
        llama_cli_path=settings.llm_cli_path,
        mmproj_path=None,
        ctx_size=CTX_SIZE,
        gpu_layers=GPU_LAYERS,
        port=PORT,
    )


def start_blocking() -> ProcessManager:
    """
    Starts the main model and BLOCKS until it's actually ready to
    generate (start_for_client() = start() + wait_until_ready()). This is
    deliberately synchronous: the main model is what the user is waiting
    on to begin chatting, so there's no benefit to returning early here
    the way there is for the background role.
    """
    pm = get_or_create()
    pm.start_for_client()
    return pm