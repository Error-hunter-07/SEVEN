import subprocess
import sys
import time
from typing import Optional, TextIO
import SessionManager.session_generator as session_generator
from Runtime.health_check import wait_until_ready
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


class ProcessManager:
    """
    Persistent llama.cpp subprocess manager.

    Responsibilities:
    - start / stop / restart the model subprocess

    MODULARITY: HTTP readiness polling used to live inline in this class
    (wait_until_ready was ~45 lines mixing two unrelated concerns —
    subprocess lifecycle vs. HTTP polling — into one class). That logic
    now lives in Runtime/health_check.py as a standalone function; this
    class just delegates to it.

    CHANGED: Enforced true singleton via __new__. Previously _instance was
    set at the end of __init__, meaning you could construct a second
    ProcessManager() and get a fresh object while _instance still pointed to
    the first one (or vice-versa). Now __new__ raises RuntimeError on a second
    construction attempt, making accidental double-instantiation impossible.
    To create the singleton: ProcessManager(model_path=..., ...).
    To retrieve it later: ProcessManager.get_instance().
    """

    _instance: Optional['ProcessManager'] = None

    def __new__(cls, *args, **kwargs):
        # CHANGED: Singleton guard in __new__. If an instance already exists
        # and the caller is trying to construct a second one, raise immediately
        # so the bug is visible rather than silently creating a duplicate.
        if cls._instance is not None:
            raise RuntimeError(
                "ProcessManager is a singleton. "
                "Use ProcessManager.get_instance() to retrieve the existing instance."
            )
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __init__(
        self,
        model_path: str = None,
        mmproj_path: str = None,
        llama_cli_path: str = None,
        ctx_size: int = 32768,
        gpu_layers: int = 999,
    ):
        # CHANGED: Guard against __init__ being called again on the already-
        # initialised singleton (Python calls __init__ every time even when
        # __new__ returns an existing instance if you bypass the guard above).
        if getattr(self, "_initialised", False):
            return

        self.model_path = model_path
        self.llama_cli_path = llama_cli_path
        self.mmproj_path = mmproj_path
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers

        self.process: Optional[subprocess.Popen] = None
        self.log_file: Optional[TextIO] = None

        self.alive = False
        self.session_id = session_generator.generate_universal_session_id()
        log.info("Session id generated: %s", self.session_id)

        # CHANGED: Mark as initialised so repeated __init__ calls are no-ops.
        self._initialised = True

    # PROCESS LIFECYCLE

    def start(self):

        if self.alive:
            log.info("Already running")
            return

        command = [
            self.llama_cli_path,

            # MODEL
            "-m", self.model_path,

            # MULTIMODAL PROJECTOR (REQUIRED FOR VL MODELS)
            # ---- removed---- "--mmproj", self.mmproj_path, ---reason:- we don't need multimodal projector for text-only works, we will run it only when needed to save on the VRAM, with this VRAM spiked to 5.3GB and without this it stays at 4.1GB with 32k context

            # CONTEXT WINDOW
            "-c", str(self.ctx_size),

            # GPU OFFLOAD LAYERS
            "-ngl", str(self.gpu_layers),

            "--cache-prompt",

            "--cache-reuse", "256",

            "--ctx-checkpoints", "64",

            "--parallel", "1", # changed to one to preserve KV cache VRAM

            "--cont-batching",

            "--host", "127.0.0.1",

            "--port", "8081"
        ]

        log.info("Starting llama.cpp subprocess...")

        # Detach the child from the console's Ctrl+C signal group. Without
        # this, pressing Ctrl+C in the terminal sends SIGINT (POSIX) /
        # CTRL_C_EVENT (Windows) to EVERY process in the console's process
        # group — including this subprocess — killing the LLM server
        # instantly and independently of our own shutdown sequence. That's
        # what caused "connection refused" during on_session_end's
        # LLM-based summary call after Ctrl+C: the server was already dead
        # before our Python-level KeyboardInterrupt handling even started.
        # With this isolation, Ctrl+C only interrupts our own process, and
        # we decide when to stop the server (see stop(), still called
        # explicitly via process.terminate() regardless of process group).
        popen_kwargs = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        try:
            self.log_file = open("llama_server.log", "a")
            self.process = subprocess.Popen(
                command,
                stdout=self.log_file,
                stderr=self.log_file,
                **popen_kwargs,
            )
        except Exception:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            raise

        self.alive = True
        log.info("Started successfully")

    def stop(self):

        if not self.process:
            return

        log.info("Stopping subprocess...")

        self.alive = False

        try:
            self.process.terminate()
            self.process.wait(timeout=5)

        except subprocess.TimeoutExpired:
            self.process.kill()

        self.process = None

        if self.log_file:
            self.log_file.close()
            self.log_file = None

        log.info("Stopped")

    def restart(self):

        log.info("Restarting model...")

        self.stop()
        time.sleep(1)
        self.start()

    def is_alive(self) -> bool:

        return (
            self.process is not None
            and self.process.poll() is None
        )

    # HIGH LEVEL API

    def wait_until_ready(self, timeout: int = 120) -> bool:
        """Delegates to Runtime.health_check.wait_until_ready — see that
        module for the two-phase readiness check itself."""
        return wait_until_ready(timeout=timeout)

    def start_for_client(self) -> None:
        self.start()
        self.wait_until_ready(timeout=120)

    def stop_from_cli(self) -> None:
        self.stop()

    def get_session_id(self):
        return self.session_id

    @staticmethod
    def get_instance() -> 'ProcessManager':
        """Get the singleton instance of ProcessManager."""
        if ProcessManager._instance is None:
            raise RuntimeError(
                "ProcessManager not initialised. "
                "Call ProcessManager(model_path=..., llama_cli_path=..., mmproj_path=...) first."
            )
        return ProcessManager._instance
