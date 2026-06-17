import subprocess
import time
from typing import Optional, TextIO
import SessionManager.session_generator as session_generator


class ProcessManager:
    """
    Persistent llama.cpp subprocess manager.

    Responsibilities:
    - start / stop / restart model

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
        ctx_size: int = 16384,
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
        print("Session id generated as:- " + self.session_id)

        # CHANGED: Mark as initialised so repeated __init__ calls are no-ops.
        self._initialised = True

    # PROCESS LIFECYCLE

    def start(self):

        if self.alive:
            print("[PROCESS] Already running")
            return

        command = [
            self.llama_cli_path,

            # MODEL
            "-m", self.model_path,

            # MULTIMODAL PROJECTOR (REQUIRED FOR VL MODELS)
            "--mmproj", self.mmproj_path,

            # CONTEXT WINDOW
            "-c", str(self.ctx_size),

            # GPU OFFLOAD LAYERS
            "-ngl", str(self.gpu_layers),

            "--cache-prompt",

            "--cache-reuse", "256",

            "--ctx-checkpoints", "64",

            "--parallel", "4",

            "--cont-batching",

            "--host", "127.0.0.1",

            "--port", "8081"
        ]

        print("[PROCESS] Starting llama.cpp subprocess...")

        try:
            self.log_file = open("llama_server.log", "a")
            self.process = subprocess.Popen(
                command,
                stdout=self.log_file,
                stderr=self.log_file,
            )
        except Exception:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            raise

        self.alive = True
        print("[PROCESS] Started successfully")

    def stop(self):

        if not self.process:
            return

        print("[PROCESS] Stopping subprocess...")

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

        print("[PROCESS] Stopped")

    def restart(self):

        print("[PROCESS] Restarting model...")

        self.stop()
        time.sleep(1)
        self.start()

    def is_alive(self) -> bool:

        return (
            self.process is not None
            and self.process.poll() is None
        )

    # HIGH LEVEL API

    def start_for_client(self) -> None:
        self.start()

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
