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

    CHANGED (background mini-LLM support): this class is now generic
    across ROLES instead of being a single global singleton. Previously
    there was only ever one llama-server process (implicitly "the" model),
    so a hard singleton via __new__ made sense. Now there can be a "main"
    chat model AND a "background" mini-LLM (see Runtime/main_process.py
    and Runtime/background_process.py — each owns its role's launch
    config and imports this class), each needing its own persistent
    instance. The singleton guard is now a registry keyed by `role`: at
    most one ProcessManager per role, same protection as before against
    accidental double-instantiation, just per-role instead of global.

    To create/retrieve a role's instance: ProcessManager(role=..., model_path=..., ...)
    — safe to call more than once for the same role; later calls with
    the same role just return the cached instance (extra kwargs on a
    later call are ignored, same as the old singleton behavior).
    To retrieve without risking an accidental fresh construction:
    ProcessManager.get_instance(role=...).
    """

    _instances: dict[str, 'ProcessManager'] = {}

    def __new__(cls, role: str = "main", *args, **kwargs):
        if role in cls._instances:
            return cls._instances[role]
        instance = super().__new__(cls)
        cls._instances[role] = instance
        return instance

    def __init__(
        self,
        role: str = "main",
        model_path: str = None,
        mmproj_path: str = None,
        llama_cli_path: str = None,
        ctx_size: int = 32768,
        gpu_layers: int = 999,
        port: int = 8081,
        threads: Optional[int] = None,
        device: Optional[str] = None,
    ):
        # CHANGED: Guard against __init__ being called again on the already-
        # initialised instance for this role (Python calls __init__ every
        # time even when __new__ returns an existing instance).
        if getattr(self, "_initialised", False):
            return

        self.role = role
        self.model_path = model_path
        self.llama_cli_path = llama_cli_path
        self.mmproj_path = mmproj_path
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.port = port
        self.threads = threads
        self.device = device

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
            log.info("[%s] Already running", self.role)
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

            "--port", str(self.port)
        ]

        # CHANGED: --threads is only meaningful for the CPU-bound
        # background role today (the main model is fully GPU-offloaded
        # via -ngl 999, so llama.cpp's default thread count is fine for
        # it) — see Runtime/background_process.py for why this role
        # deliberately doesn't grab every CPU thread on the machine.
        if self.threads is not None:
            command += ["--threads", str(self.threads)]

        # CHANGED: -ngl 0 alone does NOT fully keep a model off the GPU —
        # per llama.cpp's own docs, "The GPU may still be used to
        # accelerate some parts of the computation even when using the
        # -ngl 0 option" (KV-cache and compute buffers can still land on
        # the GPU device even with zero layers offloaded). This is what
        # was causing the background role to still show up in VRAM
        # (4.1GB -> 4.9GB) despite gpu_layers=0. --device none is the
        # flag that actually disables GPU acceleration entirely for a
        # process; see Runtime/background_process.py, which passes
        # device="none" for exactly this reason. Left unset (None) for
        # the main role, which should keep using the GPU as before.
        if self.device is not None:
            command += ["--device", self.device]

        log.info("[%s] Starting llama.cpp subprocess...", self.role)

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
            log_path = f"llama_server_{self.role}.log"
            self.log_file = open(log_path, "a")
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
        log.info("[%s] Started successfully", self.role)

    def stop(self):

        if not self.process:
            return

        log.info("[%s] Stopping subprocess...", self.role)

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

        log.info("[%s] Stopped", self.role)

    def restart(self):

        log.info("[%s] Restarting model...", self.role)

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
        module for the two-phase readiness check itself.

        CHANGED: now passes self.port through explicitly. Previously this
        relied on wait_until_ready's own hardcoded default (8081), which
        was harmless when only one role/port ever existed but would
        silently poll the WRONG port for the background role otherwise.
        """
        return wait_until_ready(port=self.port, timeout=timeout)

    def start_for_client(self) -> None:
        self.start()
        self.wait_until_ready(timeout=120)

    def stop_from_cli(self) -> None:
        self.stop()

    def get_session_id(self):
        return self.session_id

    @staticmethod
    def get_instance(role: str = "main") -> 'ProcessManager':
        """Get the existing instance for a given role."""
        if role not in ProcessManager._instances:
            raise RuntimeError(
                f"ProcessManager for role={role!r} not initialised. "
                f"Call ProcessManager(role={role!r}, model_path=..., "
                f"llama_cli_path=..., ...) first — see Runtime/main_process.py "
                f"and Runtime/background_process.py."
            )
        return ProcessManager._instances[role]


def bootstrap_all_models() -> 'ProcessManager':
    """
    Starts both model roles this app uses and returns the main model's
    ProcessManager (the one callers actually need to hang onto — e.g.
    for .session_id — since the background role is fire-and-forget).

    MODULARITY: Runtime/main_process.py and Runtime/background_process.py
    each own their role's launch configuration (model path, port,
    context size, GPU layers, thread count) and sequencing — this
    function just wires the two together in the right order:
      1. Main model starts and BLOCKS until ready (the user is waiting
         on this one directly).
      2. Background model starts on a daemon thread and does NOT block
         (see Runtime/background_process.py — it's fine for it to still
         be loading when the first chat turn happens; nothing needs it
         until the first extraction/summarization call).

    NOTE ON IMPORT DIRECTION: main_process.py and background_process.py
    both import ProcessManager FROM this module (they need the class to
    build their role's instance), so this module can't import them back
    at module load time without creating a circular import. The imports
    below are deliberately deferred to inside this function — by the
    time this function is actually CALLED (from LLMEngine/llm_client.py's
    bootstrap block), this module has already finished loading, so the
    cycle isn't a problem at call time, only at import time.
    """
    import Runtime.main_process as main_process
    import Runtime.background_process as background_process

    main_pm = main_process.start_blocking()
    background_process.start_nonblocking()
    return main_pm