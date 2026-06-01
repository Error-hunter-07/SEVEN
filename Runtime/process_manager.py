import subprocess
import time
from typing import Optional, TextIO


class ProcessManager:
    """
    Persistent llama.cpp subprocess manager.

    Responsibilities:
    - start / stop / restart model
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        llama_cli_path: str,
        ctx_size: int = 16384,
        gpu_layers: int = 999,
    ):

        self.model_path = model_path
        self.llama_cli_path = llama_cli_path
        self.mmproj_path = mmproj_path
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers

        self.process: Optional[subprocess.Popen] = None
        self.log_file: Optional[TextIO] = None

        self.alive = False

    # PROCESS LIFECYCLE

    def start(self):

        if self.alive:
            print("[PROCESS] Already running")
            return

        command = [
                self.llama_cli_path,

                # MODEL
                "-m", self.model_path,

                # MULTIMODAL PROJECTOR (REQUIRED FOR QWEN3-VL)
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
