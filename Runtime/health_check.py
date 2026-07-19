"""
Runtime/health_check.py

Polls a local OpenAI-compatible completion server until it's actually
ready to generate not just until the HTTP port is open. Split out of
ProcessManager, which previously mixed subprocess lifecycle management
(start/stop/restart) with this unrelated HTTP-polling responsibility.
"""

import time
import requests
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


def wait_until_ready(host: str = "127.0.0.1", port: int = 8081, timeout: int = 120) -> bool:
    """
    Two-phase readiness check:
      Phase 1 — wait for the HTTP server to bind (fast, ~1-2s).
      Phase 2 — send a minimal real generation request. /health alone
                returns 200 as soon as the port opens, even while model
                weights are still loading into VRAM, so it isn't a
                reliable readiness signal on its own.
    """
    base_url = f"http://{host}:{port}"
    print("[PROCESS] Waiting for model to load", end="", flush=True)
    deadline = time.time() + timeout

    # Phase 1: wait for HTTP server to bind
    while time.time() < deadline:
        try:
            requests.get(f"{base_url}/health", timeout=1)
            break
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)

    # Phase 2: wait for model to actually be ready for generation
    while time.time() < deadline:
        try:
            r = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "local",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,   # tiny — just enough to confirm generation works
                    "temperature": 0.0,
                },
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:  # got actual generated text -> model ready
                    print(" ready.")
                    return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)

    print(" timed out — proceeding anyway.")
    return False
