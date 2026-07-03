# GlobalHelpers/logger.py
import logging
import os
import contextvars
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
SESSION_DIR = os.path.join(LOG_DIR, "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

# session_id is set once per process (SessionManager) and auto-injected into every log line
_session_id_ctx = contextvars.ContextVar("session_id", default="no-session")

def set_session_id(session_id: str) -> None:
    _session_id_ctx.set(session_id)

class _SessionFilter(logging.Filter):
    def filter(self, record):
        record.session_id = _session_id_ctx.get()
        return True

_FORMAT = "%(asctime)s | %(session_id)s | %(levelname)-8s | %(name)s | %(message)s"
_formatter = logging.Formatter(_FORMAT)

def _build_root_handlers():
    app_handler = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=5_000_000, backupCount=5)
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(_formatter)

    error_handler = RotatingFileHandler(os.path.join(LOG_DIR, "errors.log"), maxBytes=5_000_000, backupCount=5)
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_formatter)

    for h in (app_handler, error_handler, console_handler):
        h.addFilter(_SessionFilter())
    return [app_handler, error_handler, console_handler]

_configured = False

def configure_logging(level=logging.DEBUG):
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(level)
    for h in _build_root_handlers():
        root.addHandler(h)
    _configured = True

def attach_session_file_handler(session_id: str) -> RotatingFileHandler:
    """Call this from on_session_start. Adds a 3rd, per-session file."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(SESSION_DIR, f"session_{session_id}_{date_str}.log")
    handler = logging.FileHandler(path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_formatter)
    handler.addFilter(_SessionFilter())
    logging.getLogger().addHandler(handler)
    return handler

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)