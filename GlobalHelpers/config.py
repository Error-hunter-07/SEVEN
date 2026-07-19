"""
GlobalHelpers/config.py

Centralized environment variable loading with validation.
All required env vars declared in one place, validated at startup.
Fail-fast with clear error messages instead of cryptic NoneType failures later.
"""

import os
import sys
from dataclasses import dataclass
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable config holder — all values set once at startup."""
    
    # Vector DB (ChromaDB)
    default_persist_dir: str
    default_embedding_model: str
    
    # PostgreSQL
    db_user: str
    db_password: str
    db_name: str
    db_host: str
    db_port: str
    
    # LLM
    llm_model: str
    llm_model_path: str
    llm_cli_path: str
    mmproj_path: str

    @property
    def db_connection_string(self) -> str:
        """Build a proper connection string with URL-encoded credentials."""
        user = quote_plus(self.db_user)
        pwd = quote_plus(self.db_password)
        return f"postgresql://{user}:{pwd}@{self.db_host}:{self.db_port}/{self.db_name}"


# Required vars — if any are missing, we fail immediately at import time
_REQUIRED = [
    "DEFAULT_PERSIST_DIR",
    "DEFAULT_EMBEDDING_MODEL",
    "DB_USER",
    "DB_PASSWORD",
    "DB_NAME",
    "LLM_MODEL",
    "LLM_MODEL_PATH",
    "LLM_CLI_PATH",
    "MMPROJ_PATH",
]

# Optional vars with sensible defaults
_DEFAULTS = {
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
}


def _load_settings() -> Settings:
    """Load and validate all required env vars. Exit early if any are missing."""
    missing = [v for v in _REQUIRED if not os.getenv(v)]
    if missing:
        msg = (
            "FATAL: missing required environment variable(s): "
            + ", ".join(sorted(missing))
            + "\nCheck your .env file against .env.example"
        )
        sys.exit(msg)
    
    return Settings(
        default_persist_dir=os.getenv("DEFAULT_PERSIST_DIR"),
        default_embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL"),
        db_user=os.getenv("DB_USER"),
        db_password=os.getenv("DB_PASSWORD"),
        db_name=os.getenv("DB_NAME"),
        db_host=os.getenv("DB_HOST", _DEFAULTS["DB_HOST"]),
        db_port=os.getenv("DB_PORT", _DEFAULTS["DB_PORT"]),
        llm_model=os.getenv("LLM_MODEL"),
        llm_model_path=os.getenv("LLM_MODEL_PATH"),
        llm_cli_path=os.getenv("LLM_CLI_PATH"),
        mmproj_path=os.getenv("MMPROJ_PATH"),
    )


# Singleton — loaded once, failures cause immediate exit
settings = _load_settings()
