from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class VectorDBClient(ABC):

    @abstractmethod
    def add(self, id: str, text: str, metadata: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5, where: dict | None = None) -> list[dict]:
        pass

    @abstractmethod
    def get(self, id: str) -> dict | None:
        pass

    @abstractmethod
    def update(self, id: str, text: str | None = None, metadata: dict | None = None) -> bool:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @staticmethod
    def new_id() -> str:
        return f"m_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().isoformat()


