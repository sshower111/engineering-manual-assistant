"""Structured JSON logging of every query (one JSON object per line)."""

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("manual_assistant.queries")


class QueryLogger:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        line = json.dumps({"timestamp": datetime.now(UTC).isoformat(), **record})
        log.info(line)
        if self._path is not None:
            with self._lock, self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
