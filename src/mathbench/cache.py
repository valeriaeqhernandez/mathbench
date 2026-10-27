"""A tiny SQLite cache for model responses.

API calls cost money and time. With a cache, re-running an experiment after fixing a
grading bug is free and instant, and results are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, text TEXT)")

    @staticmethod
    def key(base_url: str, payload: dict, sample: int) -> str:
        blob = json.dumps({"url": base_url, "payload": payload, "sample": sample}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        row = self._db.execute("SELECT text FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, text: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, text))
        self._db.commit()
