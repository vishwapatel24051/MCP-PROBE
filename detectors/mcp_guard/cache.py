"""Small sqlite cache for Stage II scores and Stage III verdicts.

Keys are sha256(tag + text), where the tag names the model/prompt/settings, so
threshold sweeps and reruns don't redo neural inference or pay for API calls twice.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class ResultCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()

    @staticmethod
    def key(tag: str, text: str) -> str:
        return hashlib.sha256(f"{tag}\x00{text}".encode("utf-8")).hexdigest()

    def get(self, tag: str, text: str):
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k = ?", (self.key(tag, text),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, tag: str, text: str, value) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (self.key(tag, text), json.dumps(value))
            )
            self._db.commit()
