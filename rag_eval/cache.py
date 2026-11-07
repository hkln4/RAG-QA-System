"""Disk cache and retry for Gemini calls made by the evaluation.

Every embedding, answer and judge score is cached under a hash of its full
input (model, prompt text, contexts), so rerunning an experiment costs no API
calls, and changing anything that reaches the model is automatically a miss.
"""
import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "cache.sqlite"

# quota and transient server errors; anything else is a real failure and is raised
RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "DEADLINE_EXCEEDED")


class Cache:
    def __init__(self, path=CACHE_PATH):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts):
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()

    def get(self, key):
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key, value):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self._conn.commit()


def retry_delay(message, attempt):
    """Seconds to wait: the server's retryDelay when it gives one, else exponential backoff."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", message)
    if match:
        return float(match.group(1)) + 1
    return min(60.0, 2.0 * 2 ** attempt)


def with_retry(fn, *args, max_attempts=8, **kwargs):
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            message = str(e)
            if attempt == max_attempts - 1 or not any(t in message for t in RETRYABLE):
                raise
            delay = retry_delay(message, attempt)
            print(f"  retryable API error, waiting {delay:.0f}s ({message[:80]})", flush=True)
            time.sleep(delay)
