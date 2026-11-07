import pytest

from rag_eval import cache as cache_module
from rag_eval.cache import Cache, retry_delay, with_retry


def test_cache_round_trip_and_key_depends_on_every_part(tmp_path):
    c = Cache(tmp_path / "c.sqlite")
    k1, k2 = Cache.key("answer", "m", "prompt"), Cache.key("answer", "m", "prompt 2")
    assert k1 != k2
    assert c.get(k1) is None
    c.set(k1, {"text": "hi"})
    assert c.get(k1) == {"text": "hi"}


def test_retry_delay_prefers_server_hint():
    assert retry_delay("429 ... 'retryDelay': '23s' ...", attempt=0) == 24.0
    assert retry_delay("503 UNAVAILABLE", attempt=2) == 8.0
    assert retry_delay("503 UNAVAILABLE", attempt=10) == 60.0


def test_with_retry_retries_quota_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr(cache_module.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert with_retry(flaky) == "ok" and len(calls) == 3


def test_with_retry_raises_non_retryable_errors_immediately(monkeypatch):
    monkeypatch.setattr(cache_module.time, "sleep", lambda s: None)
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("400 INVALID_ARGUMENT")

    with pytest.raises(ValueError):
        with_retry(broken)
    assert len(calls) == 1
