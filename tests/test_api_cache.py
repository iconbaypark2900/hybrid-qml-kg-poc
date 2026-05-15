"""Unit tests for validation_layer.api_cache."""
from __future__ import annotations

from pathlib import Path

import pytest

from validation_layer.api_cache import (
    cache_get,
    cache_put,
    cached_call,
    clear_cache,
    env_flag,
)


def test_cache_put_get_round_trip(tmp_path: Path) -> None:
    cache_put("test_provider", {"q": "metformin"}, [{"id": 1}], cache_dir=tmp_path)
    out = cache_get("test_provider", {"q": "metformin"}, cache_dir=tmp_path)
    assert out == [{"id": 1}]


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    assert cache_get("nope", {"q": "x"}, cache_dir=tmp_path) is None


def test_cache_key_is_param_dependent(tmp_path: Path) -> None:
    cache_put("p", {"q": "a"}, "value_a", cache_dir=tmp_path)
    cache_put("p", {"q": "b"}, "value_b", cache_dir=tmp_path)
    assert cache_get("p", {"q": "a"}, cache_dir=tmp_path) == "value_a"
    assert cache_get("p", {"q": "b"}, cache_dir=tmp_path) == "value_b"


def test_cached_call_uses_cache_on_second_call(tmp_path: Path) -> None:
    counter = {"n": 0}

    def fetch():
        counter["n"] += 1
        return {"hits": counter["n"]}

    first = cached_call("p", {"q": "x"}, fetch, cache_dir=tmp_path)
    second = cached_call("p", {"q": "x"}, fetch, cache_dir=tmp_path)
    third = cached_call("p", {"q": "x"}, fetch, cache_dir=tmp_path)

    assert first == {"hits": 1}
    assert second == first
    assert third == first
    assert counter["n"] == 1, "fetch_fn should run only on the first call"


def test_cached_call_force_refresh(tmp_path: Path) -> None:
    counter = {"n": 0}

    def fetch():
        counter["n"] += 1
        return {"hits": counter["n"]}

    cached_call("p", {"q": "x"}, fetch, cache_dir=tmp_path)
    refreshed = cached_call("p", {"q": "x"}, fetch, cache_dir=tmp_path,
                            force_refresh=True)
    assert refreshed == {"hits": 2}


def test_clear_cache_removes_files(tmp_path: Path) -> None:
    for i in range(3):
        cache_put("p", {"q": str(i)}, i, cache_dir=tmp_path)
    n = clear_cache("p", cache_dir=tmp_path)
    assert n == 3
    assert cache_get("p", {"q": "0"}, cache_dir=tmp_path) is None


def test_env_flag_recognises_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CT_API_CACHE", raising=False)
    monkeypatch.delenv("VALIDATION_API_CACHE_ALL", raising=False)
    assert env_flag("clinicaltrials.gov") is False

    monkeypatch.setenv("CT_API_CACHE", "1")
    assert env_flag("clinicaltrials.gov") is True


def test_env_flag_global_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALIDATION_API_CACHE_ALL", "1")
    assert env_flag("clinicaltrials.gov") is True
    assert env_flag("pubmed") is True
    assert env_flag("open_targets") is True


def test_corrupt_cache_file_returns_none(tmp_path: Path) -> None:
    # Write garbage at the path the cache key would compute
    from validation_layer.api_cache import _path
    p = _path("p", {"q": "junk"}, tmp_path)
    p.write_text("not valid json {{{")
    assert cache_get("p", {"q": "junk"}, cache_dir=tmp_path) is None
