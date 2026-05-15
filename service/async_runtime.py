"""Thread + process pools for wrapping the sync sklearn/Qiskit library code.

The kept library (kg_embedder, ClassicalLinkPredictor, QMLLinkPredictor) is all
sync. Calling it directly from an async route blocks the event loop. This
module provides typed wrappers so route handlers can stay async-clean.

Routing rule:
  - run_io_bound: sklearn predict, joblib load, file I/O, fast simulator-quantum
  - run_cpu_bound: training, large CV runs (out of scope for v1 service)
  - never sync: IBM hardware predictions go through service/jobs.py instead
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


class _Pools:
    thread: Optional[ThreadPoolExecutor] = None
    process: Optional[ProcessPoolExecutor] = None


_pools = _Pools()


def init_pools(thread_workers: int = 16, process_workers: int = 4) -> None:
    """Initialize pools. Idempotent — safe to call multiple times in tests."""
    if _pools.thread is None:
        _pools.thread = ThreadPoolExecutor(
            max_workers=thread_workers, thread_name_prefix="hetqml-io"
        )
    if _pools.process is None:
        _pools.process = ProcessPoolExecutor(max_workers=process_workers)


def shutdown_pools(wait: bool = True) -> None:
    """Tear down pools — used by lifespan shutdown and tests."""
    if _pools.thread is not None:
        _pools.thread.shutdown(wait=wait)
        _pools.thread = None
    if _pools.process is not None:
        _pools.process.shutdown(wait=wait)
        _pools.process = None


async def run_io_bound(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """For sklearn predict, joblib load, file I/O — fast, GIL-released paths."""
    if _pools.thread is None:
        init_pools()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pools.thread, lambda: fn(*args, **kwargs))


async def run_cpu_bound(fn: Callable[..., T], *args: Any) -> T:
    """For training / heavy CV — must be picklable. Args only, no kwargs."""
    if _pools.process is None:
        init_pools()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pools.process, fn, *args)
