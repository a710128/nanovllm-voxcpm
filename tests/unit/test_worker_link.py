"""Unit tests for :mod:`nanovllm_voxcpm.utils.worker_link`.

Covers the two halves of :class:`WorkerLink` separately:

- propagation (:meth:`bind` / :meth:`on_death`) with plain futures, no
  subprocess involved;
- detection (:meth:`watch`) against a real spawned process that is SIGKILLed,
  which is the scenario the OOM killer produces.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time

import multiprocessing as mp
import pytest

from nanovllm_voxcpm.utils.worker_link import (
    STREAM_FAILED,
    StreamFailed,
    WorkerDiedError,
    WorkerLink,
)


def _sleep_forever() -> None:
    """Target for the spawned process used by the detection test."""
    time.sleep(3600)


def _pending_callbacks(link: WorkerLink) -> int:
    """Callbacks registered on the death future, minus the internal one."""
    return len(link._dead._callbacks) - 1


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


async def _bind_fails_in_flight_waiters() -> None:
    loop = asyncio.get_running_loop()
    link = WorkerLink(loop)
    waiters = [link.bind(loop.create_future()) for _ in range(3)]

    assert _pending_callbacks(link) == 3
    assert not any(fut.done() for fut in waiters)

    link._fail(_FakeProcess(exitcode=-9))

    for fut in waiters:
        with pytest.raises(WorkerDiedError, match="exitcode=-9"):
            await fut


def test_bind_fails_in_flight_waiters():
    asyncio.run(_bind_fails_in_flight_waiters())


async def _bind_rejects_waiters_created_after_death() -> None:
    loop = asyncio.get_running_loop()
    link = WorkerLink(loop)
    link._fail(_FakeProcess(exitcode=1))

    # add_done_callback fires immediately on an already-resolved future, so no
    # extra "is the worker dead" check is needed in submit().
    late = link.bind(loop.create_future())
    with pytest.raises(WorkerDiedError):
        await asyncio.wait_for(late, timeout=1.0)


def test_bind_rejects_waiters_created_after_death():
    asyncio.run(_bind_rejects_waiters_created_after_death())


async def _bind_unhooks_settled_waiters() -> None:
    loop = asyncio.get_running_loop()
    link = WorkerLink(loop)

    for _ in range(5):
        fut = link.bind(loop.create_future())
        fut.set_result("ok")
        await asyncio.sleep(0)  # done-callbacks run on the next loop iteration
        # Registered callbacks track in-flight requests, not cumulative ones.
        assert _pending_callbacks(link) == 0


def test_bind_unhooks_settled_waiters():
    asyncio.run(_bind_unhooks_settled_waiters())


async def _bind_leaves_already_settled_waiters_alone() -> None:
    loop = asyncio.get_running_loop()
    link = WorkerLink(loop)

    fut = link.bind(loop.create_future())
    fut.set_result("done before the crash")
    await asyncio.sleep(0)

    link._fail(_FakeProcess(exitcode=-9))
    await asyncio.sleep(0)

    assert await fut == "done before the crash"


def test_bind_leaves_already_settled_waiters_alone():
    asyncio.run(_bind_leaves_already_settled_waiters_alone())


async def _on_death_notifies_and_can_be_released() -> None:
    loop = asyncio.get_running_loop()
    link = WorkerLink(loop)
    stream: asyncio.Queue[object] = asyncio.Queue()

    handle = link.on_death(lambda _exc: stream.put_nowait(STREAM_FAILED))
    released: list[object] = []
    released_handle = link.on_death(lambda _exc: released.append(object()))
    link.release(released_handle)

    link._fail(_FakeProcess(exitcode=-9))
    await asyncio.sleep(0)

    assert isinstance(await stream.get(), StreamFailed)
    assert released == []
    link.release(handle)  # releasing after the fact must not raise


def test_on_death_notifies_and_can_be_released():
    asyncio.run(_on_death_notifies_and_can_be_released())


async def _mark_stopping_suppresses_failure() -> None:
    loop = asyncio.get_running_loop()
    link = WorkerLink(loop)
    fut = link.bind(loop.create_future())

    link.mark_stopping()
    link._fail(_FakeProcess(exitcode=0))
    await asyncio.sleep(0)

    assert not link.dead
    assert not fut.done()
    fut.cancel()


def test_mark_stopping_suppresses_failure():
    asyncio.run(_mark_stopping_suppresses_failure())


async def _failure_reports_exitcode() -> None:
    loop = asyncio.get_running_loop()
    link = WorkerLink(loop)

    assert not link.dead
    link._fail(_FakeProcess(exitcode=-9))

    assert link.dead
    assert isinstance(link.failure(), WorkerDiedError)
    assert "exitcode=-9" in str(link.failure())


def test_failure_reports_exitcode():
    asyncio.run(_failure_reports_exitcode())


class _FakeProcess:
    """Stand-in for ``multiprocessing.Process`` in propagation-only tests."""

    def __init__(self, exitcode: int) -> None:
        self.exitcode = exitcode


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


async def _watch_detects_sigkill() -> None:
    loop = asyncio.get_running_loop()
    process = mp.get_context("spawn").Process(target=_sleep_forever, daemon=True)
    process.start()
    try:
        link = WorkerLink(loop)
        link.watch(process)
        waiter = link.bind(loop.create_future())

        # SIGKILL leaves the child no chance to run cleanup code; detection
        # relies on the kernel closing the sentinel pipe.
        os.kill(process.pid, signal.SIGKILL)

        with pytest.raises(WorkerDiedError, match="exitcode=-9"):
            await asyncio.wait_for(waiter, timeout=10.0)
        link.join(timeout=5.0)
    finally:
        if process.is_alive():  # pragma: no cover - cleanup guard
            process.kill()
        process.join(timeout=5.0)


def test_watch_detects_sigkill():
    asyncio.run(_watch_detects_sigkill())


async def _watch_ignores_expected_shutdown() -> None:
    loop = asyncio.get_running_loop()
    process = mp.get_context("spawn").Process(target=_sleep_forever, daemon=True)
    process.start()
    try:
        link = WorkerLink(loop)
        link.watch(process)
        link.mark_stopping()

        process.terminate()
        await asyncio.to_thread(process.join, 10.0)
        link.join(timeout=5.0)
        await asyncio.sleep(0.05)

        assert not link.dead
    finally:
        if process.is_alive():  # pragma: no cover - cleanup guard
            process.kill()
        process.join(timeout=5.0)


def test_watch_ignores_expected_shutdown():
    asyncio.run(_watch_ignores_expected_shutdown())


async def _watch_rejects_double_registration() -> None:
    loop = asyncio.get_running_loop()
    process = mp.get_context("spawn").Process(target=_sleep_forever, daemon=True)
    process.start()
    try:
        link = WorkerLink(loop)
        link.mark_stopping()
        link.watch(process)
        with pytest.raises(RuntimeError, match="already watching"):
            link.watch(process)
    finally:
        process.kill()
        process.join(timeout=5.0)


def test_watch_rejects_double_registration():
    asyncio.run(_watch_rejects_double_registration())
