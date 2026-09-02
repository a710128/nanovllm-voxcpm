"""Worker-process liveness tracking and fail-fast propagation.

The async servers in this package drive their model in a spawned worker
process and talk to it over ``multiprocessing`` queues. Those queues carry no
liveness information: if the worker dies (host OOM killer, segfault, uncaught
``torch.cuda.OutOfMemoryError``), ``queue.get()`` simply blocks forever and
every in-flight request hangs.

:class:`WorkerLink` closes that gap. It is deliberately split into two
independent halves:

Detection
    :meth:`WorkerLink.watch` parks a daemon thread on
    ``multiprocessing.connection.wait([process.sentinel])``. The sentinel is a
    pipe whose write end lives in the child; the kernel closes it while
    reaping the child's file descriptors, so the read end becomes readable for
    *any* form of termination without the child having to run a single
    instruction. ``wait()`` is used instead of ``process.join()`` because it
    only observes the sentinel and does not reap the exit status, so it cannot
    race with the ``join()`` calls made during shutdown.

Propagation
    :meth:`WorkerLink.bind` hooks a waiter's future onto a single "worker
    died" future via ``add_done_callback``. Since ``add_done_callback`` fires
    immediately when the target future is already resolved, one call covers
    both cases: in-flight waiters are aborted, and waiters created after the
    worker died fail right away. No separate "is the worker dead" flag has to
    be kept in sync, and a waiter that forgets to bind cannot silently hang --
    it never reaches the worker in the first place.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from multiprocessing.connection import wait
from typing import Any, Callable, TypeVar

__all__ = [
    "STREAM_FAILED",
    "DeathCallback",
    "StreamFailed",
    "WorkerDiedError",
    "WorkerLink",
    "ignore_unretrieved",
]

_T = TypeVar("_T")

DeathCallback = Callable[[BaseException], None]
"""Callback invoked with the death error when the worker process exits."""


class WorkerDiedError(RuntimeError):
    """Raised when the worker subprocess terminates unexpectedly.

    Distinct from a plain ``RuntimeError`` so callers (e.g. HTTP handlers) can
    tell "the backend is gone, retry elsewhere" apart from "your request was
    invalid".
    """


class StreamFailed:
    """Marker pushed into a stream queue when the worker dies mid-stream.

    A stream is normally terminated by ``None``. Reusing ``None`` for the
    failure case would hand the caller a silently truncated waveform, so
    failures get their own distinguishable marker.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<StreamFailed>"


STREAM_FAILED = StreamFailed()
"""Shared :class:`StreamFailed` instance; compare with ``isinstance``."""


def ignore_unretrieved(fut: asyncio.Future[Any]) -> None:
    """Mark ``fut``'s exception as retrieved.

    Futures that may fail without anyone awaiting them would otherwise make
    asyncio log "Future exception was never retrieved" when they are garbage
    collected. Awaiting the future later still raises as usual.

    Args:
        fut: Future whose exception should not be reported at collection time.
    """

    def _consume(done: asyncio.Future[Any]) -> None:
        if not done.cancelled():
            done.exception()

    fut.add_done_callback(_consume)


class WorkerLink:
    """Tracks a worker process and fails everything waiting on it when it dies.

    Args:
        loop: Event loop that owns every future bound to this link.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._dead: asyncio.Future[None] = loop.create_future()
        # Nothing awaits the death future itself; it is only a callback anchor.
        ignore_unretrieved(self._dead)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def watch(self, process: Any, name: str = "worker-watchdog") -> None:
        """Start watching ``process`` for unexpected termination.

        Args:
            process: A started ``multiprocessing.Process``.
            name: Thread name, useful when inspecting stacks.

        Raises:
            RuntimeError: If this link is already watching a process.
        """
        if self._thread is not None:
            raise RuntimeError("WorkerLink is already watching a process")
        self._thread = threading.Thread(
            target=self._wait_for_exit,
            args=(process,),
            name=name,
            daemon=True,
        )
        self._thread.start()

    def _wait_for_exit(self, process: Any) -> None:
        with contextlib.suppress(OSError, ValueError):
            wait([process.sentinel])
        # The loop may already be closed if the interpreter is shutting down.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._fail, process)

    def _fail(self, process: Any) -> None:
        if self._stopping or self._dead.done():
            return
        self._dead.set_exception(WorkerDiedError(f"worker process died unexpectedly (exitcode={process.exitcode})"))

    def mark_stopping(self) -> None:
        """Declare the upcoming worker exit as intentional.

        Call this before asking the worker to shut down so a graceful exit is
        not reported as a crash.
        """
        self._stopping = True

    def join(self, timeout: float | None = None) -> None:
        """Wait for the watchdog thread to finish. No-op if never started."""
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    @property
    def dead(self) -> bool:
        """Whether the worker has been observed to terminate unexpectedly."""
        return self._dead.done()

    def failure(self) -> BaseException:
        """Return the death error.

        Returns:
            The :class:`WorkerDiedError` describing why the worker exited.

        Raises:
            asyncio.InvalidStateError: If the worker is still alive.
            RuntimeError: If the link somehow resolved without an error.
        """
        exc = self._dead.exception()
        if exc is None:  # pragma: no cover - the link is only ever failed
            raise RuntimeError("worker link resolved without an error")
        return exc

    def bind(self, fut: asyncio.Future[_T]) -> asyncio.Future[_T]:
        """Fail ``fut`` if (or as soon as) the worker dies.

        If the worker is already dead the failure is scheduled immediately, so
        the same call both aborts in-flight waiters and rejects new ones. The
        hook is removed once ``fut`` settles, keeping the number of registered
        callbacks proportional to in-flight requests rather than to the total
        number of requests ever made.

        Args:
            fut: Future awaited by a caller that depends on the worker.

        Returns:
            The same future, for convenient chaining.
        """

        def _on_death(dead: asyncio.Future[None]) -> None:
            exc = dead.exception()
            if exc is not None and not fut.done():
                fut.set_exception(exc)

        self._dead.add_done_callback(_on_death)
        fut.add_done_callback(lambda _settled: self._dead.remove_done_callback(_on_death))
        return fut

    def on_death(self, callback: DeathCallback) -> Callable[[asyncio.Future[None]], None]:
        """Register ``callback`` to run when the worker dies.

        Used by waiters that do not block on a future (a stream queue, for
        instance) and therefore cannot use :meth:`bind`.

        Args:
            callback: Invoked with the death error. Runs on the event loop.

        Returns:
            A handle to pass to :meth:`release` when the waiter goes away.
        """

        def _on_death(dead: asyncio.Future[None]) -> None:
            exc = dead.exception()
            if exc is not None:
                callback(exc)

        self._dead.add_done_callback(_on_death)
        return _on_death

    def release(self, handle: Callable[[asyncio.Future[None]], None]) -> None:
        """Unregister a callback previously returned by :meth:`on_death`."""
        self._dead.remove_done_callback(handle)
