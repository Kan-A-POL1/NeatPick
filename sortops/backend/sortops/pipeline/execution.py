"""How a policy's action chunks are played, the way Physical AI Studio's "Run model" plays them.

* **Action rate** (``hz``): one action of the queue is sent per tick.
* **Request the next chunk at** (``request_threshold``, 0..1 of the chunk): when the queue holds fewer actions than
  that fraction of a chunk, a new chunk is requested.
* **Synchronous**: the request blocks the control loop (the arm pauses for the inference latency) and the new chunk
  starts from the observation it was computed on. **Asynchronous**: the inference runs beside the loop, which keeps
  playing the queue; the new chunk then skips the actions already sent since its observation (offset).
* New chunks are merged into the remaining queue with a 5-action linear blend (physicalai ``LerpSmoother``), so
  consecutive chunks do not jerk. An empty queue holds the last target.

Same rules as ``physicalai.runtime`` ``SyncExecution`` / ``AsyncExecution`` + ``ChunkedActionQueue(LerpSmoother())``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import Any

import numpy as np

LERP_FRAMES = 5
MODES = ("sync", "async")


def lerp_merge(remaining: np.ndarray, incoming: np.ndarray, duration: int = LERP_FRAMES) -> np.ndarray:
    """physicalai ``LerpSmoother.merge``: blend the overlap from the old queue into the new chunk, then its tail."""
    n_remain = len(remaining)
    if n_remain == 0:
        return incoming.astype(np.float32)
    lerp = min(n_remain, duration)
    weights = np.maximum(1.0 - np.arange(n_remain) / max(lerp, 1), 0.0)[:, np.newaxis]
    n_blend = min(n_remain, len(incoming))
    blended = weights[:n_blend] * remaining[:n_blend] + (1.0 - weights[:n_blend]) * incoming[:n_blend]
    return np.concatenate([blended, incoming[n_blend:]], axis=0).astype(np.float32)


class ChunkPlayer:
    """Action queue + chunk requests for one segment. ``predict(executed)`` returns a (chunk, action_dim) array;
    ``executed`` is the number of actions sent since the previous request (scripted stand-ins use it)."""

    def __init__(self, cfg: dict[str, Any], predict: Callable[[int], np.ndarray], pool: Executor) -> None:
        if cfg["mode"] not in MODES:
            raise ValueError(f"execution mode must be one of {MODES}")
        self.mode: str = cfg["mode"]
        self.threshold = float(cfg["request_threshold"])
        self._predict = predict
        self._pool = pool
        self._queue: deque[np.ndarray] = deque()
        self.chunk_size = 0
        self.pops = 0
        self.requests = 0
        self._pops_at_request = 0
        self._pops_at_async_request = 0
        self._pending: Future[np.ndarray] | None = None
        self._last: np.ndarray | None = None

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def _threshold_count(self) -> int:
        count = int(self.chunk_size * self.threshold)
        # Studio: sync re-infers at least when one action is left, async may wait for an empty queue.
        return max(1, count) if self.mode == "sync" else count

    def _push(self, chunk: np.ndarray, offset: int) -> None:
        incoming = np.asarray(chunk, np.float32)[min(max(offset, 0), len(chunk) - 1) :]
        remaining = np.stack(list(self._queue)) if self._queue else np.empty((0, incoming.shape[1]), np.float32)
        self._queue = deque(lerp_merge(remaining, incoming))

    def _request_blocking(self) -> None:
        executed = self.pops - self._pops_at_request
        self._pops_at_request = self.pops
        chunk = self._predict(executed)
        self.requests += 1
        self.chunk_size = self.chunk_size or len(chunk)
        self._push(chunk, 0)

    def next_action(self) -> np.ndarray | None:
        """The action for this tick (``None`` before the first chunk could be computed)."""
        if self.chunk_size == 0:
            self._request_blocking()  # warm-up: the first chunk is always computed in line
        elif self.mode == "sync":
            if len(self._queue) < self._threshold_count():
                self._request_blocking()
        else:
            if self._pending is not None and self._pending.done():
                chunk = self._pending.result()
                self._pending = None
                self.requests += 1
                self._push(chunk, self.pops - self._pops_at_async_request)
            if self._pending is None and len(self._queue) < self._threshold_count():
                executed = self.pops - self._pops_at_request
                self._pops_at_request = self.pops
                self._pops_at_async_request = self.pops
                self._pending = self._pool.submit(self._predict, executed)
        if self._queue:
            self._last = self._queue.popleft()
            self.pops += 1
        return self._last

    def close(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
