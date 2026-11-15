"""The chunk player follows Physical AI Studio's sync / async execution rules."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sortops.pipeline.execution import ChunkPlayer, lerp_merge

CHUNK = 10


def ramp(start: float) -> np.ndarray:
    return np.repeat(np.arange(start, start + CHUNK, dtype=np.float32)[:, None], 6, axis=1)


def test_sync_requests_when_the_queue_falls_below_the_threshold() -> None:
    calls: list[int] = []

    def predict(executed: int) -> np.ndarray:
        calls.append(executed)
        return ramp(100.0 * len(calls))

    with ThreadPoolExecutor(1) as pool:
        player = ChunkPlayer({"mode": "sync", "request_threshold": 0.5}, predict, pool)
        for _ in range(6):
            player.next_action()
    # Warm-up, then a new chunk once fewer than 5 of 10 actions remain (after 6 pops).
    assert calls == [0]
    with ThreadPoolExecutor(1) as pool:
        player = ChunkPlayer({"mode": "sync", "request_threshold": 0.5}, predict, pool)
        for _ in range(7):
            player.next_action()
    assert len(calls) == 3 and calls[-1] == 6 and player.requests == 2


@pytest.mark.parametrize(("threshold", "requests_after_25"), [(0.0, 3), (0.5, 5), (0.9, 13)])
def test_threshold_changes_how_often_chunks_are_requested(threshold: float, requests_after_25: int) -> None:
    with ThreadPoolExecutor(1) as pool:
        player = ChunkPlayer({"mode": "sync", "request_threshold": threshold}, lambda _e: ramp(0.0), pool)
        for _ in range(25):
            player.next_action()
    assert player.requests == requests_after_25


def test_async_keeps_playing_and_skips_actions_sent_during_inference() -> None:
    def slow(_executed: int) -> np.ndarray:
        time.sleep(0.05)
        return ramp(1000.0)

    with ThreadPoolExecutor(1) as pool:
        player = ChunkPlayer({"mode": "async", "request_threshold": 0.5}, lambda _e: ramp(0.0), pool)
        first = [float(player.next_action()[0]) for _ in range(6)]  # type: ignore[index]
        assert first == [0, 1, 2, 3, 4, 5]
        player._predict = slow  # noqa: SLF001
        sent = [float(player.next_action()[0]) for _ in range(3)]  # type: ignore[index]
        assert sent == [6, 7, 8], "the loop must not wait for the async inference"
        time.sleep(0.1)
        after = float(player.next_action()[0])  # type: ignore[index]
    # The new chunk was requested at pop 6; 3 more actions went out, so it starts 3 actions in, blended.
    assert 9 <= after <= 1003


def test_lerp_merge_blends_then_takes_the_new_chunk() -> None:
    merged = lerp_merge(np.zeros((3, 1), np.float32), np.ones((8, 1), np.float32))
    assert merged[0, 0] == 0.0 and merged[-1, 0] == 1.0 and len(merged) == 8
