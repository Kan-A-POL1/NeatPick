"""Safety between the policies and the motors.

* Every target is clamped to soft joint limits and rate-limited (``max_step`` per control tick).
* **E-stop** (UI button, Space key, watchdog, overheating) is latched: the running cycle is cancelled, the arm
  holds its current position with torque on, and nothing moves until an operator resets it.
* **Torque off** cuts the motors (the arm falls): last resort, explicit operator action only.
* **Shadow mode** (default at start): models run and their targets are shown, but nothing reaches the motors.
* **Watchdog**: in live mode a cycle needs an operator UI sending heartbeats; silence → e-stop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import numpy as np
from loguru import logger


class Mode(StrEnum):
    SHADOW = "shadow"
    LIVE = "live"


class SafetyGuard:
    def __init__(self, settings: Callable[[], dict[str, Any]]) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self.gate = threading.Lock()
        """Held around every motor command; an e-stop takes it, so no command can slip in after the hold."""
        self.mode = Mode.SHADOW
        self.estopped = False
        self.estop_reason = ""
        self.torque = True
        self._last_heartbeat = 0.0
        self.clamped_ticks = 0
        self._listeners: list[Callable[[str], None]] = []
        self._temps: deque[tuple[float, np.ndarray]] = deque()
        self.temp_median: np.ndarray | None = None
        """Per-motor median temperature over the last ``temp_window_s`` (what the overheat stop compares)."""
        self.temp_history: deque[dict[str, object]] = deque(maxlen=600)
        """One sample per second for the UI: time, raw and median temperature per motor."""
        self._last_history = 0.0

    def on_estop(self, listener: Callable[[str], None]) -> None:
        self._listeners.append(listener)

    def heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def heartbeat_age(self) -> float:
        return time.monotonic() - self._last_heartbeat if self._last_heartbeat else float("inf")

    def set_mode(self, mode: Mode) -> None:
        with self._lock:
            if mode == Mode.LIVE and self.estopped:
                raise RuntimeError("Reset the e-stop before going live")
            self.mode = mode

    def trigger(self, reason: str) -> None:
        with self.gate, self._lock:
            already = self.estopped
            self.estopped = True
            self.estop_reason = reason
        if not already:
            logger.warning("E-STOP: {}", reason)
        for listener in self._listeners:
            try:
                listener(reason)
            except Exception:  # noqa: BLE001 - every listener must get its chance
                logger.exception("e-stop listener failed")

    def reset(self) -> None:
        with self._lock:
            self.estopped = False
            self.estop_reason = ""
            # Back to shadow: going live again is a deliberate operator choice.
            self.mode = Mode.SHADOW

    def may_move(self) -> bool:
        return self.mode == Mode.LIVE and not self.estopped and self.torque

    def filter(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Clamp ``target`` to the soft limits and to at most ``max_step`` away from ``current`` per joint."""
        cfg = self._settings()
        low = np.asarray(cfg["soft_min"], np.float32)
        high = np.asarray(cfg["soft_max"], np.float32)
        step = float(cfg["max_step"])
        bounded = np.clip(np.asarray(target, np.float32), low, high)
        limited = np.clip(bounded, current - step, current + step)
        if not np.allclose(limited, target, atol=1e-3):
            self.clamped_ticks += 1
        return limited.astype(np.float32)

    def check(self, motor_temps: np.ndarray, cycle_running: bool) -> None:
        """Periodic checks: motor temperature and, in live mode during a cycle, the operator heartbeat."""
        cfg = self._settings()
        if motor_temps.size:
            self._check_temperature(np.asarray(motor_temps, np.float32), cfg)
        if (
            cycle_running
            and self.mode == Mode.LIVE
            and cfg["watchdog_enabled"]
            and self.heartbeat_age() > cfg["watchdog_s"]
        ):
            self.trigger(f"operator UI silent for {self.heartbeat_age():.1f} s (watchdog)")

    def _check_temperature(self, temps: np.ndarray, cfg: dict[str, Any]) -> None:
        """Overheat stop on the per-motor median over ``temp_window_s``: one noisy read cannot stop the cell."""
        now = time.monotonic()
        window = float(cfg["temp_window_s"])
        self._temps.append((now, temps))
        while self._temps and now - self._temps[0][0] > window:
            self._temps.popleft()
        median = np.median(np.stack([t for _, t in self._temps]), axis=0)
        self.temp_median = median
        if now - self._last_history >= 1.0:
            self._last_history = now
            self.temp_history.append(
                {"t": time.time(), "raw": [round(float(v), 1) for v in temps], "median": [round(float(v), 1) for v in median]}
            )
        # Wait for half a window of samples so a single read at start-up cannot trip it.
        if now - self._temps[0][0] < window / 2:
            return
        hottest = int(np.argmax(median))
        if median[hottest] >= cfg["max_motor_temp_c"]:
            self.trigger(
                f"motor {hottest + 1} median {median[hottest]:.0f} °C over {window:.0f} s ≥ {cfg['max_motor_temp_c']:.0f} °C"
            )
