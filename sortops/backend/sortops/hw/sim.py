"""Simulated cell: a stream of bricks (some defective), scripted stand-ins for the three ACT gestures, and
synthetic camera images rendered from the arm pose. It lets the whole loop run end to end without the robot:
image → usability → anomaly score → decision → place, with the same code paths as the real cell.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from sortops.hw.robot import RobotState

# Keyframes (normalized SO-101 units) and the seconds spent reaching each one.
_REST = [0.0, -95.0, 95.0, 60.0, 0.0, 2.0]
SCRIPTS: dict[str, list[tuple[list[float], float]]] = {
    "pick_present": [
        ([0.0, 10.0, 20.0, 40.0, 0.0, 60.0], 1.6),
        ([0.0, 25.0, 10.0, 45.0, 0.0, 60.0], 0.9),
        # Closes to 0 like the recorded demos: a brick stalls the jaws at 3 (SimRobot), the grip is read from the load.
        ([0.0, 25.0, 10.0, 45.0, 0.0, 0.0], 0.6),
        ([0.0, -30.0, 40.0, -10.0, -55.0, 0.0], 1.8),
        ([0.0, -30.0, 40.0, -10.0, -55.0, 0.0], 1.2),
        ([0.0, -30.0, 40.0, -10.0, 57.0, 0.0], 1.6),
        ([0.0, -30.0, 40.0, -10.0, 57.0, 0.0], 1.2),
    ],
    "place_good": [
        ([60.0, 0.0, 20.0, 40.0, 0.0, 5.0], 1.8),
        ([60.0, 0.0, 20.0, 40.0, 0.0, 70.0], 0.6),
        (_REST, 1.8),
    ],
    "place_bad": [
        ([-60.0, 0.0, 20.0, 40.0, 0.0, 5.0], 1.8),
        ([-60.0, 0.0, 20.0, 40.0, 0.0, 70.0], 0.6),
        (_REST, 1.8),
    ],
    "reinspect": [
        ([0.0, -20.0, 30.0, -20.0, 0.0, 5.0], 1.5),
        ([0.0, -30.0, 40.0, -10.0, -55.0, 5.0], 1.5),
    ],
}

# No red or orange: the crop excludes the gripper and plate hues (sortops.pipeline.brick_crop).
BRICK_COLORS = [(140, 70, 200), (30, 110, 220), (240, 190, 20), (40, 160, 70)]


@dataclass
class Piece:
    index: int
    defective: bool
    color: tuple[int, int, int]
    defect_face: str
    """Face carrying the defect mark: ``a`` (before the wrist turn) or ``b`` (after)."""


class SimWorld:
    """Bricks waiting on the table; the arm state is pulled from ``read_state`` to render the cameras."""

    def __init__(self, read_state: Callable[[], RobotState], seed: int = 7, defect_rate: float = 0.4) -> None:
        self._read_state = read_state
        self._rng = random.Random(seed)
        self._defect_rate = defect_rate
        self._lock = threading.Lock()
        self._count = 0
        self.piece = self._next_piece()
        self.remaining = 1_000_000
        self.held = False
        """The gripper holds the brick (SimRobot stalled on it)."""
        self.on_table = True
        """A brick waits at the pick spot (False once picked, until placed() brings the next one)."""
        self.supply = True
        """False: the table stays empty (tests the no-brick retry path)."""

    def available(self) -> bool:
        return self.supply and self.on_table

    def grip(self, held: bool) -> None:
        with self._lock:
            self.held = held
            if held:
                self.on_table = False

    def _next_piece(self) -> Piece:
        self._count += 1
        defective = self._rng.random() < self._defect_rate
        return Piece(
            index=self._count,
            defective=defective,
            color=self._rng.choice(BRICK_COLORS),
            defect_face=self._rng.choice(["a", "b"]),
        )

    def placed(self) -> None:
        with self._lock:
            self.piece = self._next_piece()
            self.on_table = True

    # Rendering ---------------------------------------------------------------------------------------------

    def render(self, camera: str) -> np.ndarray:
        state = self._read_state()
        with self._lock:
            piece = self.piece
        if camera == "front":
            return self._render_front(state, piece)
        if camera == "wrist":
            return self._render_wrist(state, piece)
        return self._render_overhead(state, piece)

    @staticmethod
    def _base(width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        image = np.asarray(top, np.float32) * (1 - ramp) + np.asarray(bottom, np.float32) * ramp
        return np.repeat(image, width, axis=1).astype(np.uint8)

    def _draw_brick(self, image: np.ndarray, center: tuple[int, int], scale: float, piece: Piece, face: str) -> None:
        cx, cy = center
        w, h = int(120 * scale), int(70 * scale)
        x0, y0 = cx - w // 2, cy - h // 2
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), piece.color, -1)
        darker = tuple(int(c * 0.7) for c in piece.color)
        for i in range(4):
            sx = x0 + int((i + 0.5) * w / 4)
            cv2.ellipse(image, (sx, y0), (int(11 * scale), int(5 * scale)), 0, 0, 360, darker, -1)
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), darker, max(1, int(2 * scale)))
        if piece.defective and piece.defect_face == face:
            # A marker scribble: the kind of defect the team draws on bricks.
            points = np.asarray(
                [[x0 + int(w * f), y0 + int(h * g)] for f, g in ((0.2, 0.3), (0.45, 0.75), (0.6, 0.25), (0.8, 0.7))],
                np.int32,
            )
            cv2.polylines(image, [points], False, (20, 20, 20), max(2, int(6 * scale)))

    def _holding(self, state: RobotState) -> bool:
        return self.held and float(state.ee_xyz[2]) > 0.05

    def _render_front(self, state: RobotState, piece: Piece) -> np.ndarray:
        image = self._base(640, 480, (58, 62, 70), (120, 118, 112))
        cv2.rectangle(image, (0, 360), (640, 480), (150, 140, 125), -1)
        if self._holding(state):
            x, y, z = (float(v) for v in state.ee_xyz)
            center = (int(320 - y * 900), int(400 - z * 1200))
            scale = 1.0 + (x - 0.25) * 2.0
            face = "a" if float(state.joints[4]) < 0 else "b"
            self._draw_brick(image, center, max(0.4, scale), piece, face)
            cv2.rectangle(image, (center[0] - 30, center[1] - 70), (center[0] + 30, center[1] - 44), (72, 72, 78), -1)
        noise = np.random.default_rng().integers(0, 6, image.shape, dtype=np.uint8)
        return cv2.add(image, noise)

    def _render_wrist(self, state: RobotState, piece: Piece) -> np.ndarray:
        image = self._base(640, 480, (90, 86, 80), (60, 58, 54))
        if self._holding(state):
            self._draw_brick(image, (320, 300), 2.2, piece, "a" if float(state.joints[4]) < 0 else "b")
        jaw = int(80 + float(state.joints[5]) * 1.6)
        cv2.rectangle(image, (0, 380), (320 - jaw, 480), (30, 30, 34), -1)
        cv2.rectangle(image, (320 + jaw, 380), (640, 480), (30, 30, 34), -1)
        # Sensor noise: a flat synthetic brick would otherwise read as blurred to the usability check.
        noise = np.random.default_rng().integers(0, 8, image.shape, dtype=np.uint8)
        return cv2.add(image, noise)

    def _render_overhead(self, state: RobotState, piece: Piece) -> np.ndarray:
        image = self._base(456, 480, (176, 168, 152), (160, 152, 138))
        cv2.circle(image, (228, 440), 38, (60, 60, 66), -1)
        cv2.circle(image, (90, 300), 46, (46, 150, 80), 6)
        cv2.putText(image, "GOOD", (62, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 120, 70), 2)
        cv2.circle(image, (366, 300), 46, (190, 50, 50), 6)
        cv2.putText(image, "DEFECT", (326, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 40, 40), 2)
        x, y, _ = (float(v) for v in state.ee_xyz)
        ee = (int(228 - y * 560), int(440 - x * 900))
        if self.available():
            cv2.rectangle(image, (210, 170), (246, 192), piece.color, -1)
        cv2.line(image, (228, 440), ee, (40, 40, 46), 10)
        cv2.circle(image, ee, 12, (30, 30, 34), -1)
        if self._holding(state):
            cv2.rectangle(image, (ee[0] - 18, ee[1] - 11), (ee[0] + 18, ee[1] + 11), piece.color, -1)
        return image


class ScriptedPolicy:
    """Stand-in for an ACT export: predicts chunks by following its gesture's keyframes from the current state."""

    def __init__(self, role: str, fps: float, chunk: int = 100) -> None:
        if role not in SCRIPTS:
            raise ValueError(f"No scripted gesture for role {role!r}")
        self.role = role
        self._fps = fps
        self._chunk = chunk
        self._trajectory: np.ndarray | None = None
        self._cursor = 0

    def reset(self) -> None:
        self._trajectory = None
        self._cursor = 0

    def _plan(self, start: np.ndarray) -> np.ndarray:
        points: list[np.ndarray] = []
        current = start.astype(np.float64)
        for target, seconds in SCRIPTS[self.role]:
            goal = np.asarray(target, np.float64)
            steps = max(1, int(seconds * self._fps))
            for k in range(1, steps + 1):
                t = k / steps
                ease = t * t * (3 - 2 * t)
                points.append(current + (goal - current) * ease)
            current = goal
        return np.asarray(points, np.float32)

    def predict_action_chunk(self, state: np.ndarray, executed: int) -> np.ndarray:
        if self._trajectory is None:
            self._trajectory = self._plan(state)
        self._cursor = min(self._cursor + executed, len(self._trajectory) - 1)
        chunk = self._trajectory[self._cursor : self._cursor + self._chunk]
        if len(chunk) < self._chunk:
            pad = np.repeat(self._trajectory[-1:], self._chunk - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad])
        return chunk


def sim_anomaly_score(image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
    """Stand-in for the Anomalib model: share of near-black marker pixels on the brick, squashed into [0, 1]."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    brick = (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    marks = (hsv[..., 2] < 45).astype(np.uint8)
    ys, xs = np.nonzero(brick)
    if len(xs) == 0:
        return 0.5, np.zeros(image_rgb.shape[:2], np.float32)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    region = marks[y0 : y1 + 1, x0 : x1 + 1]
    ratio = float(region.mean())
    noise = float(np.random.default_rng().normal(0, 0.04))
    score = float(np.clip(0.18 + ratio * 9.0 + noise, 0.0, 1.0))
    heat = np.zeros(image_rgb.shape[:2], np.float32)
    heat[y0 : y1 + 1, x0 : x1 + 1] = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 9) * 4
    return score, np.clip(heat, 0, 1)
