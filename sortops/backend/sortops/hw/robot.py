"""The follower arm behind one small interface: a real SO-101 (``physicalai_so101_plugin``) or a simulated one.

All positions are in the SO-101 ``normalized`` unit the Studio datasets were recorded in: body joints in
[-100, 100], gripper in [0, 100]. The serial bus is not thread-safe, so every call goes through one lock.
"""

from __future__ import annotations

import fcntl
import math
import os
import termios
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from sortops.config import JOINTS

# URDF joint limits (radians) of the SO-101 follower, used by the simulator's forward kinematics.
URDF_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}


@dataclass
class RobotState:
    joints: np.ndarray
    """Joint positions, normalized unit, shape (6,)."""
    ee_xyz: np.ndarray
    """Gripper frame position in metres (URDF base frame)."""
    temperature_c: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    load: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    current_a: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    voltage_v: float = 0.0
    timestamp: float = 0.0


class Robot(Protocol):
    name: str

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def read(self) -> RobotState: ...
    def send(self, target: np.ndarray) -> None: ...
    def hold(self) -> None: ...
    def set_torque(self, *, enabled: bool) -> None: ...


def normalized_to_radians(joints: np.ndarray) -> np.ndarray:
    """Normalized SO-101 positions to URDF joint angles, the mapping Physical AI Studio's 3D view uses."""
    out = np.empty(6, dtype=np.float64)
    for i, name in enumerate(JOINTS):
        low, high = URDF_LIMITS[name]
        fraction = joints[i] / 100.0 if name == "gripper" else (joints[i] + 100.0) / 200.0
        out[i] = low + (high - low) * min(max(float(fraction), 0.0), 1.0)
    return out


def forward_kinematics(angles_rad: np.ndarray) -> np.ndarray:
    """Gripper frame position from the official SO-101 follower URDF chain (same chain as the SO-101 driver)."""
    chain = (
        ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
        ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
        ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
        ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
        ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
    )
    tool = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))

    def origin(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
        r, p, y = rpy
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.asarray(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
                [-sp, cp * sr, cp * cr, xyz[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def rz(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

    transform = np.eye(4)
    for angle, (xyz, rpy) in zip(angles_rad[:5], chain, strict=True):
        transform = transform @ origin(xyz, rpy) @ rz(float(angle))
    transform = transform @ origin(*tool)
    return transform[:3, 3].astype(np.float32)


class SimRobot:
    """A kinematic stand-in: joints chase their target at a bounded speed, telemetry follows the effort."""

    name = "sim"
    REST = np.asarray([0.0, -95.0, 95.0, 60.0, 0.0, 2.0], dtype=np.float32)

    def __init__(self, max_speed: float = 120.0) -> None:
        self._lock = threading.Lock()
        self._joints = self.REST.copy()
        self._target = self.REST.copy()
        self._torque = True
        self._max_speed = max_speed
        self._last = time.monotonic()
        self._temp = np.full(6, 31.0, dtype=np.float32)
        # Simulated brick between the jaws: closing at the pick spot while a brick waits there, the gripper stalls at
        # GRIP_STALL (load rises, as on the real arm) until it opens again.
        self.brick_available: Callable[[], bool] = lambda: False
        self.on_grip: Callable[[bool], None] = lambda held: None
        self._gripping = False

    GRIP_STALL = 3.0

    def connect(self) -> None:
        self._last = time.monotonic()

    def close(self) -> None:
        return None

    def _advance(self) -> None:
        now = time.monotonic()
        dt = min(now - self._last, 0.2)
        self._last = now
        if not self._torque:
            # No torque: the arm sags toward its rest pose under gravity.
            self._joints += (self.REST - self._joints) * min(1.0, dt * 1.5)
            return
        step = self._max_speed * dt
        before = float(self._joints[5])
        self._joints += np.clip(self._target - self._joints, -step, step)
        after = float(self._joints[5])
        if self._gripping:
            if after > 15.0:
                self._gripping = False
                self.on_grip(False)
            else:
                self._joints[5] = max(after, self.GRIP_STALL)
        elif before >= self.GRIP_STALL > after and self.brick_available():
            x, _, z = (float(v) for v in forward_kinematics(normalized_to_radians(self._joints)))
            if x > 0.22 and z < 0.0:  # the pick spot (in front, down at the table)
                self._gripping = True
                self._joints[5] = self.GRIP_STALL
                self.on_grip(True)

    def read(self) -> RobotState:
        with self._lock:
            self._advance()
            error = np.abs(self._target - self._joints) if self._torque else np.zeros(6, dtype=np.float32)
            load = np.clip(error * 4.0 + 8.0, 0.0, 100.0).astype(np.float32)
            self._temp += (31.0 + load * 0.15 - self._temp) * 0.002
            joints = self._joints.copy()
            return RobotState(
                joints=joints,
                ee_xyz=forward_kinematics(normalized_to_radians(joints)),
                temperature_c=self._temp.copy(),
                load=load,
                current_a=(load * 0.006).astype(np.float32),
                voltage_v=12.1,
                timestamp=time.monotonic(),
            )

    def send(self, target: np.ndarray) -> None:
        with self._lock:
            self._advance()
            self._target = np.asarray(target, dtype=np.float32).copy()

    def hold(self) -> None:
        with self._lock:
            self._advance()
            self._target = self._joints.copy()
            self._torque = True

    def set_torque(self, *, enabled: bool) -> None:
        with self._lock:
            self._advance()
            self._torque = enabled
            self._target = self._joints.copy()


class SO101Robot:
    """The real follower through ``physicalai_so101_plugin.SO101`` (normalized unit, LeRobot calibration).

    ``read_only`` opens the bus without ever writing torque or goal positions: the arm keeps whatever state
    it is in (no drop, no move). Used to watch the arm and run models in shadow mode with nobody at the table.
    """

    name = "so101"

    def __init__(self, port: str, calibration: str, *, read_only: bool = False) -> None:
        from physicalai_so101_plugin import SO101  # noqa: PLC0415 - only on the robot PC

        self.read_only = read_only
        self._robot = SO101(port=port, calibration=calibration, role="follower", unit="normalized")
        if read_only:
            # connect()/disconnect() toggle torque and re-write goals; a read-only session must not.
            self._robot._set_torque = _no_torque  # noqa: SLF001
            self._robot.torque_on_disconnect = False
        self._lock = threading.Lock()

    def _writable(self) -> None:
        if self.read_only:
            raise PermissionError("Arm opened read-only (hardware.arm_access = read_only)")

    def connect(self) -> None:
        holders = port_holders(self._robot.port)
        if holders:
            # Two processes on one servo bus garble each other's packets: never share it (e.g. with a Studio session).
            raise ConnectionError(f"{self._robot.port} is held by {', '.join(holders)}")
        with self._lock:
            self._robot.connect()
            self._claim_port()

    def _claim_port(self) -> None:
        """TIOCEXCL: further open() of the tty fails with EBUSY, so a Studio session started later cannot join the
        bus mid-run (two masters garble each other's packets: "Sync read failed")."""
        connection = getattr(self._robot, "_connection", None)
        serial_port = getattr(getattr(connection, "port_handler", None), "ser", None)
        if serial_port is None:
            return
        try:
            fcntl.ioctl(serial_port.fileno(), termios.TIOCEXCL)
        except OSError as error:
            logger.warning("Could not lock {} for exclusive use: {}", self._robot.port, error)

    def port_shared_with(self) -> list[str]:
        """Other processes holding the arm's port right now (should stay empty while connected)."""
        return port_holders(self._robot.port)

    def close(self) -> None:
        with self._lock:
            self._robot.disconnect()

    def read(self) -> RobotState:
        with self._lock:
            observation = self._robot.get_observation()
        telemetry = observation.sensor_data or {}
        joints = np.asarray(observation.joint_positions, dtype=np.float32)
        pose = telemetry.get("ee.pose")
        ee = np.asarray(pose[:3], dtype=np.float32) if pose is not None else forward_kinematics(
            normalized_to_radians(joints)
        )
        voltage = telemetry.get("motor.voltage")
        return RobotState(
            joints=joints,
            ee_xyz=ee,
            temperature_c=np.asarray(telemetry.get("motor.temperature", np.zeros(6)), dtype=np.float32),
            load=np.asarray(telemetry.get("motor.load", np.zeros(6)), dtype=np.float32),
            current_a=np.asarray(telemetry.get("motor.current", np.zeros(6)), dtype=np.float32),
            voltage_v=float(np.mean(voltage)) if voltage is not None else 0.0,
            timestamp=float(observation.timestamp),
        )

    def send(self, target: np.ndarray) -> None:
        self._writable()
        with self._lock:
            self._robot.send_action(np.asarray(target, dtype=np.float32))

    def hold(self) -> None:
        """Freeze where the arm is: current position becomes the goal, torque on. No-op when read-only."""
        if self.read_only:
            return
        with self._lock:
            current = np.asarray(self._robot.get_observation().joint_positions, dtype=np.float32)
            self._robot.send_action(current)
            self._robot.set_torque(enabled=True)

    def set_torque(self, *, enabled: bool) -> None:
        self._writable()
        with self._lock:
            if enabled:
                current = np.asarray(self._robot.get_observation().joint_positions, dtype=np.float32)
                self._robot.send_action(current)
            self._robot.set_torque(enabled=enabled)


def _no_torque(*, enabled: bool) -> None:
    _ = enabled


def port_holders(port: str) -> list[str]:
    """Other processes holding ``port`` open (``pid name``), read from ``/proc/*/fd``."""
    target = os.path.realpath(port)
    holders: list[str] = []
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        pid = fd_dir.parent.name
        if int(pid) == os.getpid():
            continue
        try:
            if any(os.path.realpath(fd) == target for fd in fd_dir.iterdir()):
                cmdline = (fd_dir.parent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                name = (fd_dir.parent / "comm").read_text().strip()
                # Studio keeps the arm open for the whole session, torque off included, until the session is closed.
                owner = "Physical AI Studio session" if "physicalai" in cmdline or "multiprocessing" in cmdline else name
                holders.append(f"{owner} (pid {pid})")
        except OSError:
            continue
    return holders
