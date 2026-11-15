"""The work cell: hardware, models, safety and the sorting cycle.

One cycle::

    PICK_PRESENT ──(arm reaches the inspection pose: frames every period_s, scored in parallel of the 180° turn)──▶
    DECIDE ──good──▶ PLACE_GOOD
           ──bad───▶ PLACE_BAD
           ──uncertain──▶ RECAPTURE (N fresh frames) ──▶ [REINSPECT policy, when enabled] ──▶ still uncertain → PLACE_BAD
           ──no brick recognized in most frames──▶ PLACE_BAD ({run.stop_after_no_brick} in a row stop the batch)

Each stage is a method returning its result; the cycle runs in its own thread so the API and the safety checks
stay responsive, and an e-stop cancels it between two control ticks (≤ 1/fps).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from sortops.anomaly_train import AnomalyTrainer
from sortops.config import JOINTS, POLICY_ROLES, Settings
from sortops.hw.cameras import CameraHub, SimCameraHub, StudioCameraHub, encode_jpeg
from sortops.hw.robot import Robot, RobotState, SimRobot, SO101Robot
from sortops.hw.sim import SimWorld
from sortops.metrics import CycleBook, CycleRecord, LatencyBook, SystemMonitor
from sortops.models.registry import ModelRegistry
from sortops.models.runtime import (
    AnomalyModel,
    AnomalyOutput,
    OpenVinoAnomaly,
    OpenVinoPolicy,
    Policy,
    SimAnomaly,
    SimPolicy,
    calibration_for,
    score_output,
)
from sortops.pipeline.annotation import InspectionStore
from sortops.pipeline.brick_crop import crop_brick
from sortops.pipeline.heatmap import render as render_heat
from sortops.pipeline.execution import ChunkPlayer
from sortops.pipeline.grasp import GraspDetector, admission_radius
from sortops.pipeline.inspection import Decision, Usability, Verdict, assess, decide_cameras
from sortops.pipeline.cv_anomaly import inspect as cv_inspect
from sortops.pipeline.replay import Episode, drop_pauses, load_episode, resolve_dataset
from sortops.safety import Mode, SafetyGuard


READ_FAILURES_BEFORE_RECONNECT = 3
"""Consecutive failed reads after which the arm link is closed and reopened."""

ROBOT_RETRY_S = 3.0
"""Seconds between two attempts to open the arm while it is unavailable (port held, powered off…)."""

PICK_STILL_BAND = 1.5
"""Gripper units the jaws may still drift while "stopped" on the brick (pick end detection)."""

CAMERA_GRACE_S = 3.0
"""How long a policy waits for a camera's first frame before the cycle fails."""


class Cancelled(Exception):
    """The cycle was stopped (e-stop or operator)."""


@dataclass
class Shot:
    """One inspection frame and what the pipeline made of it."""

    id: int
    cycle: int
    phase: str
    camera: str
    t: float
    usability: dict[str, Any]
    score: float | None
    jpeg: bytes = field(repr=False, default=b"")
    heat_jpeg: bytes = field(repr=False, default=b"")


class Cell:
    def __init__(self, settings: Settings, data_dir: Path) -> None:
        self.settings = settings
        self.data_dir = data_dir
        hardware = settings.section("hardware")
        self.simulated = hardware["backend"] == "mock"
        self.latency = LatencyBook()
        self.monitor = SystemMonitor()
        self.cycles = CycleBook(data_dir / "runs")
        self.inspection_store = InspectionStore(data_dir / "runs")
        hardware = settings.section("hardware")
        self.anomaly_trainer = AnomalyTrainer(
            self.inspection_store, data_dir / "anomaly-train", Path(hardware["anomaly_model_roots"][0]), hardware["anomalib_python"]
        )
        """Every inspection frame and annotation, kept on disk per run (Data tab, Anomalib export)."""
        self.registry = ModelRegistry(settings, data_dir, simulated=self.simulated)
        self.safety = SafetyGuard(lambda: self.settings.section("safety"))
        self.safety.on_estop(self._on_estop)

        self.robot: Robot
        self.world: SimWorld | None = None
        if self.simulated:
            sim = SimRobot()
            self.world = SimWorld(self._latest_state)
            sim.brick_available = self.world.available
            sim.on_grip = self.world.grip
            self.robot = sim
            self.cameras: CameraHub = SimCameraHub(self.world.render, hardware["cameras"])
        else:
            self.robot = SO101Robot(
                hardware["follower_port"], hardware["calibration"], read_only=hardware["arm_access"] == "read_only"
            )
            self.cameras = StudioCameraHub(
                hardware["studio_url"], hardware["studio_project"], hardware["studio_backend_src"], hardware["cameras"]
            )

        self._state_lock = threading.Lock()
        self._state: RobotState | None = None
        self._speed = 0.0
        self._control_active = False
        self.execution_info: dict[str, Any] | None = None
        """Live chunk player state of the running segment, for the UI."""
        self.ghost: np.ndarray | None = None
        """Last target the policy asked for (shown in 3D; in shadow mode it never reaches the motors)."""

        self._policies: dict[str, Policy] = {}
        self._anomaly: dict[str, AnomalyModel] = {}
        self._model_lock = threading.Lock()
        self.model_errors: dict[str, str] = {}

        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self.shots: deque[Shot] = deque(maxlen=48)
        self._shot_id = 0
        self.phase = "idle"
        self.phase_started = time.monotonic()
        self.current: CycleRecord | None = None
        self.last_decision: dict[str, Any] | None = None
        self.transition: dict[str, Any] | None = None
        """Live gates of the step the cycle is in (what it waits for to move on), for the UI."""
        self.benchmarks: dict[str, Any] = {}
        self.busy: str | None = None

        self._awaiting = threading.Event()
        self._verdict_given = threading.Event()
        self._verdict: Verdict | None = None
        self._verdict_annotation: dict[str, Any] | None = None
        self._verdict_for: tuple[str, int] | None = None  # (run, cycle) of the brick waiting for the verdict
        self._verdict_lock = threading.Lock()
        self._cancel = threading.Event()
        self._stop_after_cycle = threading.Event()
        self._last_no_brick = False
        # The operator handed the arm port to another program (Physical AI Studio): no reconnection until taken back.
        self.released = False
        self._episodes: dict[tuple[str, int, float], Episode] = {}  # (folder, episode, mtime) -> episode
        self._cycle_thread: threading.Thread | None = None
        self._poller_stop = threading.Event()
        self._poller: threading.Thread | None = None
        self._scorer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="score")
        self._infer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self.connected = False
        self.connect_error: str | None = None
        self._last_connect_attempt = 0.0
        self._read_failures = 0

    # Lifecycle ------------------------------------------------------------------------------------------

    def start(self) -> None:
        self.monitor.start()
        self._connect_robot()
        try:
            self.cameras.start()
        except Exception as error:  # noqa: BLE001
            self.connect_error = f"cameras: {error}"
            self.log("error", f"Cameras failed: {error}")
        self._poller = threading.Thread(target=self._poll, name="robot-poll", daemon=True)
        self._poller.start()
        self.log("info", f"Cell ready ({'simulation' if self.simulated else 'SO-101'}), mode SHADOW")

    def shutdown(self) -> None:
        self.stop_cycle(immediate=True)
        self._poller_stop.set()
        self.monitor.stop()
        self.cameras.stop()
        if self.connected:
            try:
                self.robot.close()
            except Exception:  # noqa: BLE001
                logger.exception("robot close failed")
        self._scorer.shutdown(wait=False, cancel_futures=True)
        self._infer.shutdown(wait=False, cancel_futures=True)

    def log(self, level: str, message: str) -> None:
        self.events.append({"t": time.time(), "level": level, "msg": message})
        getattr(logger, "warning" if level == "warn" else level, logger.info)(message)

    # Robot state ----------------------------------------------------------------------------------------

    def _latest_state(self) -> RobotState:
        with self._state_lock:
            if self._state is not None:
                return self._state
        return self.robot.read()

    def _read(self) -> RobotState:
        state = self.robot.read()
        with self._state_lock:
            previous = self._state
            self._state = state
            if previous is not None and state.timestamp > previous.timestamp:
                dt = state.timestamp - previous.timestamp
                instant = float(np.max(np.abs(state.joints[:5] - previous.joints[:5]))) / dt
                self._speed = 0.6 * self._speed + 0.4 * instant
        return state

    def speed(self) -> float:
        with self._state_lock:
            return self._speed

    def _connect_robot(self) -> None:
        """Open the arm; on failure keep the reason for the UI. The poller retries every ROBOT_RETRY_S."""
        self._last_connect_attempt = time.monotonic()
        try:
            self.robot.connect()
        except Exception as error:  # noqa: BLE001 - shown in the UI; the cell still serves metrics
            message = f"robot: {error}"
            if message != self.connect_error:
                self.log("error", f"Robot connection failed: {error}")
            self.connect_error = message
            return
        self.connected = True
        if self.connect_error and self.connect_error.startswith("robot:"):
            self.connect_error = None
        self.log("info", "Robot connected")

    def release_robot(self) -> None:
        """Close the arm port so Physical AI Studio (or any program) can open it; SortOps stops reconnecting.

        Raises:
            RuntimeError: A cycle or a Home move is running.
        """
        if self.cycle_running():
            raise RuntimeError("Stop the cycle first")
        with self.safety.gate:
            self.released = True
            was_connected, self.connected = self.connected, False
        if was_connected:
            self.robot.close()
        self.connect_error = "released: the arm port is free for Physical AI Studio (Take the arm back to reconnect)"
        self.log("info", "Arm released for other programs")

    def take_robot(self) -> None:
        """Reconnect to the arm on the next poll (fails, and retries, while another program still holds the port)."""
        self.released = False
        self._last_connect_attempt = 0.0
        self.connect_error = None
        self.log("info", "Taking the arm back")

    def _on_read_failure(self, error: Exception) -> None:
        """Repeated read failures: say why (another process on the bus?) and drop the link so it reconnects."""
        if self.released:
            return
        self._read_failures += 1
        shared = getattr(self.robot, "port_shared_with", lambda: [])()
        reason = f"another process opened the arm port: {', '.join(shared)}" if shared else str(error)
        if self._read_failures == 1 or shared:
            self.log("error", f"Robot read failed: {reason}")
        if self._read_failures >= READ_FAILURES_BEFORE_RECONNECT or shared:
            self.connected = False
            self.connect_error = f"robot: {reason}"
            self.stop_cycle(immediate=True)
            try:
                self.robot.close()
            except Exception:  # noqa: BLE001 - the bus is already unusable
                logger.exception("closing the arm after read failures")
            self._read_failures = 0
            self._last_connect_attempt = time.monotonic()

    def _poll(self) -> None:
        while not self._poller_stop.is_set():
            if not self.connected and not self.released and time.monotonic() - self._last_connect_attempt >= ROBOT_RETRY_S:
                self._connect_robot()
            if self.connected and not self._control_active:
                try:
                    state = self._read()
                    self._read_failures = 0
                    self.safety.check(state.temperature_c, self.cycle_running())
                except Exception as error:  # noqa: BLE001
                    self._on_read_failure(error)
                    self._poller_stop.wait(1.0)
            elif self._control_active:
                state = self._latest_state()
                self.safety.check(state.temperature_c, True)
            self._poller_stop.wait(0.04)

    def state(self) -> RobotState | None:
        with self._state_lock:
            return self._state

    # Safety actions -------------------------------------------------------------------------------------

    def _on_estop(self, reason: str) -> None:
        self._cancel.set()
        self._stop_after_cycle.set()
        if self.connected:
            try:
                self.robot.hold()
            except Exception as error:  # noqa: BLE001
                self.log("error", f"Hold after e-stop failed: {error}")
        self.log("error", f"E-STOP — {reason}")

    def estop(self, reason: str) -> None:
        self.safety.trigger(reason)

    def set_torque(self, *, enabled: bool) -> None:
        if not enabled:
            self._stop_after_cycle.set()
        with self.safety.gate:
            if not enabled:
                self._cancel.set()
            self.robot.set_torque(enabled=enabled)
            self.safety.torque = enabled
        self.log("warn" if not enabled else "info", f"Torque {'ON' if enabled else 'OFF'}")

    def set_mode(self, mode: Mode) -> None:
        if mode == Mode.LIVE and getattr(self.robot, "read_only", False):
            raise RuntimeError("The arm is opened read-only: set hardware.arm_access to control and restart")
        self.safety.set_mode(mode)
        self.log("warn" if mode == Mode.LIVE else "info", f"Mode {mode.value.upper()}")

    # Models ---------------------------------------------------------------------------------------------

    def invalidate_models(self) -> None:
        with self._model_lock:
            self._policies.clear()
            self._anomaly.clear()
            self.model_errors.clear()

    def policy(self, role: str) -> Policy:
        with self._model_lock:
            device = self.settings.section("devices")[role]
            cached = self._policies.get(role)
            if cached is not None and cached.device == device:
                return cached
            export = self.registry.export_dir(role)
            fps = float(self.settings.section("execution")[role]["hz"])
            loaded: Policy
            if export is not None:
                loaded = OpenVinoPolicy(role, self.registry.active()[role]["name"], export, device, self.latency)
            elif self.simulated:
                loaded = SimPolicy(role, fps, device, self.latency)
            else:
                raise RuntimeError(f"No model imported for role {role}")
            self._policies[role] = loaded
            return loaded

    def anomaly(self, camera: str) -> AnomalyModel:
        """The anomaly model of an inspection camera (role ``anomaly_<camera>``)."""
        role = f"anomaly_{camera}"
        with self._model_lock:
            device = self.settings.section("devices")[role]
            cached = self._anomaly.get(role)
            if cached is not None and cached.device == device:
                return cached
            export = self.registry.export_dir(role)
            model: AnomalyModel
            if export is not None:
                xml = next(iter(sorted(export.rglob("*.xml"))), None)
                if xml is None:
                    raise RuntimeError(f"No OpenVINO .xml in {export}")
                model = OpenVinoAnomaly(
                    self.registry.active()[role]["name"], xml, device, self.latency, self.registry.calibration(role), role
                )
            elif self.simulated:
                model = SimAnomaly(device, self.latency, role)
            else:
                raise RuntimeError(f"No anomaly model imported for the {camera} camera")
            self._anomaly[role] = model
            return model

    def crop_settings(self, camera: str) -> dict[str, Any]:
        """Brick-crop settings of ``camera``: its own search zone (the capture-app camera keeps ``roi``)."""
        cfg = self.settings.section("inspection")
        if camera != cfg["camera"]:
            cfg["roi"] = cfg["rois"].get(camera, [0.0, 0.0, 1.0, 1.0])
        return cfg

    def model_image(self, camera: str, frame: np.ndarray, crop_image: np.ndarray) -> np.ndarray:
        """What ``camera``'s anomaly model scores: the brick crop, or the whole frame for a full-frame model."""
        kind = self.settings.section("inspection")["model_input"].get(camera, "crop")
        return frame if kind == "frame" else crop_image

    def live_scores(self) -> dict[str, dict[str, Any]]:
        """Each inspection camera's current brick crop scored now: raw score, SortOps score, crop found (for tuning)."""
        out: dict[str, dict[str, Any]] = {}
        for camera in self.settings.section("inspection")["cameras"]:
            frame = self.cameras.latest(camera)
            if frame is None:
                out[camera] = {"found": False, "error": "no frame"}
                continue
            crop = crop_brick(frame.image, self.crop_settings(camera))
            image = self.model_image(camera, frame.image, crop.image)
            inspection_cfg = self.settings.section("inspection")
            if inspection_cfg["anomaly_method"] == "cv":
                found = cv_inspect(frame.image, inspection_cfg)
                out[camera] = {"found": found.found, "raw": round(max(found.stain, found.mark), 4),
                               "exported": round(found.score, 4), "score": round(found.score, 4)}
                continue
            try:
                model = self.anomaly(camera)
                if isinstance(model, OpenVinoAnomaly):
                    out_ = model.infer(image)
                    exported, raw = out_.exported, out_.raw
                    score = score_output(out_, model.calibration)
                else:
                    score, _ = model.score(image)
                    exported, raw = score, None
            except Exception as error:  # noqa: BLE001 - shown next to the camera
                out[camera] = {"found": crop.found, "error": str(error)[:200]}
                continue
            out[camera] = {
                "found": crop.found,
                # inf/nan would break the JSON: report them as missing (and the overflow in the error field).
                "raw": None if raw is None or not np.isfinite(raw) else round(raw, 3),
                **({"error": "raw score is not finite (precision overflow)"} if raw is not None and not np.isfinite(raw) else {}),
                "exported": round(exported, 4),
                "score": round(score, 4),
            }
        return out

    def calibrate_anomaly(self, camera: str, seconds: float, min_frames: int = 12) -> dict[str, Any]:
        """Score live crops of good bricks on ``camera`` for ``seconds`` and set that model's scale from their raw scores.

        The operator shows several good bricks (turning them) in front of the camera meanwhile.
        """
        if self.cycle_running():
            raise RuntimeError("Stop the cycle before calibrating")
        model = self.anomaly(camera)
        if not isinstance(model, OpenVinoAnomaly) or not (model.has_raw or model.has_raw_map):
            raise RuntimeError("This anomaly model exposes no raw score to calibrate")
        cfg = self.crop_settings(camera)
        goods: list[AnomalyOutput] = []
        self.busy = f"calibrating {camera} anomaly score"
        try:
            deadline = time.monotonic() + seconds
            last = 0.0
            while time.monotonic() < deadline:
                frame = self.cameras.latest(camera)
                if frame is not None and frame.timestamp != last:
                    last = frame.timestamp
                    crop = crop_brick(frame.image, cfg)
                    if crop.found:
                        goods.append(model.infer(self.model_image(camera, frame.image, crop.image)))
                time.sleep(0.15)
        finally:
            self.busy = None
        if len(goods) < min_frames:
            raise RuntimeError(f"Only {len(goods)} crops with a brick in {seconds:.0f} s (need {min_frames}): show good bricks to the {camera} camera")
        values: dict[str, Any] = {
            **calibration_for(goods, cfg["score_peak_weight"], cfg["score_peak_fraction"]),
            "created": time.time(),
            "manual": False,
        }
        self.registry.save_calibration(values, f"anomaly_{camera}")
        self.invalidate_models()
        method = "peak+mean of the raw map" if values.get("terms") else "raw score"
        self.log("info", f"{camera} anomaly score calibrated on {len(goods)} good crops ({method}): defect line {values['threshold']:.2f}")
        return values

    def set_defect_line(self, camera: str, threshold: float, spread: float | None) -> dict[str, Any]:
        """Manual defect line on the raw score of ``camera``'s model: raw = threshold scores 0.5 (SortOps scale)."""
        role = f"anomaly_{camera}"
        current = self.registry.calibration(role) or {}
        width = spread if spread is not None else current.get("spread") or max(abs(threshold) * 0.25, 1e-3)
        if width <= 0:
            raise ValueError("spread must be positive")
        values = {**current, "threshold": float(threshold), "spread": float(width), "manual": True, "created": time.time()}
        self.registry.save_calibration(values, role)
        self.invalidate_models()
        self.log("info", f"{camera} defect line set by hand: raw {threshold:.3f} (spread {width:.3f})")
        return values

    def preload(self) -> None:
        """Load (compile) every model the cycle needs, so the first cycle does not pay the compile time."""
        roles = ["pick_present", "place_good", "place_bad"]
        if self.settings.section("decision")["reinspect_enabled"]:
            roles.append("reinspect")
        for role in roles:
            try:
                self.policy(role)
                self.model_errors.pop(role, None)
            except Exception as error:  # noqa: BLE001
                self.model_errors[role] = str(error)
        for camera in self.settings.section("inspection")["cameras"]:
            role = f"anomaly_{camera}"
            if self.operator_mode():
                self.model_errors.pop(role, None)
                continue
            try:
                self.anomaly(camera)
                self.model_errors.pop(role, None)
            except Exception as error:  # noqa: BLE001
                self.model_errors[role] = str(error)

    def _images(self, keys: list[str]) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for key in keys:
            frame = self.cameras.latest(key)
            # Cameras that just opened (or reconnect) get a short grace period before the first frame.
            deadline = time.monotonic() + CAMERA_GRACE_S
            while frame is None and time.monotonic() < deadline and not self._cancel.is_set():
                time.sleep(0.05)
                frame = self.cameras.latest(key)
            if frame is None:
                raise RuntimeError(f"Camera {key} has no frame")
            images[key] = frame.image
        return images

    def benchmark(self, role: str, devices: list[str], runs: int = 20) -> dict[str, Any]:
        """Latency of ``role``'s model on each device with the live observation (p50/p95, compile time)."""
        if self.cycle_running():
            raise RuntimeError("Stop the cycle before benchmarking")
        results: dict[str, Any] = {}
        original = self.settings.section("devices")[role]
        self.busy = f"benchmark {role}"
        try:
            for device in devices:
                self.settings.update({"devices": {role: device}})
                try:
                    started = time.perf_counter()
                    timings: list[float] = []
                    if role.startswith("anomaly_"):
                        camera = role.removeprefix("anomaly_")
                        model = self.anomaly(camera)
                        compile_ms = (time.perf_counter() - started) * 1000.0
                        full = self._images([camera])[camera]
                        image = self.model_image(camera, full, crop_brick(full, self.crop_settings(camera)).image)
                        for i in range(runs + 2):
                            t0 = time.perf_counter()
                            model.score(image)
                            if i >= 2:
                                timings.append((time.perf_counter() - t0) * 1000.0)
                    else:
                        policy = self.policy(role)
                        compile_ms = (time.perf_counter() - started) * 1000.0
                        state = self._latest_state().joints
                        images = self._images(policy.camera_keys)
                        for i in range(runs + 2):
                            policy.reset()
                            t0 = time.perf_counter()
                            policy.predict(state, images, 0)
                            if i >= 2:
                                timings.append((time.perf_counter() - t0) * 1000.0)
                    values = np.asarray(timings)
                    results[device] = {
                        "ok": True,
                        "compile_ms": round(compile_ms, 1),
                        "p50": round(float(np.percentile(values, 50)), 2),
                        "p95": round(float(np.percentile(values, 95)), 2),
                        "runs": runs,
                    }
                except Exception as error:  # noqa: BLE001 - a device may not support the model
                    results[device] = {"ok": False, "error": str(error)[:300]}
        finally:
            self.settings.update({"devices": {role: original}})
            self.invalidate_models()
            self.busy = None
        self.benchmarks[role] = {"t": time.time(), "results": results}
        self.log("info", f"Benchmark {role}: " + ", ".join(f"{d} {r.get('p50', 'n/a')} ms" for d, r in results.items()))
        return self.benchmarks[role]

    # Cycle control --------------------------------------------------------------------------------------

    def cycle_running(self) -> bool:
        return self._cycle_thread is not None and self._cycle_thread.is_alive()

    def start_cycles(self, *, loop: bool, forever: bool = False) -> None:
        if self.cycle_running():
            raise RuntimeError("A cycle is already running")
        if self.safety.estopped:
            raise RuntimeError("E-stop is latched: reset it first")
        if not self.connected:
            raise RuntimeError(f"Robot not connected: {self.connect_error}")
        self._cancel.clear()
        self._stop_after_cycle.clear()
        self.safety.heartbeat()
        self._cycle_thread = threading.Thread(target=self._run, args=(loop, forever), name="cycle", daemon=True)
        self._cycle_thread.start()

    def go_home(self) -> None:
        """Bring the arm to the home pose (``home.joints``) at ``home.speed``, in its own stoppable motion thread."""
        if self.cycle_running():
            raise RuntimeError("The arm is busy: stop the cycle first")
        if self.safety.estopped:
            raise RuntimeError("E-stop is latched: reset it first")
        if not self.connected:
            raise RuntimeError(f"Robot not connected: {self.connect_error}")
        if not self.safety.may_move():
            raise RuntimeError("Switch to Live to move the arm (Shadow never moves it)")
        self._cancel.clear()
        self._stop_after_cycle.clear()
        self.safety.heartbeat()
        self._cycle_thread = threading.Thread(target=self._run_home, name="home", daemon=True)
        self._cycle_thread.start()

    def _run_home(self) -> None:
        self._set_phase("home")
        try:
            self._home_motion()
        except Cancelled:
            self.log("warn", "Going home stopped")
        finally:
            if self._cancel.is_set() and self.connected:
                try:
                    self.robot.hold()
                except Exception:  # noqa: BLE001
                    logger.exception("hold after home")
            self._set_phase("idle")

    def _ensure_released(self) -> None:
        """Hold the arm where it is and open the gripper to ``release.open`` until it is open (2 s at most)."""
        release = self.settings.section("release")
        state = self._read()
        target = state.joints.copy()
        target[len(JOINTS) - 1] = float(release["open"])
        began = time.monotonic()
        self._control_active = True
        try:
            while True:
                self._check()
                state = self._send(target)
                if float(state.joints[-1]) >= float(release["open"]) - float(release["open_tolerance"]):
                    self.sleep(float(release["hold_s"]))  # let the brick fall
                    return
                if time.monotonic() - began > 2.0:
                    self.log("warn", f"release: gripper at {float(state.joints[-1]):.0f} after 2 s, going on")
                    return
                self.sleep(1.0 / 30.0)
        finally:
            self._control_active = False

    def _home_motion(self) -> bool:
        """Move to ``home.joints`` at ``home.speed`` (straight line in joint space, eased); Cancelled on stop.

        Returns True once home is reached, False when it was not (timeout, motion error): the arm then holds.
        """
        home = self.settings.section("home")
        target = np.asarray(home["joints"], np.float32)
        speed = float(home["speed"])
        period = 1.0 / 30.0
        self._control_active = True
        self.log("info", "Going home")
        try:
            start = self._read().joints.copy()
            duration = max(float(np.max(np.abs(target - start))) / max(speed, 1e-3), 0.3)
            began = time.monotonic()
            while True:
                self._check()
                tick = time.monotonic()
                state = self._read()
                fraction = min(1.0, (tick - began) / duration)
                ease = fraction * fraction * (3 - 2 * fraction)
                goal = start + (target - start) * ease
                self.ghost = target
                safe = self.safety.filter(goal, state.joints)
                with self.safety.gate:
                    if not self._cancel.is_set() and self.safety.may_move():
                        self.robot.send(safe)
                if fraction >= 1.0 and float(np.max(np.abs(state.joints[:5] - target[:5]))) < 2.0:
                    self.log("info", "Home reached")
                    return True
                if tick - began > duration + 5.0:
                    self.log("warn", "Home: not reached within 5 s of the planned time, holding here")
                    self._hold_quietly()
                    return False
                remaining = period - (time.monotonic() - tick)
                if remaining > 0:
                    self.sleep(remaining)
        except Cancelled:
            raise
        except Exception as error:  # noqa: BLE001 - reported, the arm holds
            self.log("error", f"Going home failed: {error}")
            self._hold_quietly()
            return False
        finally:
            self._control_active = False

    def _hold_quietly(self) -> None:
        """Freeze the arm where it is (under the motor gate); a failure is logged, never raised."""
        if not self.connected:
            return
        try:
            with self.safety.gate:
                self.robot.hold()
        except Exception as error:  # noqa: BLE001
            self.log("error", f"Hold failed: {error}")

    def teach_home(self) -> list[float]:
        state = self.state()
        if state is None:
            raise RuntimeError("No robot state yet")
        joints = [round(float(v), 2) for v in state.joints]
        self.settings.update({"home": {"joints": joints}})
        self.log("info", f"Home pose taught: {joints}")
        return joints

    def stop_cycle(self, *, immediate: bool) -> None:
        self._stop_after_cycle.set()
        if immediate:
            # Under the motor gate: the control loop re-checks cancellation inside it, so no command follows the hold.
            with self.safety.gate:
                self._cancel.set()
                if self.connected and self.cycle_running():
                    try:
                        self.robot.hold()
                    except Exception as error:  # noqa: BLE001
                        self.log("error", f"Hold failed: {error}")

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()

    def _run(self, loop: bool, forever: bool = False) -> None:
        """Cycles until a stop. ``forever``: never ends by itself (no brick, failures, max_cycles): pick → … → home →
        pick again, until Stop or the e-stop."""
        max_cycles = int(self.settings.section("run")["max_cycles"])
        stop_after_no_brick = int(self.settings.section("run")["stop_after_no_brick"])
        done = 0
        no_brick_streak = 0
        try:
            self.preload()
            if self.model_errors:
                self.log("error", "Model load failed: " + "; ".join(f"{k}: {v}" for k, v in self.model_errors.items()))
                return
            while not self._cancel.is_set():
                outcome = self._cycle()
                done += 1
                # Every cycle ends at home (place done, no brick or failure), unless stopped / e-stopped.
                if not self._cancel.is_set() and not self.safety.estopped:
                    self._set_phase("home")
                    self._home_motion()
                if self._stop_after_cycle.is_set() or self._cancel.is_set():
                    break
                if forever:
                    if outcome == "failed":
                        self.sleep(1.0)  # a failing cycle must not spin
                    continue
                no_brick_streak = no_brick_streak + 1 if self._last_no_brick else 0
                if no_brick_streak >= stop_after_no_brick:
                    self.log("info", f"No brick recognized {no_brick_streak} cycles in a row: batch finished")
                    break
                if outcome == "no_piece":
                    self.log("info", "No brick presented: batch finished")
                    break
                if outcome in {"failed", "aborted"} or not loop:
                    break
                if max_cycles and done >= max_cycles:
                    break
        except Cancelled:
            pass
        finally:
            self._control_active = False
            self._set_phase("idle")
            self.current = None

    def _cycle(self) -> str:
        self._last_no_brick = False
        record = self.cycles.open()
        self.current = record
        energy_start = self.monitor.energy_j()
        inferences_start = self._inference_count()
        self.log("info", f"Cycle {record.id} started")
        try:
            inspection = _Inspection(self, record)
            if self.settings.section("present")["enabled"]:
                # Pick with the pick model, then present by replaying the recorded episode; inspection frames are taken
                # whenever the arm is near the inspection pose, during either step.
                attempts = int(self.settings.section("run")["pick_attempts"])
                attempt = 0
                while True:
                    attempt += 1
                    record.pick_attempts = attempt
                    pick = self._pick_done(inspection.on_tick)
                    if self._timed(record, "pick", lambda: self._segment("pick_present", pick)) != "timeout":
                        self._note(record, "pick", "present", f"attempt {attempt}: {pick.summary()}", True)
                        break
                    if not self._pick_failed(record, "pick", attempt, attempts, pick.summary()):
                        return record.outcome
                    inspection.reset_pick()  # frames of the failed attempt must not judge the next brick
                self._timed(record, "present", lambda: self._present(inspection.on_tick))
            else:
                if not self._pick_and_present(record, inspection):
                    return record.outcome
            if self.operator_mode():
                inspection.finish_frames()
                decision = self._timed(record, "await_operator", lambda: self._await_operator(record))
            else:
                decision = self._timed(record, "inspect", inspection.finish)
            if decision.verdict == Verdict.UNCERTAIN:
                decision = self._timed(record, "recapture", lambda: inspection.recapture(decision))
            if (
                decision.verdict == Verdict.UNCERTAIN
                and self.settings.section("decision")["reinspect_enabled"]
                and self.registry.active().get("reinspect") is not None
            ):
                decision = self._timed(record, "reinspect", lambda: inspection.reinspect(decision))
            if decision.verdict == Verdict.UNCERTAIN:
                decision = Decision(Verdict.BAD, decision.score, f"still uncertain ({decision.reason}) → defect bin")
            record.score = decision.score
            record.decision_reason = decision.reason
            record.camera_decision = decision.camera_scores
            opinions = inspection.opinions()
            opinions["manual"] = {"verdict": decision.verdict.value, "score": None} if self.operator_mode() else None
            record.opinions = opinions
            self.last_decision = {"cycle": record.id, "verdict": decision.verdict.value, **asdict(decision), "opinions": opinions}
            self._annotate(record, decision)
            self.log("info", f"Cycle {record.id}: {decision.verdict.value.upper()} — {decision.reason}")
            nxt = {Verdict.GOOD: "place_good", Verdict.BAD: "place_bad"}.get(decision.verdict, "end")
            self._note(record, "inspect", nxt, f"{decision.verdict.value.upper()}: {decision.reason}", decision.verdict != Verdict.NO_PIECE, log=False)
            if decision.verdict == Verdict.NO_PIECE:
                record.outcome = "no_piece"
                return record.outcome
            self._last_no_brick = decision.no_brick
            role = "place_good" if decision.verdict == Verdict.GOOD else "place_bad"
            place = self.settings.section("place_replay")
            if place["enabled"]:
                # No place model: replay one recorded deposit demo (it opens the gripper over the bin at the right spot).
                target = place[role]
                self._timed(
                    record, role, lambda: self._present(lambda *_: None, str(target["dataset"]), int(target["episode"]), role)
                )
            else:
                self._timed(record, role, lambda: self._segment(role, None))
            # The brick must be out of the gripper before going home and starting the next one.
            self._timed(record, "release", self._ensure_released)
            record.outcome = decision.verdict.value
            if self.world is not None:
                self.world.placed()
            return record.outcome
        except Cancelled:
            record.outcome = "aborted"
            record.error = self.safety.estop_reason or "stopped by operator"
            return record.outcome
        except Exception as error:  # noqa: BLE001 - the cell holds and reports
            record.outcome = "failed"
            record.error = str(error)
            self.log("error", f"Cycle {record.id} failed: {error}")
            if self.connected:
                try:
                    self.robot.hold()
                except Exception:  # noqa: BLE001
                    logger.exception("hold after failure")
            return record.outcome
        finally:
            sample = self.monitor.latest()
            if sample is not None and sample.package_w is not None:
                record.energy_j = round(self.monitor.energy_j() - energy_start, 1)
            record.inferences = self._inference_count() - inferences_start
            self.cycles.close(record)

    def _annotate(self, record: CycleRecord, decision: Decision) -> None:
        """Save the cycle's annotation: the operator's drawing when given, else one built from the best frames (piece =
        automatic brick box; for a model defect verdict, the regions above the defect contour)."""
        verdict = decision.verdict.value if decision.verdict != Verdict.UNCERTAIN else "bad"
        operator = self.operator_mode()
        try:
            drawn = self._verdict_annotation if operator else None
            if drawn and drawn.get("cameras"):
                self.inspection_store.save_annotation(record.run, record.id, {**drawn, "verdict": verdict, "source": "operator"})
            else:
                self.inspection_store.auto_annotation(record.run, record.id, verdict, "operator" if operator else "model")
            if operator and verdict in ("good", "bad"):
                record.truth = verdict
        except Exception as error:  # noqa: BLE001 - a bad drawing must not stop the cell
            self.log("error", f"Cycle {record.id}: annotation not saved: {error}")
        finally:
            self._verdict_annotation = None

    # Operator verdict (stands in for the anomaly model) -----------------------------------------------------

    def operator_mode(self) -> bool:
        return self.settings.section("decision")["verdict_source"] == "operator"

    def awaiting_verdict(self) -> bool:
        return self._awaiting.is_set()

    def give_verdict(
        self, verdict: str, annotation: dict[str, Any] | None = None, run: str | None = None, cycle: int | None = None
    ) -> None:
        """The operator's call on the presented brick: ``good``, ``bad`` or ``no_piece``, with optionally the piece box
        and defect shapes drawn on the frames (``sortops.pipeline.annotation``), saved with the cycle. ``run``/``cycle``
        name the brick the answer is for: a late answer for a previous brick is refused instead of answering the next."""
        with self._verdict_lock:
            # One answer per brick: a double click must not overwrite it or answer the next brick.
            if not self._awaiting.is_set() or self._verdict_given.is_set():
                raise RuntimeError("No brick is waiting for a verdict")
            if (run is not None or cycle is not None) and (run, cycle) != self._verdict_for:
                waiting = self._verdict_for
                raise RuntimeError(
                    f"This verdict is for {run} cycle {cycle}, but the brick waiting is "
                    f"{waiting[0] if waiting else '?'} cycle {waiting[1] if waiting else '?'}"
                )
            self._verdict = Verdict(verdict)
            self._verdict_annotation = annotation
            self._verdict_given.set()
            self._awaiting.clear()

    def _await_operator(self, record: CycleRecord) -> Decision:
        """Hold the brick at the camera until the operator says good / defect / no brick (or the cycle stops)."""
        with self._verdict_lock:
            self._verdict = None
            self._verdict_annotation = None
            self._verdict_for = (record.run, record.id)
            self._verdict_given.clear()
            self._awaiting.set()
        self.log("info", f"Cycle {record.id}: waiting for the operator's verdict")
        try:
            while not self._verdict_given.wait(0.1):
                self._check()
        finally:
            self._awaiting.clear()
        verdict = self._verdict
        if verdict is None or verdict == Verdict.UNCERTAIN:
            raise RuntimeError("Invalid operator verdict")
        return Decision(verdict, None, f"operator: {verdict.value.replace('_', ' ')}")

    def _inference_count(self) -> int:
        return int(sum(int(v["count"]) for k, v in self.latency.summary().items() if not k.startswith("load:")))

    def _timed(self, record: CycleRecord, stage: str, action: Callable[[], Any]) -> Any:
        self._set_phase(stage)
        started = time.monotonic()
        try:
            return action()
        finally:
            # Accumulated: a stage run twice (pick retried from home) shows its total time.
            record.stages[stage] = round(record.stages.get(stage, 0.0) + time.monotonic() - started, 2)

    def _check(self) -> None:
        if self._cancel.is_set() or self.safety.estopped:
            raise Cancelled

    def sleep(self, seconds: float) -> None:
        if self._cancel.wait(seconds):
            raise Cancelled

    def _segment(
        self,
        role: str,
        on_tick: Callable[[RobotState, float], bool | None] | None,
        can_settle: Callable[[], bool] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Run ``role``'s policy until the arm settles (when ``can_settle`` allows), ``on_tick`` says done, or timeout
        (``timeout_s``, else ``segments.<role>.timeout_s``)."""
        policy = self.policy(role)
        policy.reset()
        execution = self.settings.section("execution")[role]
        segments = self.settings.section("segments")
        limits = segments[role]
        timeout = float(limits["timeout_s"]) if timeout_s is None else timeout_s
        period = 1.0 / float(execution["hz"])
        latest: dict[str, RobotState] = {}

        def predict(executed: int) -> np.ndarray:
            # Observation taken when the chunk is requested (async: the loop keeps playing meanwhile).
            state_now = latest["state"]
            return policy.predict(state_now.joints.copy(), self._images(policy.camera_keys), executed)

        player = ChunkPlayer(execution, predict, self._infer)
        started = time.monotonic()
        still_since: float | None = None
        release = self.settings.section("release")
        reflex = bool(release["enabled"]) and bool(release.get(role, False))
        frozen: np.ndarray | None = None  # joints held while the gripper opens (release reflex)
        opening_since = open_since = 0.0
        self._control_active = True
        self.execution_info = {"role": role, **execution, "chunk": 0, "queue": 0, "requests": 0}
        try:
            while True:
                self._check()
                tick_start = time.monotonic()
                state = self._read()
                latest["state"] = state
                elapsed = time.monotonic() - started
                if frozen is None:
                    target = player.next_action()
                    self.execution_info.update(chunk=player.chunk_size, queue=player.remaining, requests=player.requests)
                    gripper = len(JOINTS) - 1
                    if (
                        reflex
                        and target is not None
                        and elapsed >= float(release["min_s"])
                        and float(target[gripper]) >= float(release["trigger"])
                    ):
                        # The policy starts opening over the bowl: hold the arm still and open wide.
                        frozen = state.joints.copy()
                        frozen[gripper] = float(release["open"])
                        opening_since = time.monotonic()
                        self.log("info", f"{role}: release at {elapsed:.1f} s (gripper command {float(target[gripper]):.0f})")
                if frozen is not None:
                    target = frozen
                    now = time.monotonic()
                    is_open = float(state.joints[-1]) >= float(release["open"]) - float(release["open_tolerance"])
                    if is_open and not open_since:
                        open_since = now
                    if (open_since and now - open_since >= float(release["hold_s"])) or (
                        now - opening_since >= float(release["max_open_s"])
                    ):
                        if not open_since:
                            self.log("warn", f"{role}: gripper at {float(state.joints[-1]):.0f} after {float(release['max_open_s'])} s, moving on")
                        self.ghost = target
                        return "released"
                if target is not None:
                    self.ghost = target
                    safe = self.safety.filter(target, state.joints)
                    with self.safety.gate:
                        # Authoritative check: stop, e-stop and torque changes all take this gate before acting.
                        if not self._cancel.is_set() and self.safety.may_move():
                            self.robot.send(safe)

                elapsed = time.monotonic() - started
                if on_tick is not None and on_tick(state, elapsed):
                    return "done"
                settle_allowed = can_settle is None or can_settle()
                if frozen is None and settle_allowed and self.speed() < segments["still_speed"] and elapsed >= limits["min_s"]:
                    still_since = still_since or time.monotonic()
                    if time.monotonic() - still_since >= limits["still_s"]:
                        return "settled"
                else:
                    still_since = None
                if frozen is None and elapsed >= timeout:
                    self.log("warn", f"{role}: timeout after {elapsed:.0f} s")
                    if not reflex:
                        return "timeout"
                    # Place with the release reflex: open anyway, the next pick must not start with the brick held.
                    frozen = state.joints.copy()
                    frozen[len(JOINTS) - 1] = float(release["open"])
                    opening_since = time.monotonic()
                remaining = period - (time.monotonic() - tick_start)
                if remaining > 0:
                    self.sleep(remaining)
        finally:
            player.close()
            self.execution_info = None
            if self._cancel.is_set() and self.connected:
                # Whatever the last tick sent, the arm freezes where it is now.
                try:
                    self.robot.hold()
                except Exception as error:  # noqa: BLE001
                    self.log("error", f"Hold failed: {error}")
            self._control_active = False

    # Present: replay of a recorded episode ----------------------------------------------------------------

    def _present_episode(self, dataset: str | None = None, episode_index: int | None = None) -> Episode | None:
        """The episode the present step replays (cached until its dataset changes); None in simulation without one.

        Raises:
            FileNotFoundError, ValueError: On the real cell, the dataset or the episode is missing or unusable.
        """
        cfg = self.settings.section("present")
        dataset = str(cfg["dataset"]) if dataset is None else dataset
        episode_index = int(cfg["episode"]) if episode_index is None else episode_index
        hardware = self.settings.section("hardware")
        try:
            folder = resolve_dataset(
                dataset,
                Path(hardware["studio_datasets_dir"]).expanduser(),
                hardware["studio_url"],
                hardware["studio_project"],
            )
        except FileNotFoundError:
            if self.world is not None:
                return None
            raise
        key = (str(folder), episode_index, (folder / "meta" / "info.json").stat().st_mtime)
        if key not in self._episodes:
            self._episodes[key] = load_episode(folder, episode_index)
        episode = self._episodes[key]
        return Episode(drop_pauses(episode.actions, float(cfg["pause_epsilon"])), episode.fps)

    def _note(self, record: CycleRecord, stage: str, nxt: str, text: str, ok: bool, *, log: bool = True) -> None:
        """Record one transition decision of the cycle (shown in the Cockpit and the cycle table) and log it."""
        record.decisions.append(
            {"t": round(time.time() - record.started, 1), "stage": stage, "next": nxt, "text": text, "ok": ok}
        )
        if log:
            self.log("info" if ok else "warn", f"Cycle {record.id}: {stage} → {nxt}: {text}")

    def _pick_failed(self, record: CycleRecord, stage: str, attempt: int, attempts: int, why: str) -> bool:
        """No brick held: home, wait ``pick.retry_wait_s``, try again (True) — or give up after ``attempts`` (False)."""
        wait = float(self.settings.section("pick")["retry_wait_s"])
        limit = f"{attempt}/{attempts}" if attempts else f"{attempt}"
        if attempts and attempt >= attempts:
            self._note(record, stage, "end", f"attempt {limit}: {why} → no brick, cycle ends", False)
            record.outcome = "no_piece"
            record.error = f"no brick held after {attempts} picks"
            return False
        self._note(record, stage, "home", f"attempt {limit}: {why} → home, wait {wait:.1f} s, pick again", False)
        self.transition = None
        if not self._timed(record, "home", self._home_motion):
            # Never start a new pick from an unknown pose: the arm holds and the cycle stops here.
            self._note(record, "home", "end", "home not reached → arm held, no new pick", False)
            record.outcome = "failed"
            record.error = "home not reached before a pick retry"
            return False
        self.sleep(wait)
        return True

    def _pick_and_present(self, record: CycleRecord, inspection: _Inspection) -> bool:
        """Step 1 (pick & present model) until its gates pass (see ``_PickTransition``); retried from home when no
        brick is held in time. False when the cycle ends here (no brick after ``run.pick_attempts``)."""
        attempts = int(self.settings.section("run")["pick_attempts"])
        attempt = 0
        while True:
            attempt += 1
            record.pick_attempts = attempt
            gate = _PickTransition(self, inspection, attempt, attempts)
            try:
                self._timed(
                    record,
                    "pick_present",
                    lambda gate=gate: self._segment("pick_present", gate, lambda: False, timeout_s=gate.hard_timeout_s),
                )
            finally:
                self.transition = None
            if gate.outcome in {"ready", "present_max"}:
                self._note(record, "pick_present", "inspect", gate.explain(), gate.outcome == "ready")
                return True
            if not self._pick_failed(record, "pick_present", attempt, attempts, gate.explain()):
                return False
            inspection.reset_pick()

    def _pick_done(self, on_tick: Callable[[RobotState, float], bool | None]) -> _PickWatch:
        """Tick of the pick step (present on): done once the gripper opened, then closed and stopped on the brick."""
        return _PickWatch(self, on_tick)

    def _present(
        self,
        on_tick: Callable[[RobotState, float], bool | None],
        dataset: str | None = None,
        episode_index: int | None = None,
        label: str = "Present",
    ) -> str:
        """Glide to the episode's first pose, then send its actions as recorded, through the safety filter."""
        episode = self._present_episode(dataset, episode_index)
        if episode is None:
            self.log("warn", f"{label}: no recorded episode in simulation, skipped")
            return "skipped"
        cfg = self.settings.section("present")
        period = 1.0 / 30.0
        self._control_active = True
        name = cfg["dataset"] if dataset is None else dataset
        self.log("info", f"{label}: replaying {name} episode {cfg['episode'] if episode_index is None else episode_index} ({episode.duration_s:.0f} s)")
        try:
            first = episode.actions[0]
            start = self._read().joints.copy()
            glide_s = max(float(np.max(np.abs(first - start))) / float(cfg["approach_speed"]), 0.3)
            began = time.monotonic()
            while True:
                tick = self._present_tick(began, on_tick)
                fraction = min(1.0, (tick - began) / glide_s)
                ease = fraction * fraction * (3 - 2 * fraction)
                state = self._send(start + (first - start) * ease)
                # Arm joints only: holding the brick, the gripper cannot close to the recorded value.
                if fraction >= 1.0 and float(np.max(np.abs(state.joints[:5] - first[:5]))) < 3.0:
                    break
                if tick - began > glide_s + float(cfg["approach_settle_s"]):
                    # Holding the brick (or against gravity) the arm can stop a few units short: no point waiting.
                    break
                self._pace(tick, period)
            rate = float(cfg["playback_speed"])
            began = time.monotonic()
            while True:
                tick = self._present_tick(began, on_tick)
                index = int((tick - began) * rate * episode.fps)
                if index >= len(episode.actions):
                    return "replayed"
                self._send(episode.actions[index])
                self._pace(tick, period)
        finally:
            if self._cancel.is_set() and self.connected:
                try:
                    self.robot.hold()
                except Exception as error:  # noqa: BLE001
                    self.log("error", f"Hold failed: {error}")
            self._control_active = False

    def _present_tick(self, began: float, on_tick: Callable[[RobotState, float], bool | None]) -> float:
        self._check()
        tick = time.monotonic()
        on_tick(self._latest_state(), tick - began)
        return tick

    def _send(self, target: np.ndarray) -> RobotState:
        """Send ``target`` through the safety filter under the motor gate; returns the state it was filtered from."""
        state = self._read()
        self.ghost = target
        safe = self.safety.filter(target, state.joints)
        with self.safety.gate:
            if not self._cancel.is_set() and self.safety.may_move():
                self.robot.send(safe)
        return state

    def _pace(self, tick: float, period: float) -> None:
        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            self.sleep(remaining)

    # Inspection frames ----------------------------------------------------------------------------------

    def record_shot(
        self,
        record: CycleRecord,
        phase: str,
        camera: str,
        frame: np.ndarray,
        roi: np.ndarray,
        box: tuple[float, float, float, float],
        found: bool,
        usability: Usability,
        score: float | None,
        heat: np.ndarray | None,
        raw_map: np.ndarray | None = None,
        calibration: dict[str, Any] | None = None,
    ) -> None:
        """Keep one inspection frame: in memory for the live strip, and on disk with the whole frame for the Data tab.

        ``roi`` is what the model scored (or would have); the anomaly map is rendered on it at a fixed, calibrated scale
        (``sortops.pipeline.heatmap``) instead of the export's per-frame normalization.
        """
        self._shot_id += 1
        shot_id = self._shot_id
        heat_rgb: np.ndarray | None = None
        heat_info: dict[str, Any] | None = None
        if heat is not None or raw_map is not None:
            try:
                heat_rgb, info = render_heat(roi, raw_map, heat, calibration)
                heat_info = asdict(info)
            except Exception as error:  # noqa: BLE001 - a bad map must not lose the frame
                self.log("warn", f"Anomaly map of shot {shot_id} not rendered: {error}")
        try:
            self.inspection_store.save_shot(
                record.run, record.id, shot_id, camera, phase, frame, roi, box, asdict(usability), score,
                heat_rgb, heat_info, raw_map, found,
            )
        except Exception as error:  # noqa: BLE001 - the cycle goes on even if the disk is full
            self.log("error", f"Inspection frame {shot_id} not saved: {error}")
        self.shots.append(
            Shot(
                id=shot_id,
                cycle=record.id,
                phase=phase,
                camera=camera,
                t=time.time(),
                usability=asdict(usability),
                score=None if score is None else round(score, 3),
                jpeg=encode_jpeg(roi, 70),
                heat_jpeg=b"" if heat_rgb is None else encode_jpeg(heat_rgb, 85),
            )
        )

    # Snapshot for the UI --------------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        state = self.state()
        sample = self.monitor.latest()
        current = self.current
        return {
            "t": time.time(),
            "simulated": self.simulated,
            "read_only": bool(getattr(self.robot, "read_only", False)),
            "connected": self.connected,
            "connect_error": self.connect_error,
            "released": self.released,
            "mode": self.safety.mode.value,
            "estop": {"latched": self.safety.estopped, "reason": self.safety.estop_reason},
            "torque": self.safety.torque,
            "phase": self.phase,
            "phase_s": round(time.monotonic() - self.phase_started, 2),
            "running": self.cycle_running(),
            "busy": self.busy,
            "cycle": None if current is None else {"id": current.id, "stages": current.stages, "scores": current.scores,
                                                   "camera_scores": current.camera_scores,
                                                   "frames_seen": current.frames_seen, "frames_usable": current.frames_usable,
                                                   "decisions": current.decisions, "pick_attempts": current.pick_attempts},
            "robot": None if state is None else {
                "joints": [round(float(v), 2) for v in state.joints],
                "names": list(JOINTS),
                "ee": [round(float(v), 4) for v in state.ee_xyz],
                "temp": [round(float(v), 1) for v in state.temperature_c],
                "load": [round(float(v), 1) for v in state.load],
                "current": [round(float(v), 3) for v in state.current_a],
                "voltage": round(state.voltage_v, 2),
                "speed": round(self.speed(), 2),
            },
            "execution": self.execution_info,
            "ghost": None if self.ghost is None else [round(float(v), 2) for v in self.ghost],
            "temp_median": None if self.safety.temp_median is None else [round(float(v), 1) for v in self.safety.temp_median],
            "temp_limit": self.settings.section("safety")["max_motor_temp_c"],
            "clamped_ticks": self.safety.clamped_ticks,
            "heartbeat_age": round(min(self.safety.heartbeat_age(), 999.0), 2),
            "system": None if sample is None else asdict(sample),
            "last_decision": self.last_decision,
            "verdict_source": self.settings.section("decision")["verdict_source"],
            "inspection_cameras": self.settings.section("inspection")["cameras"],
            "awaiting_verdict": self.awaiting_verdict(),
            "inspection": None if current is None else {"run": current.run, "cycle": current.id},
            "transition": self.transition,
            "piece": None if self.world is None else {"index": self.world.piece.index, "defective": self.world.piece.defective},
        }


class _Inspection:
    """Inspection of one cycle: frames sampled while the arm is near the inspection pose, scored off-thread."""

    def __init__(self, cell: Cell, record: CycleRecord) -> None:
        self.cell = cell
        self.record = record
        self.cfg = cell.settings.section("inspection")
        self.decision_cfg = cell.settings.section("decision")
        self.last_capture = 0.0
        self.first_capture: float | None = None
        self.futures: list[Future[None]] = []
        self.no_brick_frames: dict[str, int] = {}
        self.camera_frames: dict[str, int] = {}
        # best_frame_only: per camera, the most usable frame seen so far (phase, image, crop, usability); only it is
        # scored and kept, when the inspection is finished, instead of every frame of the window.
        self.best: dict[str, tuple[str, np.ndarray, Any, Usability]] = {}
        self.method_scores: dict[str, dict[str, list[float]]] = {}  # "anomalib" / "algorithm" -> camera -> scores
        self.lock = threading.Lock()
        # Step 1 sampling (see sample()): counters of the current pick attempt.
        self._generation = 0
        self._tick = 0
        self.samples = 0
        self.admissible = 0
        """Samples where an inspection camera saw a brick (anywhere): they widen the admission zone."""
        self.admissible_in_zone = 0
        self.first_in_zone: float | None = None
        self.last_distance: float | None = None
        self.zone_radius: float | None = None
        """Admission radius of the step-1 sampling (None when the pick was not sampled this way)."""

    def _capture(self, phase: str, state: RobotState, speed: float) -> None:
        """One frame from every inspection camera, each scored off-thread by that camera's model."""
        ee = state.ee_xyz.copy()
        for camera in self.cfg["cameras"]:
            frame = self.cell.cameras.latest(camera)
            if frame is None:
                continue
            with self.lock:
                self.record.frames_seen += 1
            image = frame.image.copy()
            self.futures.append(self.cell._scorer.submit(self._score, phase, camera, image, ee, speed))  # noqa: SLF001

    def sample(self, state: RobotState, radius: float) -> None:
        """Step 1 sampling (brick held): every ``period_s`` one frame per inspection camera, wherever the arm is.

        Each sample says whether a brick was seen (admissible) and whether the arm was in the admission zone
        (``radius`` around ``pose_xyz``); only in-zone samples become inspection frames.
        """
        now = time.monotonic()
        if now - self.last_capture < float(self.cfg["period_s"]):
            return
        self.last_capture = now
        ee = state.ee_xyz.copy()
        distance = float(np.linalg.norm(ee - np.asarray(self.cfg["pose_xyz"], np.float32)))
        in_zone = distance <= radius
        self.zone_radius = radius
        frames = [(cam, f) for cam in self.cfg["cameras"] if (f := self.cell.cameras.latest(cam)) is not None]
        if not frames:
            return
        with self.lock:
            self._tick += 1
            tick = _Sample(self._generation, self._tick, now, in_zone, radius, distance, len(frames))
            self.samples += 1
            if in_zone:
                self.record.frames_seen += len(frames)
        speed = self.cell.speed()
        for camera, frame in frames:
            image = frame.image.copy()
            self.futures.append(self.cell._scorer.submit(self._score, "present", camera, image, ee, speed, tick))

    def reset_pick(self) -> None:
        """A new pick attempt: forget everything the previous attempt saw (a dropped brick, an empty grasp).

        Frames still being rated are finished first (and ignored), so none lands in the next attempt's evidence.
        """
        with self.lock:
            self._generation += 1
        for future in self.futures:
            try:
                future.result(timeout=30)
            except Exception as error:  # noqa: BLE001 - a frame of the discarded attempt: logged, not fatal
                logger.debug("discarded frame failed: {}", error)
        self.futures.clear()
        with self.lock:
            self.best = {}
            self.no_brick_frames, self.camera_frames = {}, {}
            self.method_scores = {}
            self.record.scores.clear()
            self.record.camera_scores.clear()
            self.first_capture = None
            self.samples = self.admissible = self.admissible_in_zone = 0
            self.first_in_zone = None
            self.last_distance = None

    def _score(
        self, phase: str, camera: str, image: np.ndarray, ee: np.ndarray, speed: float, tick: _Sample | None = None
    ) -> None:
        cfg = self.cell.crop_settings(camera)
        radius = tick.radius if tick is not None else self.zone_radius
        if radius is not None:
            # A frame taken in the admission zone is at the pose: "off pose" only counts beyond its radius
            # (recaptures keep the zone the pick ended with).
            cfg["pose_tolerance_m"] = radius
        crop = crop_brick(image, cfg)
        usability = assess(crop, ee, speed, cfg)
        if tick is not None:
            seen = crop.found and usability.brick_fraction >= cfg["min_brick_fraction"]
            with self.lock:
                if tick.generation != self._generation:
                    return  # sampled during a previous pick attempt
                tick.seen = tick.seen or seen
                tick.pending -= 1
                if tick.pending == 0:
                    self.last_distance = tick.distance
                    if tick.seen:
                        self.admissible += 1
                        if tick.in_zone:
                            self.admissible_in_zone += 1
                            self.first_in_zone = self.first_in_zone or tick.t
            if not tick.in_zone:
                return
        with self.lock:
            self.camera_frames[camera] = self.camera_frames.get(camera, 0) + 1
            if not crop.found or usability.brick_fraction < cfg["min_brick_fraction"]:
                self.no_brick_frames[camera] = self.no_brick_frames.get(camera, 0) + 1
        if usability.usable:
            with self.lock:
                self.record.frames_usable += 1
        if self.cfg["best_frame_only"]:
            with self.lock:
                kept = self.best.get(camera)
                if kept is None or (usability.usable, usability.score) > (kept[3].usable, kept[3].score):
                    self.best[camera] = (phase, image, crop, usability)
            return
        self._analyze(phase, camera, image, crop, usability)

    def _analyze(self, phase: str, camera: str, image: np.ndarray, crop: Any, usability: Usability) -> None:
        """Score one rated frame with BOTH methods (Anomalib model, CV algorithm) and keep it.

        The verdict source picks which scores decide (model = Anomalib, cv = Algorithm, operator = Manual); the other
        method's scores are still kept as its opinion (shown next to the decision to compare the three).
        """
        cfg = self.cell.settings.section("inspection")
        found = cv_inspect(image, cfg)
        model_crop = found.crop if found.found else self.cell.model_image(camera, image, crop.image)
        anomalib: float | None = None
        algorithm: float | None = None
        heat: np.ndarray | None = None
        raw_map: np.ndarray | None = None
        calibration: dict[str, Any] | None = None
        if usability.usable and camera in cfg["anomalib_cameras"]:
            try:
                model = self.cell.anomaly(camera)
                if isinstance(model, OpenVinoAnomaly):
                    out = model.infer(model_crop)
                    anomalib, heat, raw_map = score_output(out, model.calibration), out.heat, out.raw_map
                    calibration = model.calibration
                else:
                    anomalib, heat = model.score(model_crop)
            except Exception as error:  # noqa: BLE001 - no model for this camera: no Anomalib opinion
                logger.debug("anomalib {}: {}", camera, error)
            if found.found and camera in cfg["cv_cameras"]:
                algorithm = found.score
        source = self.cell.settings.section("decision")["verdict_source"]
        with self.lock:
            for method, value in (("anomalib", anomalib), ("algorithm", algorithm)):
                if value is not None:
                    self.method_scores.setdefault(method, {}).setdefault(camera, []).append(round(value, 4))
            chosen = {"model": anomalib, "cv": algorithm}.get(source)
            if chosen is not None:
                self.record.scores.append(round(chosen, 4))
                self.record.camera_scores.setdefault(camera, []).append(round(chosen, 4))
        shown_score = algorithm if source == "cv" else anomalib
        if source == "cv" and found.found:
            heat, raw_map, calibration = found.heat, None, None
        shown, box = (found.crop, found.box) if found.found else (crop.image, crop.box)
        self.cell.record_shot(
            self.record, phase, camera, image, shown, box, crop.found or found.found, usability, shown_score, heat,
            raw_map, calibration,
        )

    def opinions(self) -> dict[str, dict[str, Any] | None]:
        """What Anomalib and the CV algorithm each say about this brick, whichever decides."""
        out: dict[str, dict[str, Any] | None] = {}
        with self.lock:
            scores = {m: {c: list(v) for c, v in cams.items()} for m, cams in self.method_scores.items()}
            seen = self.record.frames_seen
        for method in ("anomalib", "algorithm"):
            cams = scores.get(method)
            if not cams:
                out[method] = None
                continue
            decision = decide_cameras(cams, seen, 0, self.decision_cfg, self.cfg["aggregate"], 1)
            out[method] = {"verdict": decision.verdict.value, "score": decision.score}
        return out

    def on_tick(self, state: RobotState, elapsed: float) -> bool:
        _ = elapsed
        distance = float(np.linalg.norm(state.ee_xyz - np.asarray(self.cfg["pose_xyz"], np.float32)))
        now = time.monotonic()
        if distance <= self.cfg["pose_tolerance_m"] * 2.0 and now - self.last_capture >= self.cfg["period_s"]:
            self.last_capture = now
            if self.first_capture is None:
                self.first_capture = now
            self._capture("present", state, self.cell.speed())
        return False

    def can_settle(self) -> bool:
        """Pick may end once the inspection window (both faces) is over, or if the pose was never reached."""
        return self.first_capture is None or time.monotonic() - self.first_capture >= self.cfg["window_s"]

    def _wait(self) -> None:
        for future in self.futures:
            future.result(timeout=30)
        self.futures.clear()

    def finish(self) -> Decision:
        self._wait()
        self._commit_best()
        return self._decide()

    def finish_frames(self) -> None:
        """Operator mode: the frames are still rated and kept (dataset, display), nothing is scored."""
        self._wait()
        self._commit_best()

    def _commit_best(self) -> None:
        """best_frame_only: analyze and keep the one best frame of each camera gathered since the last commit."""
        with self.lock:
            best, self.best = self.best, {}
        for camera, (phase, image, crop, usability) in best.items():
            self._analyze(phase, camera, image, crop, usability)

    def _decide(self) -> Decision:
        with self.lock:
            scores = {camera: list(values) for camera, values in self.record.camera_scores.items()}
            seen = self.record.frames_seen
            # A brick counts as missing only if no camera recognized it often enough: front may score whole frames
            # (no crop) while wrist sees the brick.
            seen, missing = min(
                ((n, self.no_brick_frames.get(cam, 0)) for cam, n in self.camera_frames.items()),
                key=lambda pair: pair[1] / pair[0],
                default=(seen, 0),
            )
        # One scored frame per camera (per commit) when best_frame_only: a camera with its best frame usable takes part.
        min_frames = 1 if self.cfg["best_frame_only"] else int(self.cfg["min_usable_frames"])
        return decide_cameras(scores, seen, missing, self.decision_cfg, self.cfg["aggregate"], min_frames)

    def recapture(self, first: Decision) -> Decision:
        self.cell.log("warn", f"Uncertain ({first.reason}): recapturing {self.decision_cfg['recapture_frames']} frames")
        self.record.recaptures += 1
        for _ in range(int(self.decision_cfg["recapture_frames"])):
            self.cell.sleep(float(self.decision_cfg["recapture_period_s"]))
            self._capture("recapture", self.cell._latest_state(), self.cell.speed())  # noqa: SLF001
        self._wait()
        self._commit_best()
        return self._decide()

    def reinspect(self, previous: Decision) -> Decision:
        self.cell.log("warn", f"Still uncertain ({previous.reason}): re-inspecting from another angle")

        def tick(state: RobotState, elapsed: float) -> bool:
            _ = elapsed
            now = time.monotonic()
            if now - self.last_capture >= self.cfg["period_s"]:
                self.last_capture = now
                self._capture("reinspect", state, self.cell.speed())
            return False

        self.cell._segment("reinspect", tick)  # noqa: SLF001
        self._wait()
        self._commit_best()
        return self._decide()


class _PickWatch:
    """Pick tick (present on): done once the gripper opened, then closed and stopped on the brick.

    "Closed" is below `pick_closed_below`, or `pick_close_drop` under the widest opening seen: a brick stops the jaws
    before they reach the fully closed value. ``summary()`` says what was seen, for the timeout log.
    """

    def __init__(self, cell: Cell, on_tick: Callable[[RobotState, float], bool | None]) -> None:
        self.cell = cell
        self.on_tick = on_tick
        self.cfg = cell.settings.section("present")
        self.opened = False
        self.peak = 0.0
        self.low = 100.0
        self.since = 0.0
        self.ref = 0.0
        pick = cell.settings.section("pick")
        self.grasp = GraspDetector(pick) if pick["require_grasp"] else None
        self.held_text = ""

    def __call__(self, state: RobotState, elapsed: float) -> bool:
        self.on_tick(state, elapsed)
        gripper = float(state.joints[-1])
        if self.grasp is not None:
            self.grasp.update(gripper, float(state.load[-1]), time.monotonic())
        self.low = min(self.low, gripper)
        if gripper > float(self.cfg["pick_open_above"]):
            self.opened = True
        if self.opened:
            self.peak = max(self.peak, gripper)
        closed = self.opened and (
            gripper < float(self.cfg["pick_closed_below"]) or gripper <= self.peak - float(self.cfg["pick_close_drop"])
        )
        if not closed:
            self.since = 0.0
            return False
        now = time.monotonic()
        if not self.since or abs(gripper - self.ref) > PICK_STILL_BAND:
            # Closed but still moving: the hold counts from the moment the jaws stop.
            self.since, self.ref = now, gripper
            return False
        if now - self.since >= float(self.cfg["pick_closed_s"]) and (self.grasp is None or self.grasp.held):
            self.cell.log("info", f"pick: brick held (gripper {gripper:.0f}, opened to {self.peak:.0f}) at {elapsed:.1f} s")
            self.held_text = f"brick held at {elapsed:.1f} s (gripper {gripper:.0f}, opened to {self.peak:.0f}" + (
                f", load {self.grasp.peak_load:.0f})" if self.grasp is not None else ")"
            )
            return True
        return False

    def summary(self) -> str:
        if self.held_text:
            return self.held_text
        if not self.opened:
            return f"gripper never opened above {float(self.cfg['pick_open_above']):.0f} (lowest {self.low:.0f})"
        if self.grasp is not None and not self.grasp.held:
            return f"gripper opened to {self.peak:.0f}, no brick held ({self.grasp.last.reason})"
        return f"gripper opened to {self.peak:.0f}, never stayed closed"


@dataclass
class _Sample:
    """One step-1 sampling instant (every inspection camera), filled in as its frames are rated."""

    generation: int
    id: int
    t: float
    in_zone: bool
    radius: float
    distance: float
    pending: int
    seen: bool = False


class _PickTransition:
    """Tick of step 1 (pick & present model): decides, explicitly, when the cycle moves on.

    Gates, all shown live (``info()`` → snapshot ``transition``):
      1. grasp — a brick is held (gripper motor: closed, not fully, squeezing; ``sortops.pipeline.grasp``);
      2. frames — ``pick.min_admissible_in_zone`` admissible frames (a brick seen) taken in the admission zone, whose
         radius grows by ``pick.zone_growth`` every ``pick.zone_step_frames`` admissible frames seen anywhere;
      3. window — ``inspection.window_s`` since the first of them (both faces under the cameras).
    Outcomes: ``ready`` (all gates), ``present_max`` (held ``pick.present_max_s`` without passing: inspect anyway),
    ``no_grasp`` (nothing held after ``segments.pick_present.timeout_s``: home and retry).
    """

    def __init__(self, cell: Cell, inspection: _Inspection, attempt: int, attempts: int) -> None:
        self.cell = cell
        self.inspection = inspection
        self.attempt = attempt
        self.attempts = attempts
        self.cfg = cell.settings.section("pick")
        self.icfg = cell.settings.section("inspection")
        self.grasp_timeout_s = float(cell.settings.section("segments")["pick_present"]["timeout_s"])
        self.hard_timeout_s = self.grasp_timeout_s + float(self.cfg["present_max_s"]) + 5.0
        self.grasp = GraspDetector(self.cfg)
        self.outcome: str | None = None
        self.elapsed = 0.0
        self.grasped_at: float | None = None
        """Seconds into the step at which the brick was (last) grasped."""
        self.radius, self.growth = admission_radius({**self.icfg, **self.cfg}, 0)

    def __call__(self, state: RobotState, elapsed: float) -> bool:
        self.elapsed = elapsed
        reading = self.grasp.update(float(state.joints[-1]), float(state.load[-1]), time.monotonic())
        held = reading.held or not self.cfg["require_grasp"]
        if held and self.grasped_at is None:
            self.grasped_at = elapsed
            if reading.held:
                self.cell.log("info", f"pick: {reading.reason} at {elapsed:.1f} s")
        elif not held and self.grasped_at is not None:
            self.cell.log("warn", f"pick: {self.grasp.lost_reason or 'brick lost'} at {elapsed:.1f} s")
            self.grasped_at = None
            self.inspection.reset_pick()
        self.radius, self.growth = admission_radius({**self.icfg, **self.cfg}, self.inspection.admissible)
        if held:
            self.inspection.sample(state, self.radius)
        gates = self.gates(state)
        if held and all(g["ok"] for g in gates):
            self.outcome = "ready"
        elif held and self.grasped_at is not None and elapsed - self.grasped_at >= float(self.cfg["present_max_s"]):
            self.outcome = "present_max"
        elif not held and elapsed >= self.grasp_timeout_s:
            self.outcome = "no_grasp"
        self.cell.transition = self.info(state, gates)
        return self.outcome is not None

    def gates(self, state: RobotState) -> list[dict[str, Any]]:
        ins = self.inspection
        reading = self.grasp.last
        need = int(self.cfg["min_admissible_in_zone"])
        window = float(self.icfg["window_s"])
        since = 0.0 if ins.first_in_zone is None else time.monotonic() - ins.first_in_zone
        distance = float(np.linalg.norm(state.ee_xyz - np.asarray(self.icfg["pose_xyz"], np.float32)))
        grasp_ok = reading.held or not self.cfg["require_grasp"]
        return [
            {"key": "grasp", "label": "Brick held", "ok": grasp_ok,
             "value": f"gripper {reading.position:.0f} · load {reading.load:.0f}",
             "target": f"{self.cfg['min_closed_pos']:.0f} ≤ gripper < {self.cfg['open_below']:.0f}, load ≥ {self.cfg['min_load']:.0f}"
             if self.cfg["require_grasp"] else "not required", "detail": reading.reason},
            {"key": "zone", "label": "In admission zone", "ok": distance <= self.radius,
             "value": f"{distance * 100:.1f} cm", "target": f"≤ {self.radius * 100:.1f} cm",
             "detail": f"+{self.growth * 100:.0f} % after {ins.admissible} admissible frames" if self.growth else "base radius"},
            {"key": "frames", "label": "Admissible frames in zone", "ok": ins.admissible_in_zone >= need,
             "value": f"{ins.admissible_in_zone}", "target": f"≥ {need}",
             "detail": f"{ins.admissible} admissible of {ins.samples} sampled"},
            {"key": "window", "label": "Inspection window", "ok": ins.first_in_zone is not None and since >= window,
             "value": f"{since:.1f} s", "target": f"≥ {window:.1f} s", "detail": "since the first admissible frame in zone"},
        ]

    def info(self, state: RobotState, gates: list[dict[str, Any]]) -> dict[str, Any]:
        base, _ = admission_radius({**self.icfg, **self.cfg}, 0)
        step = max(1, int(self.cfg["zone_step_frames"]))
        admissible = self.inspection.admissible
        capped = self.growth >= float(self.cfg["zone_max_growth"])
        held_for = None if self.grasped_at is None else self.elapsed - self.grasped_at
        return {
            "stage": "pick_present",
            "next": "inspect",
            "attempt": self.attempt,
            "attempts": self.attempts,
            "elapsed_s": round(self.elapsed, 1),
            "gates": gates,
            "zone": {
                "base_m": round(base, 4),
                "radius_m": round(self.radius, 4),
                "growth": round(self.growth, 3),
                "admissible": admissible,
                "next_growth_at": None if capped else (admissible // step + 1) * step,
                "distance_m": None if self.inspection.last_distance is None else round(self.inspection.last_distance, 4),
            },
            "timers": [
                {"label": "No brick → home", "elapsed": round(self.elapsed, 1) if self.grasped_at is None else None,
                 "limit": self.grasp_timeout_s,
                 "active": self.grasped_at is None},
                {"label": "Held, inspect anyway", "elapsed": None if held_for is None else round(held_for, 1),
                 "limit": float(self.cfg["present_max_s"]), "active": held_for is not None},
            ],
        }

    def explain(self) -> str:
        """Why step 1 ended, in one line (decision log)."""
        ins = self.inspection
        reading = self.grasp.last
        zone = f"radius {self.radius * 100:.0f} cm" + (f", +{self.growth * 100:.0f} % after {ins.admissible} admissible" if self.growth else "")
        if self.outcome == "ready":
            grasp = f"held (load {self.grasp.peak_load:.0f})" if self.cfg["require_grasp"] else "grasp not required"
            return f"{grasp}, {ins.admissible_in_zone} admissible frames in zone ({zone}), window {self.icfg['window_s']:.1f} s passed"
        if self.outcome == "present_max":
            return (
                f"held {self.cfg['present_max_s']:.0f} s but only {ins.admissible_in_zone}/{self.cfg['min_admissible_in_zone']} "
                f"admissible frames in zone ({zone}, last distance "
                f"{'—' if ins.last_distance is None else f'{ins.last_distance * 100:.0f} cm'}) → inspecting what we have"
            )
        return f"no brick held after {self.elapsed:.0f} s ({reading.reason})"
