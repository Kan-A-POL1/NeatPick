"""Settings: one JSON document, defaults in code, operator overrides persisted in ``<data>/config.json``.

Everything the UI can tune lives here (thresholds, devices, poses, safety limits), so a run is reproducible
from its config snapshot, which every cycle record carries.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from sortops.pipeline.cv_anomaly import CV_DEFAULTS
from sortops.pipeline.anomaly_score import SCORE_DEFAULTS, validate_score
from sortops.pipeline.brick_crop import CROP_DEFAULTS, validate_crop
from sortops.pipeline.grasp import GRASP_DEFAULTS

JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
"""SO-101 joint order, the order of every state/action vector (``physicalai_so101_plugin`` order)."""

INSPECTION_CAMERAS: tuple[str, ...] = ("front", "wrist")
"""Cameras that may inspect the brick; each has its own anomaly model (role ``anomaly_<camera>``)."""
ANOMALY_ROLES: tuple[str, ...] = tuple(f"anomaly_{camera}" for camera in INSPECTION_CAMERAS)
ROLES: tuple[str, ...] = ("pick_present", "place_good", "place_bad", *ANOMALY_ROLES, "reinspect")
POLICY_ROLES: tuple[str, ...] = ("pick_present", "place_good", "place_bad", "reinspect")

HOME = Path.home()

DEFAULTS: dict[str, Any] = {
    "hardware": {
        # "mock" simulates the arm, cameras and models; "real" opens the SO-101 and the Studio cameras.
        "backend": "mock",
        # "control" drives the follower; "read_only" never writes torque or goals (watch + shadow only).
        "arm_access": "control",
        "follower_port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045645-if00",
        "calibration": str(HOME / ".cache/huggingface/lerobot/calibration/robots/so_follower/Junction2026.json"),
        "studio_url": "http://127.0.0.1:7860",
        "studio_project": "374f558c-2968-43ab-9fe9-fbd2f1e1cac5",
        "studio_backend_src": str(HOME / "hackathon/physical-ai-studio/application/backend/src"),
        "studio_models_dir": str(HOME / ".local/share/physicalai/models"),
        "studio_datasets_dir": str(HOME / ".local/share/physicalai/datasets"),
        "anomaly_model_roots": [str(HOME / "hackathon/anomaly-models"), str(HOME / "hackathon/anomalib-studio-data")],
        # Python of Anomalib's own venv, used by Data -> Train (sortops/anomaly_worker.py); models land in the 1st root.
        "anomalib_python": str(HOME / "hackathon/anomalib/application/backend/.venv/bin/python"),
        "urdf_dir": str(
            HOME
            / "hackathon/physical-ai-studio/application/backend/.venv/lib/python3.13/site-packages/urdf/so101-follower"
        ),
        "cameras": ["front", "overhead", "wrist"],
    },
    # How each policy's chunks are played, as in Physical AI Studio's "Run model": sync/async, action rate (Hz),
    # request the next chunk when the queue falls below this fraction of a chunk. Tune per model.
    "execution": {
        "pick_present": {"mode": "sync", "hz": 30, "request_threshold": 0.5},
        "place_good": {"mode": "sync", "hz": 30, "request_threshold": 0.5},
        "place_bad": {"mode": "sync", "hz": 30, "request_threshold": 0.5},
        "reinspect": {"mode": "sync", "hz": 30, "request_threshold": 0.5},
    },
    # Measured on the Core Ultra X9 388H (ACT exports): NPU p50 ≈ 35 ms, iGPU ≈ 70 ms, CPU ≈ 200 ms.
    "devices": {
        "pick_present": "NPU",
        "place_good": "NPU",
        "place_bad": "NPU",
        "anomaly_front": "GPU",
        "anomaly_wrist": "GPU",
        "reinspect": "NPU",
    },
    "roles": {
        # Studio datasets whose models may fill each role; the newest, most-trained one is offered first.
        "pick_present": ["001-onecube", "001-onecube-nodepth"],
        "place_good": ["delivery-success", "delivery-success-nodepth", "neo-success"],
        "place_bad": ["delivery-default", "delivery-default-nodepth", "neo-discard"],
        "reinspect": [],
    },
    # Studio model names (regex) also offered for a role, whatever their dataset: the act-00x-… and pi… pick series.
    "role_name_patterns": {"pick_present": [r"^act-00\d-", r"^pi"], "place_good": [r"^neo-success"], "place_bad": [r"^neo-discard"]},
    # Release reflex: in a place segment, once the policy's gripper command reaches ``trigger`` (it starts opening over
    # the bowl), SortOps freezes the other joints where they are, opens the gripper to ``open``, waits ``hold_s`` once
    # it is open (or ``max_open_s`` at most) and ends the segment: the demos open only to ~20-28, too little to drop.
    "release": {
        "enabled": True,
        "place_good": True,
        "place_bad": True,
        "trigger": 10.0,
        "open": 100.0,
        "open_tolerance": 15.0,
        "hold_s": 0.3,
        "max_open_s": 1.5,
        "min_s": 1.0,
    },
    # Present step: after the pick, replay one recorded episode as is (no model), then inspect and place as usual.
    # Off by default: Pick & present is one model; on, a recorded episode presents the brick after the pick.
    "present": {
        "enabled": False,
        "dataset": "present-cube-v1",  # Studio dataset name (or its folder id)
        "episode": 0,
        "approach_speed": 30.0,  # units/s on the farthest joint, to reach the episode's first pose
        "playback_speed": 1.0,  # 1 = as recorded
        # Pauses of the recording (no joint moving `pause_epsilon` units) are skipped: same path, less waiting. 0 = off.
        "pause_epsilon": 0.3,
        # The pick ends once the gripper, after opening above `pick_open_above`, is closed (below `pick_closed_below`,
        # or `pick_close_drop` below its widest opening: a brick stops the jaws early) and has not moved for
        # `pick_closed_s` (the brick is held): no waiting for the pick model to stand still.
        "pick_open_above": 15.0,
        "pick_closed_below": 8.0,
        "pick_close_drop": 10.0,
        "pick_closed_s": 0.8,
        # After the glide, at most this long for the arm to reach the episode's first pose before replaying anyway.
        "approach_settle_s": 1.0,
    },
    # Place without a model: replay one recorded deposit demo per bin (the place ACT models open too late or never).
    "place_replay": {
        "enabled": False,
        "place_good": {"dataset": "delivery-success-release-fix-open", "episode": 0},
        "place_bad": {"dataset": "delivery-default-release-fix-open", "episode": 10},
    },
    "segments": {
        # A segment ends when the arm has been still for still_s after at least min_s, or after timeout_s.
        # (Place with the release reflex on: at the timeout the gripper still opens, so the brick is never carried over.)
        # pick_present: timeout_s = time allowed to grasp a brick (then home, retry); see "pick" for what follows.
        "pick_present": {"min_s": 4.0, "still_s": 0.8, "timeout_s": 20.0},
        "place_good": {"min_s": 2.0, "still_s": 0.8, "timeout_s": 20.0},
        "place_bad": {"min_s": 2.0, "still_s": 0.8, "timeout_s": 20.0},
        "reinspect": {"min_s": 2.0, "still_s": 0.8, "timeout_s": 30.0},
        "still_speed": 1.5,
    },
    "inspection": {
        # The wrist camera: the Anomalib training photos were taken with it (capture app), the brick fills its view.
        "camera": "wrist",
        # Cameras scored at each inspection frame, each by its own model (anomaly_<camera>); their scores are fused.
        "cameras": ["front", "wrist"],
        # Search zone of the brick crop per camera; the "camera" above uses "roi" (shared with the capture app).
        "rois": {"front": [0.0, 0.0, 1.0, 1.0], "wrist": [0.0, 0.0, 1.0, 1.0]},
        # What each camera's model is fed: "crop" (the brick crop, like the capture app photos) or "frame" (the whole
        # image, for a model trained on full frames). Usability is always judged on the crop.
        "model_input": {"front": "crop", "wrist": "crop"},
        # Normalized search zone (x, y, w, h) of the inspection camera: the brick is looked for in it, then cropped by
        # sortops.pipeline.brick_crop (crop_* keys below), the same code the Anomalib capture app cuts its photos with.
        "roi": [0.0, 0.0, 1.0, 1.0],
        **CROP_DEFAULTS,
        # Anomaly score = quadratic combination of a localized peak and the whole-map mean (pipeline/anomaly_score.py).
        # Recalibrate (Models tab) after changing these: the calibration stores the values it was computed with.
        **SCORE_DEFAULTS,
        # Gripper position (x, y, z in metres, URDF base frame) where the brick faces the inspection camera.
        "pose_xyz": [0.33, 0.0, 0.215],
        "pose_tolerance_m": 0.05,
        "period_s": 0.5,
        # Inspection lasts at least this long from the first frame at the pose (covers the 180° wrist turn).
        "window_s": 3.5,
        "min_usable_frames": 2,
        # One frame per camera: only the most usable frame of each camera is scored and kept (operator prompt, data),
        # instead of every frame of the window. Off: every usable frame is scored (min_usable_frames applies).
        "best_frame_only": True,
        # "model": the imported Anomalib model of each camera; "cv": no model, colour auto-crop + stain / dark-mark
        # share against cv_stain_max / cv_mark_max (sortops/pipeline/cv_anomaly.py), score 0.5 at the threshold.
        "anomaly_method": "model",
        # Cameras whose CV score decides (the wrist view still catches background with the brick; front is clean).
        "cv_cameras": ["front"],
        # Cameras whose Anomalib model scores (front: PatchCore trained on the tight front crops; the wrist model saw
        # other crops and scores everything 1.0).
        "anomalib_cameras": ["front"],
        **CV_DEFAULTS,
        "usability_threshold": 0.55,
        "min_sharpness": 60.0,
        # Brick pixels (crop mask) / search-zone pixels under which a frame counts as "no brick".
        "min_brick_fraction": 0.01,
        "max_speed_for_capture": 6.0,
        "aggregate": "p90",
    },
    "decision": {
        # "model": the anomaly model decides; "operator": no model yet, the operator says good / defect in the UI.
        "verdict_source": "model",
        # How the per-camera scores become one: max (defect if any camera sees one), mean, min (all must), weighted.
        "fusion": "max",
        "weights": {"front": 0.5, "wrist": 0.5},
        # Anomalib normalizes scores so 0.5 is its own threshold; good below good_below, defective above bad_above.
        "good_below": 0.42,
        "bad_above": 0.58,
        "recapture_frames": 4,
        "recapture_period_s": 0.3,
        "reinspect_enabled": False,
        # The arm presented but no brick was recognized in at least this share of the frames: defect bin.
        "no_brick_defect_ratio": 0.5,
    },
    "safety": {
        "soft_min": [-95.0, -95.0, -95.0, -95.0, -100.0, 0.0],
        "soft_max": [95.0, 95.0, 95.0, 95.0, 100.0, 100.0],
        # Largest change of a joint target per control tick (normalized units); 6 at 30 fps = 180 units/s.
        "max_step": 6.0,
        "watchdog_s": 2.0,
        "watchdog_enabled": True,
        "max_motor_temp_c": 65.0,
        # The overheat stop compares the per-motor median over this window, not a single read.
        "temp_window_s": 5.0,
    },
    # Step 1 (pick & present) → inspection: explicit gates, shown live in the Cockpit (snapshot "transition").
    "pick": {
        # Grasp read from the gripper motor (sortops.pipeline.grasp): closed below open_below, not below
        # min_closed_pos (closed on nothing), load ≥ min_load (signed: opening reads negative) for hold_s.
        # Lost after lost_s without grip force.
        **GRASP_DEFAULTS,
        # Without a grasp, no frame is sampled and the cycle never leaves step 1 (false: sample from the start).
        "require_grasp": True,
        # Admission zone around inspection.pose_xyz: radius 2 × pose_tolerance_m, + zone_growth (0.5 = +50 %) every
        # zone_step_frames admissible frames (a brick seen by an inspection camera), at most + zone_max_growth.
        "zone_step_frames": 10,
        "zone_growth": 0.5,
        "zone_max_growth": 2.0,
        # Inspection starts once the brick is held, min_admissible_in_zone admissible frames were taken in the zone and
        # inspection.window_s passed since the first of them (no need for the arm to stand still).
        "min_admissible_in_zone": 3,
        # Held but never admitted: go on to the inspection anyway this long after the grasp (logged as such).
        "present_max_s": 15.0,
        # No brick held within segments.pick_present.timeout_s: home, wait retry_wait_s, pick again.
        "retry_wait_s": 0.5,
    },
    # Batch end: this many cycles in a row without a recognized brick (empty supply, missed picks) stop the run.
    # pick_attempts: picks tried per brick (home between two, 0 = no limit) before the cycle counts as "no brick".
    "run": {"loop": True, "max_cycles": 0, "stop_after_no_brick": 3, "pick_attempts": 3},
    # Pose of the "Home" button (normalized joints, folded rest pose; teach it in Settings) and its joint speed (units/s).
    "home": {"joints": [0.0, -95.0, 95.0, 60.0, 0.0, 2.0], "speed": 40.0},
}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Settings:
    """Thread-safe settings store. ``get()`` returns a deep copy; ``update()`` merges and persists."""

    def __init__(self, data_dir: Path, overrides: dict[str, Any] | None = None) -> None:
        self.data_dir = data_dir
        self.path = data_dir / "config.json"
        self._lock = threading.Lock()
        stored: dict[str, Any] = {}
        if self.path.exists():
            stored = json.loads(self.path.read_text())
        self._values = _merge(_merge(DEFAULTS, stored), overrides or {})
        try:
            # A hand-edited or older config.json must not reach the arm loop unchecked.
            validate(self._values)
        except ValueError as error:
            raise ValueError(f"{self.path}: {error}") from error

    def get(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._values)

    def section(self, name: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._values[name])

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        unknown = [key for key in patch if key not in DEFAULTS]
        if unknown:
            raise ValueError(f"Unknown settings sections: {unknown}")
        with self._lock:
            merged = _merge(self._values, patch)
            validate(merged)
            self._values = merged
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._values, indent=2))
            tmp.replace(self.path)
            return copy.deepcopy(self._values)


# (min, max) of every bounded number; a value outside refuses the whole update.
BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("inspection", "pose_tolerance_m"): (0.005, 0.3),
    ("inspection", "period_s"): (0.05, 5),
    ("inspection", "window_s"): (0, 30),
    ("inspection", "min_usable_frames"): (1, 50),
    ("inspection", "usability_threshold"): (0, 1),
    ("inspection", "min_sharpness"): (0, 10_000),
    ("inspection", "min_brick_fraction"): (0, 1),
    ("inspection", "max_speed_for_capture"): (0.1, 200),
    ("decision", "good_below"): (0, 1),
    ("decision", "bad_above"): (0, 1),
    ("decision", "recapture_frames"): (0, 20),
    ("decision", "recapture_period_s"): (0.05, 5),
    ("safety", "max_step"): (0.1, 15),
    ("release", "trigger"): (1, 95),
    ("release", "open"): (5, 100),
    ("release", "open_tolerance"): (1, 60),
    ("release", "hold_s"): (0, 5),
    ("release", "max_open_s"): (0.2, 10),
    ("release", "min_s"): (0, 20),
    ("safety", "watchdog_s"): (0.3, 10),
    ("safety", "max_motor_temp_c"): (30, 120),
    ("safety", "temp_window_s"): (1, 60),
    ("run", "max_cycles"): (0, 10_000),
    ("home", "speed"): (5, 120),
    ("present", "episode"): (0, 100_000),
    ("present", "approach_speed"): (5, 120),
    ("present", "playback_speed"): (0.25, 3),
    ("present", "pause_epsilon"): (0, 5),
    ("present", "pick_open_above"): (1, 100),
    ("present", "pick_closed_below"): (0, 99),
    ("present", "pick_closed_s"): (0, 10),
    ("present", "pick_close_drop"): (1, 100),
    ("present", "approach_settle_s"): (0, 10),
    ("run", "stop_after_no_brick"): (1, 100),
    ("run", "pick_attempts"): (0, 100),
    ("pick", "open_below"): (3, 100),
    ("pick", "min_closed_pos"): (0, 50),
    ("pick", "min_load"): (1, 100),
    ("pick", "hold_s"): (0, 5),
    ("pick", "lost_s"): (0.05, 5),
    ("pick", "zone_step_frames"): (1, 1000),
    ("pick", "zone_growth"): (0, 5),
    ("pick", "zone_max_growth"): (0, 20),
    ("pick", "min_admissible_in_zone"): (1, 100),
    ("pick", "present_max_s"): (1, 120),
    ("pick", "retry_wait_s"): (0, 10),
    ("decision", "no_brick_defect_ratio"): (0.05, 1),
}
JOINT_RANGE = ([-100.0] * 5 + [0.0], [100.0] * 6)


def _same_type(default: Any, value: Any) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value  # noqa: PLR0124 - NaN
    return isinstance(value, type(default))


def validate(values: dict[str, Any]) -> None:
    """Refuse settings of the wrong type or outside safe bounds (motion limits above all)."""
    for section, defaults in DEFAULTS.items():
        for key, default in defaults.items():
            value = values[section].get(key)
            if not _same_type(default, value):
                raise ValueError(f"{section}.{key} must be a {type(default).__name__}")
    for (section, key), (low, high) in BOUNDS.items():
        value = values[section][key]
        if not low <= value <= high:
            raise ValueError(f"{section}.{key} must be within [{low}, {high}], got {value}")
    safety = values["safety"]
    soft_min, soft_max = safety["soft_min"], safety["soft_max"]
    for name, limits in (("soft_min", soft_min), ("soft_max", soft_max)):
        if len(limits) != len(JOINTS) or not all(_same_type(0.0, v) for v in limits):
            raise ValueError(f"safety.{name} must be {len(JOINTS)} numbers")
    for i, joint in enumerate(JOINTS):
        if not JOINT_RANGE[0][i] <= soft_min[i] < soft_max[i] <= JOINT_RANGE[1][i]:
            raise ValueError(f"safety soft limits of {joint} must satisfy {JOINT_RANGE[0][i]} ≤ min < max ≤ {JOINT_RANGE[1][i]}")
    home = values["home"]["joints"]
    if len(home) != len(JOINTS) or not all(_same_type(0.0, v) for v in home):
        raise ValueError("home.joints must be 6 numbers")
    for i, joint in enumerate(JOINTS):
        if not safety["soft_min"][i] <= home[i] <= safety["soft_max"][i]:
            raise ValueError(f"home.joints {joint} = {home[i]} is outside the safety soft limits")
    if values["pick"]["min_closed_pos"] >= values["pick"]["open_below"]:
        raise ValueError("pick.min_closed_pos must be below pick.open_below")
    if values["decision"]["good_below"] >= values["decision"]["bad_above"]:
        raise ValueError("decision.good_below must be below decision.bad_above")
    roi = values["inspection"]["roi"]
    if len(roi) != 4 or not all(_same_type(0.0, v) and 0 <= v <= 1 for v in roi):
        raise ValueError("inspection.roi must be 4 numbers in [0, 1]")
    validate_crop(values["inspection"])
    release = values["release"]
    if release["open"] < release["trigger"] + 10:
        # The reflex replaces a command >= trigger by `open`: below it, it would close on the brick instead.
        raise ValueError("release.open must be at least 10 above release.trigger")
    validate_score(values["inspection"])
    if not str(values["present"]["dataset"]).strip():
        raise ValueError("present.dataset must name a dataset")
    if values["present"]["pick_closed_below"] >= values["present"]["pick_open_above"]:
        raise ValueError("present.pick_closed_below must be below present.pick_open_above")
    if len(values["inspection"]["pose_xyz"]) != 3:
        raise ValueError("inspection.pose_xyz must be 3 numbers")
    for role, patterns in values["role_name_patterns"].items():
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"role_name_patterns.{role}: bad regex {pattern!r}: {error}") from error
    for role, exe in values["execution"].items():
        if exe.get("mode") not in {"sync", "async"}:
            raise ValueError(f"execution.{role}.mode must be 'sync' or 'async'")
        hz, threshold = exe.get("hz"), exe.get("request_threshold")
        if not _same_type(0.0, hz) or not 5 <= hz <= 100:
            raise ValueError(f"execution.{role}.hz must be within [5, 100]")
        if not _same_type(0.0, threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"execution.{role}.request_threshold must be within [0, 1]")
    inspection = values["inspection"]
    if not inspection["cameras"] or any(cam not in INSPECTION_CAMERAS for cam in inspection["cameras"]):
        raise ValueError(f"inspection.cameras must be a non-empty subset of {list(INSPECTION_CAMERAS)}")
    for cam, roi in inspection["rois"].items():
        if cam not in INSPECTION_CAMERAS or len(roi) != 4 or not all(_same_type(0.0, v) and 0 <= v <= 1 for v in roi):
            raise ValueError("inspection.rois must map inspection cameras to 4 numbers in [0, 1]")
    for cam, kind in inspection["model_input"].items():
        if cam not in INSPECTION_CAMERAS or kind not in {"crop", "frame"}:
            raise ValueError("inspection.model_input must map inspection cameras to 'crop' or 'frame'")
    if values["decision"]["fusion"] not in {"max", "mean", "min", "weighted"}:
        raise ValueError("decision.fusion must be max, mean, min or weighted")
    for cam, weight in values["decision"]["weights"].items():
        if cam not in INSPECTION_CAMERAS or not _same_type(0.0, weight) or not 0 <= weight <= 1:
            raise ValueError("decision.weights must map inspection cameras to numbers in [0, 1]")
    if values["decision"]["verdict_source"] not in {"model", "cv", "operator"}:
        raise ValueError("decision.verdict_source must be 'model' (Anomalib), 'cv' (Algorithm) or 'operator' (Manual)")
    for role, device in values["devices"].items():
        if device not in {"CPU", "GPU", "NPU"}:
            raise ValueError(f"devices.{role} must be CPU, GPU or NPU")


def data_dir_from_env() -> Path:
    return Path(os.environ.get("SORTOPS_DATA", Path(__file__).resolve().parents[2] / "data"))
