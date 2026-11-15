"""Inference back ends: ACT policies through ``physicalai.inference.InferenceModel`` and Anomalib exports through
the OpenVINO runtime, plus the simulated stand-ins. Every call is timed into ``LatencyBook`` under its role.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import cv2
import numpy as np
from loguru import logger

from sortops.hw.sim import ScriptedPolicy, sim_anomaly_score
from sortops.metrics import LatencyBook
from sortops.pipeline.anomaly_score import combined, map_terms, term_references


class Policy(Protocol):
    role: str
    label: str
    device: str
    camera_keys: list[str]

    def reset(self) -> None: ...
    def predict(self, state: np.ndarray, images: dict[str, np.ndarray], executed: int) -> np.ndarray: ...


class AnomalyModel(Protocol):
    label: str
    device: str
    role: str

    def score(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]: ...


def manifest_images(export_dir: Path) -> dict[str, tuple[int, int]]:
    """Image inputs of a Studio export (``images.front`` → (H, W)) read from its ``manifest.json``."""
    manifest = json.loads((export_dir / "manifest.json").read_text())
    images: dict[str, tuple[int, int]] = {}
    for feature in manifest["model"]["input_features"]:
        args = feature["init_args"]
        if args["ftype"] == "VISUAL":
            _, height, width = args["shape"]
            images[args["name"]] = (int(height), int(width))
    return images


class OpenVinoPolicy:
    """An exported Studio policy (``policy_package``) run with ``InferenceModel`` on CPU, GPU or NPU."""

    def __init__(self, role: str, label: str, export_dir: Path, device: str, book: LatencyBook) -> None:
        from physicalai.inference import InferenceModel  # noqa: PLC0415 - only on the robot PC

        self.role = role
        self.label = label
        self.device = device
        self._book = book
        self._inputs = manifest_images(export_dir)
        self.camera_keys = [name.removeprefix("images.") for name in self._inputs]
        started = time.perf_counter()
        self._model = InferenceModel(export_dir, backend="openvino", device=device)
        book.record(f"load:{role}", (time.perf_counter() - started) * 1000.0, device)
        self._warned: set[str] = set()

    def reset(self) -> None:
        self._model.reset()

    def predict(self, state: np.ndarray, images: dict[str, np.ndarray], executed: int) -> np.ndarray:
        _ = executed
        observation: dict[str, np.ndarray] = {"state": np.asarray(state, np.float32)[np.newaxis]}
        for name, (height, width) in self._inputs.items():
            key = name.removeprefix("images.")
            image = images[key]
            if image.shape[:2] != (height, width):
                if key not in self._warned:
                    logger.warning("{}: camera {} is {}, export expects {}", self.role, key, image.shape[:2], (height, width))
                    self._warned.add(key)
                image = cv2.resize(image, (width, height))
            observation[name] = image[np.newaxis]
        started = time.perf_counter()
        chunk = self._model.predict_action_chunk(observation)
        self._book.record(self.role, (time.perf_counter() - started) * 1000.0, self.device)
        return np.asarray(chunk, np.float32)


class SimPolicy:
    def __init__(self, role: str, fps: float, device: str, book: LatencyBook) -> None:
        self.role = role
        self.label = f"sim-{role}"
        self.device = device
        self.camera_keys = ["front", "overhead", "wrist"]
        self._inner = ScriptedPolicy(role, fps)
        self._book = book
        # Simulated per-device latency, in the range measured for small ACT exports.
        self._latency_ms = {"CPU": 38.0, "GPU": 21.0, "NPU": 14.0}.get(device.upper(), 30.0)

    def reset(self) -> None:
        self._inner.reset()

    def predict(self, state: np.ndarray, images: dict[str, np.ndarray], executed: int) -> np.ndarray:
        _ = images
        started = time.perf_counter()
        chunk = self._inner.predict_action_chunk(state, executed)
        time.sleep(self._latency_ms * float(np.random.default_rng().uniform(0.85, 1.25)) / 1000.0)
        self._book.record(self.role, (time.perf_counter() - started) * 1000.0, self.device)
        return chunk


def raw_score_output(model: object, output: str = "pred_score") -> object | None:
    """The raw anomaly score inside an Anomalib export, before its built-in normalization.

    Anomalib 2.x graphs end with ``pred_score = clip((raw - min) / (max - min) …)`` around the threshold learnt at
    training; the raw score is the tensor compared to that threshold constant (first Greater/Less with a scalar
    Constant input, walking up from ``pred_score``). ``output="anomaly_map"`` finds the raw map the same way (it is
    compared to the pixel threshold). ``None`` when the graph does not look like that.
    """
    outputs = [o for o in model.outputs if output in o.get_names()]  # type: ignore[attr-defined]
    if not outputs:
        return None
    affine = _affine_input(outputs[0].get_node().input_value(0).get_node())  # Result → its producer
    if affine is not None:
        return affine
    frontier = [outputs[0].get_node()]
    seen: set[str] = set()
    for _ in range(16):
        following = []
        for node in frontier:
            for i in range(node.get_input_size()):
                source = node.input_value(i)
                parent = source.get_node()
                if parent.get_friendly_name() in seen:
                    continue
                seen.add(parent.get_friendly_name())
                if parent.get_type_name() in {"Greater", "Less"}:
                    inputs = [parent.input_value(k) for k in range(parent.get_input_size())]
                    constants = [x for x in inputs if x.get_node().get_type_name() == "Constant"]
                    others = [x for x in inputs if x.get_node().get_type_name() != "Constant"]
                    if len(constants) == 1 and len(others) == 1 and constants[0].get_node().get_data().size == 1:
                        return others[0]
                following.append(parent)
        frontier = following
    return None


def _affine_input(node: object) -> object | None:
    """``pred_score = clamp(raw * a + b)`` (FRE-style exports): walk Clamp → Add(const) → Multiply(const) to ``raw``."""
    current = node
    if current.get_type_name() == "Clamp":  # type: ignore[attr-defined]
        current = current.input_value(0).get_node()  # type: ignore[attr-defined]
    for op in ("Add", "Multiply"):
        if current.get_type_name() != op:  # type: ignore[attr-defined]
            return None
        inputs = [current.input_value(i) for i in range(current.get_input_size())]  # type: ignore[attr-defined]
        others = [x for x in inputs if x.get_node().get_type_name() != "Constant"]
        if len(others) != 1:
            return None
        if op == "Multiply":
            return others[0]
        current = others[0].get_node()
    return None


def calibrated(raw: float, calibration: dict[str, float]) -> float:
    """Raw score on SortOps' own scale: 0 = median good brick, 0.5 = ``threshold``, 1 = threshold + spread."""
    return float(np.clip(0.5 + (raw - calibration["threshold"]) / (2.0 * calibration["spread"]), 0.0, 1.0))


def score_output(out: AnomalyOutput, calibration: dict[str, Any] | None) -> float:
    """SortOps' 0..1 score of one inference: the peak+mean combination when calibrated with its term references
    (sortops.pipeline.anomaly_score), else the calibrated raw score, else the export's own pred_score."""
    if calibration is not None and calibration.get("terms") and out.raw_map is not None:
        terms = calibration["terms"]
        peak, mean = map_terms(out.raw_map, terms["peak_fraction"], out.raw if terms.get("peak_from_raw") else None)
        return calibrated(combined(peak, mean, terms, terms["peak_weight"]), calibration)
    if calibration is not None and not calibration.get("terms") and out.raw is not None:
        return calibrated(out.raw, calibration)
    return out.exported


def calibration_for(goods: list[AnomalyOutput], peak_weight: float, peak_fraction: float) -> dict[str, Any]:
    """SortOps' scale from inferences on known-good crops. With raw anomaly maps: references of the peak and mean
    terms, then the 0..1 scale of their quadratic combination (sortops.pipeline.anomaly_score); else the raw score's."""
    if goods and all(o.raw_map is not None for o in goods):
        # The peak is the raw image score when every inference has one (same source at scoring time).
        peak_from_raw = all(o.raw is not None for o in goods)
        terms_of = [map_terms(o.raw_map, peak_fraction, o.raw if peak_from_raw else None) for o in goods]  # type: ignore[arg-type]
        terms: dict[str, Any] = {
            **term_references([t[0] for t in terms_of], [t[1] for t in terms_of]),
            "peak_weight": peak_weight,
            "peak_fraction": peak_fraction,
            "peak_from_raw": peak_from_raw,
        }
        return {**calibration_from([combined(p, m, terms, peak_weight) for p, m in terms_of]), "terms": terms}
    raws = [o.raw for o in goods if o.raw is not None]
    if not raws:
        raise RuntimeError("The anomaly model gave no raw score or raw map to calibrate on")
    return calibration_from(raws)


def calibration_from(good_raw: list[float]) -> dict[str, float]:
    """Scale from raw scores of known-good crops: the 95th percentile sits at 0.25, twice that margin above it at 0.5."""
    values = np.asarray(good_raw, np.float64)
    median = float(np.median(values))
    high = float(np.percentile(values, 95))
    spread = max(high - median, 0.05 * abs(median), 1e-6)
    return {"median": median, "p95": high, "threshold": high + spread, "spread": 2.0 * spread, "n": float(len(values))}


class AnomalyOutput(NamedTuple):
    exported: float
    """The export's own pred_score."""
    raw: float | None
    """Raw image score inside the graph, before normalization (None when not found)."""
    heat: np.ndarray
    """Normalized anomaly map at the image's size, 0..1 (display)."""
    raw_map: np.ndarray | None
    """Raw anomaly map before normalization (None when not found): input of sortops.pipeline.anomaly_score."""


class OpenVinoAnomaly:
    """An Anomalib OpenVINO export (``model.xml``): pre-processing is part of Anomalib 2.x graphs.

    Its built-in score normalization comes from the export's validation set and can be off (a PatchCore trained on
    goods only scored its own training crops 1.0), so SortOps reads the raw score too and, once calibrated on
    known-good bricks, uses its own scale. Uncalibrated, the export's ``pred_score`` is used as is.
    """

    def __init__(
        self,
        label: str,
        xml: Path,
        device: str,
        book: LatencyBook,
        calibration: dict[str, float] | None = None,
        role: str = "anomaly",
    ) -> None:
        import openvino as ov  # noqa: PLC0415 - only on the robot PC

        self.label = label
        self.role = role
        self.device = device
        self.calibration = calibration
        self._book = book
        core = ov.Core()
        model = core.read_model(str(xml))
        raw = raw_score_output(model)
        self.has_raw = raw is not None
        self._raw_names: set[str] = set()
        if raw is not None:
            raw.get_tensor().add_names({"sortops_raw_score"})
            # Its own names too: a reshape or a compile may keep only the original ones.
            self._raw_names = {"sortops_raw_score", *raw.get_names()}
            model.add_outputs([raw])
        raw_map = raw_score_output(model, "anomaly_map")
        self.has_raw_map = raw_map is not None
        if raw_map is not None:
            raw_map.get_tensor().add_names({"sortops_raw_map"})
            model.add_outputs([raw_map])
        shape = model.inputs[0].get_partial_shape()
        # Some Anomalib Studio exports freeze the batch (e.g. [2,3,?,?]): keep it, feed copies, read the first result.
        self._batch = shape[0].get_length() if shape[0].is_static else 1
        if shape.is_dynamic:
            # NPU needs static shapes; 256x256 is Anomalib's default training size (and the brick crop's size).
            model.reshape({model.inputs[0].get_any_name(): [self._batch, 3, 256, 256]})
        dims = model.inputs[0].get_shape()
        self._size = (int(dims[3]), int(dims[2]))
        started = time.perf_counter()
        # The iGPU runs FP16 by default: anomaly distances (sums of squared feature errors) overflow to inf there.
        config = {"INFERENCE_PRECISION_HINT": "f32"} if device.upper().startswith("GPU") else {}
        self._compiled = core.compile_model(model, device, config)
        book.record(f"load:{role}", (time.perf_counter() - started) * 1000.0, device)
        self._lock = threading.Lock()

    def infer(self, image_rgb: np.ndarray) -> AnomalyOutput:
        """Export pred_score, raw score, display heat map and raw anomaly map of one crop."""
        tensor = cv2.resize(image_rgb, self._size).astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]
        if self._batch > 1:
            tensor = np.repeat(tensor, self._batch, axis=0)
        started = time.perf_counter()
        with self._lock:
            result = self._compiled(tensor)
        self._book.record(self.role, (time.perf_counter() - started) * 1000.0, self.device)
        # A port has several names ("pred_score" can come with "clamp"): index every one of them.
        # First item of the batch only (the others are copies of the same crop).
        outputs = {name: np.asarray(value)[:1] for port, value in result.items() for name in port.get_names()}
        if "pred_score" not in outputs:
            raise RuntimeError(f"Anomaly model outputs {sorted(outputs)} carry no pred_score")
        heat_value = outputs.get("anomaly_map")
        heat = np.squeeze(heat_value).astype(np.float32) if heat_value is not None else np.zeros((1, 1), np.float32)
        heat = cv2.resize(heat, (image_rgb.shape[1], image_rgb.shape[0]))
        raw = next((outputs[name] for name in self._raw_names if name in outputs), None)
        raw_map = outputs.get("sortops_raw_map")
        return AnomalyOutput(
            float(np.squeeze(outputs["pred_score"])),
            None if raw is None else float(np.squeeze(raw)),
            np.clip(heat, 0.0, 1.0),
            None if raw_map is None else np.squeeze(raw_map).astype(np.float32),
        )

    def score(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        out = self.infer(image_rgb)
        return score_output(out, self.calibration), out.heat


class SimAnomaly:
    def __init__(self, device: str, book: LatencyBook, role: str = "anomaly") -> None:
        self.label = "sim-anomaly"
        self.role = role
        self.calibration: dict[str, float] | None = None
        self.device = device
        self._book = book
        self._latency_ms = {"CPU": 24.0, "GPU": 9.0, "NPU": 7.0}.get(device.upper(), 15.0)

    def score(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        started = time.perf_counter()
        result = sim_anomaly_score(image_rgb)
        remaining = self._latency_ms / 1000.0 - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        self._book.record(self.role, (time.perf_counter() - started) * 1000.0, self.device)
        return result
