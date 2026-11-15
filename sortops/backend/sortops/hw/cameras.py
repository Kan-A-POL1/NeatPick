"""Latest-frame camera hub. Frames are RGB uint8 HxWx3, keyed like the policy inputs: ``front``, ``front_depth``…

The real hub opens the cameras exactly as Physical AI Studio records them (same factory, undistortion, crop and
depth-to-image), by importing Studio's backend helpers, so a policy sees at inference what it saw in training.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from loguru import logger


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp: float


class CameraHub(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self, key: str) -> Frame | None: ...
    def keys(self) -> list[str]: ...


class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, Frame] = {}

    def put(self, key: str, image: np.ndarray, timestamp: float) -> None:
        with self._lock:
            self._frames[key] = Frame(image, timestamp)

    def get(self, key: str) -> Frame | None:
        with self._lock:
            return self._frames.get(key)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._frames)


class StudioFramePrep:
    """What Physical AI Studio does to a camera frame before recording it: resize to the configured size, undistort
    (calibration file, if any), digital crop. SortOps and the Anomalib capture app both build their frames with it.

    ``raw`` is one row of ``/api/projects/<id>/cameras``; Studio's backend ``src`` must be on ``sys.path``.
    """

    def __init__(self, raw: dict[str, object]) -> None:
        from schemas.project_camera import CameraAdapter  # noqa: PLC0415 - Studio backend on sys.path
        from utils.camera_calibration import load_undistorter  # noqa: PLC0415
        from utils.camera_crop import crop_frame, output_size  # noqa: PLC0415

        self.config = CameraAdapter.validate_python(raw)
        payload = self.config.payload
        self.crop = getattr(payload, "crop", None)
        self.full = (payload.width or 640, payload.height or 480)
        self.size = output_size(self.full[0], self.full[1], self.crop)
        self.undistort = load_undistorter(self.config.name)
        self._crop_frame = crop_frame

    def __call__(self, image: np.ndarray) -> np.ndarray:
        data = image if image.shape[1::-1] == self.full else cv2.resize(image, self.full)
        if self.undistort is not None:
            data = self.undistort(data)
        return self._crop_frame(data, self.crop, self.size)


class StudioCameraHub:
    """Studio's camera configs (``/api/projects/<id>/cameras``) opened with Studio's own capture pipeline."""

    STALE_S = 2.0

    def __init__(self, studio_url: str, project: str, backend_src: str, names: list[str]) -> None:
        if backend_src not in sys.path:
            sys.path.insert(0, backend_src)
        self._url = f"{studio_url.rstrip('/')}/api/projects/{project}/cameras"
        self._names = names
        self._store = _Store()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.errors: dict[str, str] = {}

    def start(self) -> None:
        with urllib.request.urlopen(self._url, timeout=5) as response:  # noqa: S310 - local Studio API
            configs = json.loads(response.read())
        by_name = {item["name"]: item for item in configs}
        missing = [name for name in self._names if name not in by_name]
        if missing:
            raise RuntimeError(f"Studio project has no camera named {missing}; it has {sorted(by_name)}")
        for name in self._names:
            thread = threading.Thread(target=self._run, args=(by_name[name],), name=f"cam-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _run(self, raw: dict[str, object]) -> None:
        from utils.camera_crop import crop_frame  # noqa: PLC0415 - Studio backend on sys.path
        from utils.camera_factory import build_shared_camera  # noqa: PLC0415
        from workers.camera_worker import depth_to_image, records_depth  # noqa: PLC0415

        prep = StudioFramePrep(raw)
        config = prep.config
        name = config.name
        payload = config.payload
        crop, full, size, undistort = prep.crop, prep.full, prep.size, prep.undistort
        with_depth = records_depth(config)
        while not self._stop.is_set():
            try:
                # Attach without reconfiguring: a Studio session may share the same publisher.
                camera = build_shared_camera(config=config, validate_on_connect=False, overwrite_settings=False)
                camera.connect()
            except Exception as error:  # noqa: BLE001 - reported to the UI, retried
                self.errors[name] = str(error)
                logger.warning("camera {} failed to open: {}", name, error)
                self._stop.wait(2.0)
                continue
            self.errors.pop(name, None)
            previous = None
            last_new = time.monotonic()
            try:
                while not self._stop.is_set():
                    frame = camera.read_latest()
                    if frame is not None and frame is not previous:
                        previous, last_new = frame, time.monotonic()
                        self._store.put(name, prep(frame.data), time.monotonic())
                        if with_depth:
                            depth = camera.read_latest_depth()
                            if depth is not None:
                                image = depth_to_image(depth.data, payload.depth_range_min, payload.depth_range_max)
                                if image.shape[1::-1] != full:
                                    image = cv2.resize(image, full, interpolation=cv2.INTER_NEAREST)
                                if undistort is not None:
                                    image = undistort(image)
                                image = crop_frame(image, crop, size, interpolation=cv2.INTER_NEAREST)
                                self._store.put(f"{name}_depth", image, time.monotonic())
                    elif time.monotonic() - last_new > self.STALE_S:
                        self.errors[name] = "no frames, reconnecting"
                        break
                    self._stop.wait(1.0 / max(1, payload.fps or 30))
            finally:
                try:
                    camera.disconnect()
                except Exception:  # noqa: BLE001
                    logger.exception("camera {} disconnect failed", name)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3)

    def latest(self, key: str) -> Frame | None:
        return self._store.get(key)

    def keys(self) -> list[str]:
        return self._store.keys()


class SimCameraHub:
    """Synthetic cameras rendered from the simulated world, so the whole loop runs image → score → decision."""

    def __init__(self, render: Callable[[str], np.ndarray], names: list[str], fps: float = 15.0) -> None:
        self._render = render
        self._names = names
        self._fps = fps
        self._store = _Store()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sim-cams", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            for name in self._names:
                self._store.put(name, self._render(name), now)
            self._stop.wait(1.0 / self._fps)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def latest(self, key: str) -> Frame | None:
        return self._store.get(key)

    def keys(self) -> list[str]:
        return self._store.keys()


def encode_jpeg(image_rgb: np.ndarray, quality: int = 75) -> bytes:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
