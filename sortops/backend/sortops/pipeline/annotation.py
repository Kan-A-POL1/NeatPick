"""Inspection data kept for good: every inspection frame of every cycle, the operator's (or model's) annotation of the
brick, and an Anomalib-ready export.

On disk, per run (no size limit; nothing is ever pruned)::

    <data>/runs/<run>/inspection/c0007/
        shots.jsonl                  one line per frame: camera, usability, score, boxes, heat info, file names
        front_00042_frame.jpg        the whole camera frame (q95)
        front_00042_crop.png         what the model scored (brick crop, or the frame for a whole-frame model)
        front_00042_heat.png         readable anomaly map (sortops.pipeline.heatmap), when a model ran
        front_00042_raw.npy          raw anomaly map (float16), when the model exposes it
        annotation.json              verdict + per camera: reference frame, piece box, defect shapes
        review.json                  per-image review (Data -> Review): piece box, defects, excluded, per shot id

Annotation (all coordinates normalized to the camera FRAME, so the piece box can move without redrawing defects)::

    {"verdict": "good" | "bad" | "no_piece", "source": "operator" | "model", "updated": 1731600000.0, "note": "",
     "cameras": {"wrist": {"shot": 42, "piece": [x, y, w, h],
                           "defects": [{"type": "rect", "box": [x, y, w, h]},
                                       {"type": "stroke", "points": [[x, y], ...], "width": 0.02}]}}}

Review (``review_queue`` / ``save_review``): every stored frame is gone through one by one; the operator confirms the
piece box (the only zone analysed) and, on defect cycles, draws the defects of THAT frame (the brick turns, they move)::

    review.json = {"shots": {"42": {"piece": [x, y, w, h], "defects": [...], "excluded": false, "updated": t}}}

With ``reviewed_only`` the export takes only reviewed, non-excluded frames: good -> train/good (1 in 5 -> test/good),
defect -> test/defect + its mask (or test/defect_nomask when nothing is drawn on that frame).

Legacy export (``reviewed_only=False``), Anomalib Folder / MVTec layout per camera, crops cut to the piece box, square, ``size`` px::

    <run>/<camera>/train/good/c0007_s00042.png     every usable frame of a good cycle (auto brick box)
    <run>/<camera>/test/good/…                     the reference frame of each good cycle (operator piece box)
    <run>/<camera>/test/defect/c0008_s00051.png    the reference frame of each defect cycle
    <run>/<camera>/ground_truth/defect/c0008_s00051.png   (same name as the image: Anomalib pairs them by name)
    <run>/<camera>/test/defect_nomask/…            defect cycle, nothing drawn on this camera (no mask)
    <run>/<camera>/full/c0008_s00051.jpg           whole frame
    <run>/annotations.jsonl, README.txt
"""

from __future__ import annotations

import io
import json
import math
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VERDICTS = ("good", "bad", "no_piece")
CAMERA_NAME_OK = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


def _box(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("a box is [x, y, w, h]")
    x, y, w, h = (float(v) for v in value)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("a box needs finite values and a positive size")
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    return [round(x, 5), round(y, 5), round(min(w, 1.0 - x), 5), round(min(h, 1.0 - y), 5)]


def _defects(raw: Any) -> list[dict[str, Any]]:
    defects = []
    for shape in raw or []:
        if not isinstance(shape, dict):
            raise ValueError("a defect is an object")
        if shape.get("type") == "rect":
            defects.append({"type": "rect", "box": _box(shape.get("box"))})
        elif shape.get("type") == "stroke":
            points = [[round(min(max(float(x), 0.0), 1.0), 5), round(min(max(float(y), 0.0), 1.0), 5)]
                      for x, y in shape.get("points") or []]
            if not points:
                continue
            width = float(shape.get("width", 0.02))
            if not 0 < width <= 0.5:
                raise ValueError("stroke width must be within ]0, 0.5]")
            defects.append({"type": "stroke", "points": points, "width": round(width, 5)})
        else:
            raise ValueError("a defect is a 'rect' or a 'stroke'")
    return defects


def validate_review(raw: dict[str, Any]) -> dict[str, Any]:
    """Clean one frame's review from the UI: ``{"piece": box | None, "defects": [...], "excluded": bool}``."""
    excluded = bool(raw.get("excluded", False))
    piece = None if raw.get("piece") is None else _box(raw["piece"])
    if piece is None and not excluded:
        raise ValueError("a kept image needs its piece box (or mark it excluded)")
    return {"piece": piece, "defects": _defects(raw.get("defects")), "excluded": excluded}


def validate_annotation(raw: dict[str, Any]) -> dict[str, Any]:
    """Clean an annotation coming from the UI (shape described in the module doc). Raises ``ValueError``."""
    verdict = raw.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    cameras: dict[str, Any] = {}
    for camera, item in (raw.get("cameras") or {}).items():
        if not set(camera) <= CAMERA_NAME_OK or not item:
            raise ValueError(f"bad camera {camera!r}")
        defects = _defects(item.get("defects"))
        cameras[camera] = {
            "shot": None if item.get("shot") is None else int(item["shot"]),
            "piece": None if item.get("piece") is None else _box(item["piece"]),
            "defects": defects,
        }
    return {
        "verdict": verdict,
        "source": raw.get("source") if raw.get("source") in ("operator", "model") else "operator",
        "note": str(raw.get("note") or "")[:500],
        "cameras": cameras,
    }


def defect_mask(shape: tuple[int, int], defects: list[dict[str, Any]]) -> np.ndarray:
    """uint8 mask (0 / 255) of the defect shapes on a frame of ``shape`` = (height, width)."""
    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    for d in defects:
        if d["type"] == "rect":
            x, y, w, h = d["box"]
            cv2.rectangle(mask, (int(x * width), int(y * height)), (int((x + w) * width), int((y + h) * height)), 255, -1)
        else:
            pts = np.array([[int(px * width), int(py * height)] for px, py in d["points"]], np.int32)
            thickness = max(1, int(d["width"] * width))
            if len(pts) == 1:
                cv2.circle(mask, tuple(pts[0]), max(1, thickness // 2), 255, -1)
            else:
                cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def square_cut(image: np.ndarray, box: list[float], size: int) -> np.ndarray:
    """The normalized ``box`` of ``image`` grown to a square around its centre (inside the frame), resized to ``size``."""
    height, width = image.shape[:2]
    x, y, w, h = box
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    side = max(w * width, h * height, 8.0)
    side = min(side, width, height)
    x0 = int(round(min(max(cx - side / 2, 0), width - side)))
    y0 = int(round(min(max(cy - side / 2, 0), height - side)))
    s = int(round(side))
    cut = image[y0:y0 + s, x0:x0 + s]
    interp = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(cut, (size, size), interpolation=interp)


class InspectionStore:
    """Frames and annotations of every cycle, on disk, under ``<runs_dir>/<run>/inspection``."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._lock = threading.Lock()

    # Paths ------------------------------------------------------------------------------------------------

    def _check_run(self, run: str) -> str:
        if not run or not set(run) <= set("0123456789-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            raise KeyError(f"bad run {run!r}")
        return run

    def cycle_dir(self, run: str, cycle: int) -> Path:
        return self.runs_dir / self._check_run(run) / "inspection" / f"c{int(cycle):04d}"

    def file(self, run: str, cycle: int, name: str) -> Path:
        """A stored file of a cycle (raises ``KeyError`` for anything outside it)."""
        if "/" in name or "\\" in name or name.startswith("."):
            raise KeyError(name)
        path = self.cycle_dir(run, cycle) / name
        if not path.is_file():
            raise KeyError(name)
        return path

    # Writing ----------------------------------------------------------------------------------------------

    def save_shot(
        self,
        run: str,
        cycle: int,
        shot_id: int,
        camera: str,
        phase: str,
        frame_rgb: np.ndarray,
        crop_rgb: np.ndarray,
        crop_box: tuple[float, float, float, float],
        usability: dict[str, Any],
        score: float | None,
        heat_rgb: np.ndarray | None = None,
        heat_info: dict[str, Any] | None = None,
        raw_map: np.ndarray | None = None,
        found: bool = True,
    ) -> dict[str, Any]:
        """Write one inspection frame of ``cycle`` and index it in ``shots.jsonl``; returns the index line."""
        folder = self.cycle_dir(run, cycle)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{camera}_{shot_id:05d}"
        cv2.imwrite(str(folder / f"{stem}_frame.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(folder / f"{stem}_crop.png"), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        line: dict[str, Any] = {
            "shot": shot_id, "camera": camera, "phase": phase, "t": round(time.time(), 3),
            "frame": f"{stem}_frame.jpg", "crop": f"{stem}_crop.png",
            "frame_size": [int(frame_rgb.shape[1]), int(frame_rgb.shape[0])],
            "crop_box": [round(float(v), 5) for v in crop_box], "found": bool(found),
            "usability": usability, "score": None if score is None else round(float(score), 4),
        }
        if heat_rgb is not None:
            cv2.imwrite(str(folder / f"{stem}_heat.png"), cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
            line["heat"] = f"{stem}_heat.png"
            line["heat_info"] = heat_info
        if raw_map is not None:
            np.save(folder / f"{stem}_raw.npy", np.asarray(raw_map, np.float16))
            line["raw"] = f"{stem}_raw.npy"
        with self._lock, (folder / "shots.jsonl").open("a") as handle:
            handle.write(json.dumps(line) + "\n")
        return line

    def save_annotation(self, run: str, cycle: int, annotation: dict[str, Any]) -> dict[str, Any]:
        clean = validate_annotation(annotation)
        clean["updated"] = round(time.time(), 3)
        folder = self.cycle_dir(run, cycle)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "annotation.json.tmp"
        tmp.write_text(json.dumps(clean, indent=1))
        tmp.replace(folder / "annotation.json")
        return clean

    def auto_annotation(self, run: str, cycle: int, verdict: str, source: str) -> dict[str, Any]:
        """Annotation built from the cycle's best frame per camera: piece = the auto brick box; for a defect verdict
        with a model map, its regions above the defect contour (converted from crop to frame coordinates)."""
        cameras: dict[str, Any] = {}
        for camera, shot in self.best_shots(run, cycle).items():
            defects = []
            if verdict == "bad" and shot.get("heat_info"):
                bx, by, bw, bh = shot["crop_box"]
                for rx, ry, rw, rh in shot["heat_info"].get("regions", [])[:5]:
                    defects.append({"type": "rect", "box": [bx + rx * bw, by + ry * bh, rw * bw, rh * bh]})
            cameras[camera] = {"shot": shot["shot"], "piece": shot["crop_box"], "defects": defects}
        return self.save_annotation(run, cycle, {"verdict": verdict, "source": source, "cameras": cameras})

    def reviews(self, run: str, cycle: int) -> dict[str, dict[str, Any]]:
        path = self.cycle_dir(run, cycle) / "review.json"
        return json.loads(path.read_text()).get("shots", {}) if path.exists() else {}

    def save_review(self, run: str, cycle: int, shot_id: int, review: dict[str, Any]) -> dict[str, Any]:
        """Store the operator's review of one frame (``validate_review``); the frame must exist."""
        clean = validate_review(review)
        if not any(s["shot"] == int(shot_id) for s in self.shots(run, cycle)):
            raise KeyError(f"no frame {shot_id} in run {run} cycle {cycle}")
        clean["updated"] = round(time.time(), 3)
        folder = self.cycle_dir(run, cycle)
        with self._lock:
            current = self.reviews(run, cycle)
            current[str(int(shot_id))] = clean
            tmp = folder / "review.json.tmp"
            tmp.write_text(json.dumps({"shots": current}, indent=1))
            tmp.replace(folder / "review.json")
        return clean

    def review_queue(self, run: str) -> dict[str, Any]:
        """Every stored frame of ``run`` in order (cycle, camera, shot), with its review or a proposal to confirm:
        the cycle annotation for its reference frame, else the automatic brick box (unusable frames: excluded)."""
        items: list[dict[str, Any]] = []
        for summary in self.cycles(run):
            cycle = summary["cycle"]
            annotation = self.annotation(run, cycle)
            verdict = None if annotation is None else annotation["verdict"]
            reviews = self.reviews(run, cycle)
            cams = (annotation or {}).get("cameras", {})
            for shot in sorted(self.shots(run, cycle), key=lambda s: (s["camera"], s["shot"])):
                done = reviews.get(str(shot["shot"]))
                ref = cams.get(shot["camera"]) or {}
                usable = bool((shot.get("usability") or {}).get("usable"))
                if done is not None:
                    state = {k: done[k] for k in ("piece", "defects", "excluded")}
                elif ref.get("shot") == shot["shot"]:
                    state = {"piece": ref.get("piece") or shot["crop_box"], "defects": ref.get("defects", []), "excluded": False}
                else:
                    state = {"piece": shot["crop_box"], "defects": [], "excluded": not usable}
                items.append({
                    "cycle": cycle, "shot": shot["shot"], "camera": shot["camera"], "verdict": verdict,
                    "frame": shot["frame"], "crop_box": shot["crop_box"], "frame_size": shot.get("frame_size"),
                    "usable": usable, "found": bool(shot.get("found")), "reviewed": done is not None, **state,
                })
        reviewed = sum(1 for i in items if i["reviewed"])
        return {"run": run, "items": items, "total": len(items), "reviewed": reviewed,
                "kept": sum(1 for i in items if i["reviewed"] and not i["excluded"])}

    # Reading ----------------------------------------------------------------------------------------------

    def shots(self, run: str, cycle: int) -> list[dict[str, Any]]:
        path = self.cycle_dir(run, cycle) / "shots.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def best_shots(self, run: str, cycle: int) -> dict[str, dict[str, Any]]:
        """Per camera, the frame to annotate: usable frames first, then the most anomalous (model) or sharpest."""
        best: dict[str, dict[str, Any]] = {}

        def rank(shot: dict[str, Any]) -> tuple[float, float, float]:
            u = shot.get("usability") or {}
            heat = (shot.get("heat_info") or {}).get("max_units") or 0.0
            return (1.0 if u.get("usable") else 0.0, 1.0 if shot.get("found") else 0.0, heat or u.get("sharpness", 0.0))

        for shot in self.shots(run, cycle):
            current = best.get(shot["camera"])
            if current is None or rank(shot) > rank(current):
                best[shot["camera"]] = shot
        return best

    def annotation(self, run: str, cycle: int) -> dict[str, Any] | None:
        path = self.cycle_dir(run, cycle) / "annotation.json"
        return json.loads(path.read_text()) if path.exists() else None

    def cycle(self, run: str, cycle: int) -> dict[str, Any]:
        shots = self.shots(run, cycle)
        if not shots and self.annotation(run, cycle) is None:
            raise KeyError(f"no inspection data for run {run} cycle {cycle}")
        return {"run": run, "cycle": int(cycle), "shots": shots, "best": self.best_shots(run, cycle),
                "annotation": self.annotation(run, cycle)}

    def cycles(self, run: str) -> list[dict[str, Any]]:
        root = self.runs_dir / self._check_run(run) / "inspection"
        out = []
        for folder in sorted(root.glob("c[0-9][0-9][0-9][0-9]")) if root.exists() else []:
            cycle = int(folder.name[1:])
            best = self.best_shots(run, cycle)
            annotation = self.annotation(run, cycle)
            out.append({
                "cycle": cycle,
                "verdict": None if annotation is None else annotation["verdict"],
                "source": None if annotation is None else annotation["source"],
                "defects": 0 if annotation is None else sum(len(c["defects"]) for c in annotation["cameras"].values()),
                "best": {cam: {"shot": s["shot"], "crop": s["crop"], "heat": s.get("heat"), "score": s.get("score"),
                               "max_units": (s.get("heat_info") or {}).get("max_units")} for cam, s in best.items()},
                "frames": sum(1 for _ in (folder / "shots.jsonl").open()) if (folder / "shots.jsonl").exists() else 0,
            })
        return out

    def runs(self) -> list[dict[str, Any]]:
        out = []
        for folder in sorted(self.runs_dir.glob("*/inspection"), reverse=True):
            run = folder.parent.name
            counts = {"good": 0, "bad": 0, "no_piece": 0, "unlabeled": 0}
            frames = 0
            size = 0
            reviewed = 0
            kept: dict[str, int] = {}
            for cycle_dir in folder.glob("c[0-9][0-9][0-9][0-9]"):
                review = cycle_dir / "review.json"
                if review.exists():
                    done = json.loads(review.read_text()).get("shots", {})
                    reviewed += len(done)
                    cameras = {str(s["shot"]): s["camera"] for s in self.shots(run, int(cycle_dir.name[1:]))}
                    for shot, item in done.items():
                        if not item.get("excluded") and shot in cameras:
                            kept[cameras[shot]] = kept.get(cameras[shot], 0) + 1
                annotation = cycle_dir / "annotation.json"
                verdict = json.loads(annotation.read_text())["verdict"] if annotation.exists() else "unlabeled"
                counts[verdict] = counts.get(verdict, 0) + 1
                for f in cycle_dir.iterdir():
                    size += f.stat().st_size
                    frames += f.name.endswith("_frame.jpg")
            out.append({"run": run, "cycles": sum(counts.values()), **counts, "frames": frames, "reviewed": reviewed, "kept_by_camera": kept, "bytes": size})
        return out

    # Export -----------------------------------------------------------------------------------------------

    def export_zip(self, runs: list[str], dest: Path, size: int = 256, reviewed_only: bool = False) -> dict[str, int]:
        """Write the Anomalib-ready dataset of ``runs`` (layout in the module doc) to ``dest``; returns counts."""
        if reviewed_only:
            return self._export_reviewed(runs, dest, size)
        counts = {"train_good": 0, "test_good": 0, "test_defect": 0, "test_defect_nomask": 0, "skipped": 0}
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as archive:
            rows: list[str] = []
            for run in runs:
                for item in self.cycles(run):
                    cycle = item["cycle"]
                    annotation = self.annotation(run, cycle)
                    shots = self.shots(run, cycle)
                    if annotation is None or annotation["verdict"] == "no_piece":
                        counts["skipped"] += 1
                        continue
                    rows.append(json.dumps({"run": run, "cycle": cycle, **annotation}))
                    by_id = {s["shot"]: s for s in shots}
                    for camera, cam_ann in annotation["cameras"].items():
                        ref = by_id.get(cam_ann.get("shot")) or self.best_shots(run, cycle).get(camera)
                        if ref is None:
                            continue
                        frame = cv2.imread(str(self.cycle_dir(run, cycle) / ref["frame"]))
                        if frame is None:
                            continue
                        piece = cam_ann.get("piece") or ref["crop_box"]
                        name = f"c{cycle:04d}_s{ref['shot']:05d}"
                        base = f"{run}/{camera}"
                        archive.writestr(f"{base}/full/{name}.jpg", (self.cycle_dir(run, cycle) / ref["frame"]).read_bytes())
                        cut = square_cut(frame, piece, size)
                        if annotation["verdict"] == "good":
                            archive.writestr(f"{base}/test/good/{name}.png", _png(cut))
                            counts["test_good"] += 1
                            for shot in shots:  # every other usable frame of a good brick is training material
                                if shot["camera"] != camera or shot["shot"] == ref["shot"] or not (shot.get("usability") or {}).get("usable"):
                                    continue
                                other = cv2.imread(str(self.cycle_dir(run, cycle) / shot["frame"]))
                                if other is not None:
                                    archive.writestr(f"{base}/train/good/c{cycle:04d}_s{shot['shot']:05d}.png",
                                                     _png(square_cut(other, shot["crop_box"], size)))
                                    counts["train_good"] += 1
                        else:
                            mask = square_cut(defect_mask(frame.shape[:2], cam_ann["defects"]), piece, size)
                            if not mask.any():
                                # Defect brick, but nothing drawn on this camera (defect not visible from it, or not
                                # marked): an empty mask would tell Anomalib "defective, nowhere". Kept apart.
                                archive.writestr(f"{base}/test/defect_nomask/{name}.png", _png(cut))
                                counts["test_defect_nomask"] += 1
                                continue
                            archive.writestr(f"{base}/test/defect/{name}.png", _png(cut))
                            archive.writestr(f"{base}/ground_truth/defect/{name}.png", _png(mask))
                            counts["test_defect"] += 1
            archive.writestr("annotations.jsonl", "\n".join(rows) + ("\n" if rows else ""))
            archive.writestr("README.txt", README.format(runs=", ".join(runs), size=size, **counts))
        return counts


    def _export_reviewed(self, runs: list[str], dest: Path, size: int) -> dict[str, int]:
        counts = {"train_good": 0, "test_good": 0, "test_defect": 0, "test_defect_nomask": 0, "skipped": 0}
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as archive:
            rows: list[str] = []
            for run in runs:
                good_seen: dict[str, int] = {}
                for item in self.review_queue(run)["items"]:
                    if not item["reviewed"] or item["excluded"] or item["verdict"] not in ("good", "bad") or not item["piece"]:
                        counts["skipped"] += 1
                        continue
                    folder = self.cycle_dir(run, item["cycle"])
                    frame = cv2.imread(str(folder / item["frame"]))
                    if frame is None:
                        counts["skipped"] += 1
                        continue
                    camera, name = item["camera"], f"c{item['cycle']:04d}_s{item['shot']:05d}"
                    base = f"{run}/{camera}"
                    cut = square_cut(frame, item["piece"], size)
                    archive.writestr(f"{base}/full/{name}.jpg", (folder / item["frame"]).read_bytes())
                    rows.append(json.dumps({"run": run, **{k: item[k] for k in ("cycle", "shot", "camera", "verdict", "piece", "defects")}}))
                    if item["verdict"] == "good":
                        good_seen[camera] = good_seen.get(camera, 0) + 1
                        split = "test" if good_seen[camera] % 5 == 0 else "train"  # 1 in 5 good frames held out
                        archive.writestr(f"{base}/{split}/good/{name}.png", _png(cut))
                        counts[f"{split}_good"] += 1
                        continue
                    mask = square_cut(defect_mask(frame.shape[:2], item["defects"]), item["piece"], size)
                    if not mask.any():
                        archive.writestr(f"{base}/test/defect_nomask/{name}.png", _png(cut))
                        counts["test_defect_nomask"] += 1
                        continue
                    archive.writestr(f"{base}/test/defect/{name}.png", _png(cut))
                    archive.writestr(f"{base}/ground_truth/defect/{name}.png", _png(mask))
                    counts["test_defect"] += 1
            archive.writestr("annotations.jsonl", "\n".join(rows) + ("\n" if rows else ""))
            archive.writestr("README.txt", README_REVIEWED.format(runs=", ".join(runs), size=size, **counts))
        return counts


def _png(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return io.BytesIO(data.tobytes()).getvalue()


README = """SortOps inspection dataset (Anomalib Folder / MVTec layout)
Runs: {runs}
Crops: the operator's piece box (else the automatic brick box), grown to a square, {size} x {size} px.

<run>/<camera>/train/good/          every usable frame of the cycles labeled good   ({train_good})
<run>/<camera>/test/good/           the reference frame of each good cycle          ({test_good})
<run>/<camera>/test/defect/         the reference frame of each defect cycle        ({test_defect})
<run>/<camera>/ground_truth/defect/ defect masks (255 = defect) drawn by the operator, same names as test/defect
<run>/<camera>/test/defect_nomask/  defect bricks with nothing drawn on this camera (no mask: not for pixel metrics) ({test_defect_nomask})
<run>/<camera>/full/                the whole camera frames of the reference shots
annotations.jsonl                   one line per cycle: verdict, source, piece box, defect shapes (frame coords)
Skipped (no piece / not annotated): {skipped}

Anomalib: Folder(name=..., root="<run>/<camera>", normal_dir="train/good", abnormal_dir="test/defect",
                 normal_test_dir="test/good", mask_dir="ground_truth/defect")
"""


README_REVIEWED = """SortOps inspection dataset, REVIEWED frames only (Anomalib Folder / MVTec layout)
Runs: {runs}
Every image was checked by the operator in Data -> Review: cut to the piece box drawn on it, square, {size} x {size} px.

<run>/<camera>/train/good/          reviewed frames of good bricks (4 in 5)              ({train_good})
<run>/<camera>/test/good/           reviewed frames of good bricks held out (1 in 5)     ({test_good})
<run>/<camera>/test/defect/         reviewed frames of defect bricks with defects drawn  ({test_defect})
<run>/<camera>/ground_truth/defect/ their masks (255 = defect), same names
<run>/<camera>/test/defect_nomask/  defect bricks, nothing drawn on that frame           ({test_defect_nomask})
<run>/<camera>/full/                the whole camera frames
annotations.jsonl                   one line per exported frame
Skipped (not reviewed, excluded, no verdict): {skipped}

Anomalib: Folder(name=..., root="<run>/<camera>", normal_dir="train/good", abnormal_dir="test/defect",
                 normal_test_dir="test/good", mask_dir="ground_truth/defect")
"""


def remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except IsADirectoryError:
        shutil.rmtree(path, ignore_errors=True)
