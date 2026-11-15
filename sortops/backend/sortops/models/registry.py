"""Which model fills which role. Reads Physical AI Studio's catalogue (``/api/projects/<id>/models`` + the model
folders), ranks each role's candidates newest training first then most steps, and imports one: its OpenVINO
export is copied under ``<data>/models/<role>/<model id>/`` with a ``provenance.json`` (name, dataset, date,
steps, losses) so the UI always shows exactly what runs. Anomalib exports are found by scanning folders for
OpenVINO ``.xml`` files.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sortops.config import ANOMALY_ROLES, POLICY_ROLES, Settings

_STEP = re.compile(r"-step(\d+)$")


@dataclass
class Candidate:
    id: str
    name: str
    role: str
    dataset: str
    created_at: str
    steps: int | None
    train_loss: float | None
    val_loss: float | None
    source: str
    """Folder of the OpenVINO export to import."""
    checkpoint: bool
    """An intermediate ``-stepN`` checkpoint rather than the end of the training run."""
    uses_depth: bool = False
    imported: bool = False
    active: bool = False
    recommended: bool = False
    """What ``Import latest trained models`` picks for this role."""


def _training_metrics(model_dir: Path) -> tuple[int | None, float | None, float | None]:
    """Last step and last train/val loss of a Studio model, from its Lightning ``version_*/metrics.csv``."""
    files = sorted(model_dir.glob("version_*/metrics.csv"))
    if not files:
        return None, None, None
    step: int | None = None
    train: float | None = None
    val: float | None = None
    with files[-1].open() as handle:
        for row in csv.DictReader(handle):
            if row.get("step"):
                step = max(step or 0, int(float(row["step"])))
            if row.get("train/loss"):
                train = float(row["train/loss"])
            if row.get("val/loss"):
                val = float(row["val/loss"])
    return step, train, val


class ModelRegistry:
    def __init__(self, settings: Settings, data_dir: Path, *, simulated: bool) -> None:
        self._settings = settings
        self._root = data_dir / "models"
        self._root.mkdir(parents=True, exist_ok=True)
        self._active_path = self._root / "active.json"
        self._simulated = simulated
        self._lock = threading.Lock()
        self._migrate_single_anomaly()

    def _migrate_single_anomaly(self) -> None:
        """Before per-camera models there was one ``anomaly`` role (trained on wrist photos): it becomes anomaly_wrist."""
        if not self._active_path.exists():
            return
        chosen: dict[str, str] = json.loads(self._active_path.read_text())
        model_id = chosen.pop("anomaly", None)
        if model_id is None:
            return
        old, new = self._root / "anomaly", self._root / "anomaly_wrist"
        if old.exists() and not new.exists():
            shutil.move(str(old), str(new))
        if (new / model_id / "provenance.json").exists():
            provenance = json.loads((new / model_id / "provenance.json").read_text())
            provenance["role"] = "anomaly_wrist"
            (new / model_id / "provenance.json").write_text(json.dumps(provenance, indent=2))
        chosen.setdefault("anomaly_wrist", model_id)
        self._active_path.write_text(json.dumps(chosen, indent=2))

    # Active selection ------------------------------------------------------------------------------------

    def active(self) -> dict[str, dict[str, Any] | None]:
        with self._lock:
            chosen: dict[str, str] = json.loads(self._active_path.read_text()) if self._active_path.exists() else {}
        out: dict[str, dict[str, Any] | None] = {}
        for role in (*POLICY_ROLES, *ANOMALY_ROLES):
            model_id = chosen.get(role)
            path = self._root / role / model_id / "provenance.json" if model_id else None
            out[role] = json.loads(path.read_text()) if path is not None and path.exists() else None
        return out

    def export_dir(self, role: str) -> Path | None:
        provenance = self.active().get(role)
        if provenance is None or provenance.get("simulated"):
            return None
        return self._root / role / provenance["id"] / "export"

    def calibration(self, role: str) -> dict[str, Any] | None:
        """SortOps' score scale for the active anomaly model (``calibration.json`` next to its export)."""
        provenance = self.active().get(role)
        if provenance is None:
            return None
        path = self._root / role / provenance["id"] / "calibration.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save_calibration(self, values: dict[str, Any] | None, role: str) -> None:
        provenance = self.active().get(role)
        if provenance is None:
            raise KeyError(f"No active {role} model")
        path = self._root / role / provenance["id"] / "calibration.json"
        if values is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(json.dumps(values, indent=2))

    def _set_active(self, role: str, model_id: str | None) -> None:
        with self._lock:
            chosen: dict[str, str] = json.loads(self._active_path.read_text()) if self._active_path.exists() else {}
            if model_id is None:
                chosen.pop(role, None)
            else:
                chosen[role] = model_id
            self._active_path.write_text(json.dumps(chosen, indent=2))

    def deactivate(self, role: str) -> None:
        self._set_active(role, None)

    # Catalogue -------------------------------------------------------------------------------------------

    def catalog(self) -> dict[str, list[Candidate]]:
        hardware = self._settings.section("hardware")
        roles = self._settings.section("roles")
        patterns = {role: [re.compile(p) for p in items] for role, items in self._settings.section("role_name_patterns").items()}
        candidates = self._simulated_catalog(roles) if self._simulated else self._studio_catalog(hardware, roles, patterns)
        for role in ANOMALY_ROLES:
            # Any Anomalib export may serve any camera: pick the one trained on that camera's photos.
            if self._simulated:
                candidates[role] = [self._sim(role, f"sim-patchcore-{role.removeprefix('anomaly_')}", "anomaly-shots", 0, "2026-11-14T09:00:00Z")]
            else:
                candidates[role] = self._anomaly_catalog(hardware["anomaly_model_roots"], role)
        active = self.active()
        for role, items in candidates.items():
            for item in items:
                item.imported = (self._root / role / item.id / "provenance.json").exists()
                current = active.get(role)
                item.active = current is not None and current.get("id") == item.id
            items.sort(key=lambda c: (c.created_at, c.steps or 0), reverse=True)
            best = pick_best(items)
            if best is not None:
                best.recommended = True
        return candidates

    def _studio_catalog(
        self, hardware: dict[str, Any], roles: dict[str, list[str]], patterns: dict[str, list[re.Pattern[str]]]
    ) -> dict[str, list[Candidate]]:
        base = f"{hardware['studio_url'].rstrip('/')}/api/projects/{hardware['studio_project']}"
        with urllib.request.urlopen(base, timeout=5) as response:  # noqa: S310 - local Studio API
            project = json.loads(response.read())
        with urllib.request.urlopen(f"{base}/models", timeout=5) as response:  # noqa: S310
            models = json.loads(response.read())
        datasets = {item["id"]: item["name"] for item in project.get("datasets", [])}
        out: dict[str, list[Candidate]] = {role: [] for role in POLICY_ROLES}
        for model in models:
            dataset = datasets.get(model.get("dataset_id", ""), "")
            if "openvino" not in model.get("available_backends", []):
                continue
            model_dir = Path(model["path"])
            steps, train, val = _training_metrics(model_dir)
            match = _STEP.search(model["name"])
            if steps is None and match:
                steps = int(match.group(1))
            source = model_dir / "exports" / "openvino"
            uses_depth = False
            manifest = source / "manifest.json"
            if manifest.exists():
                uses_depth = "_depth" in manifest.read_text()
            for role, names in roles.items():
                if dataset in names or any(p.search(model["name"]) for p in patterns.get(role, [])):
                    out[role].append(
                        Candidate(
                            id=model["id"],
                            name=model["name"],
                            role=role,
                            dataset=dataset,
                            created_at=model.get("created_at", ""),
                            steps=steps,
                            train_loss=train,
                            val_loss=val,
                            source=str(source),
                            checkpoint=bool(match),
                            uses_depth=uses_depth,
                        )
                    )
        return out

    def _anomaly_catalog(self, roots: list[str], role: str) -> list[Candidate]:
        found: list[Candidate] = []
        for root in roots:
            base = Path(root)
            if not base.exists():
                continue
            for xml in base.rglob("*.xml"):
                if not xml.with_suffix(".bin").exists() or (xml.parent / "manifest.json").exists():
                    continue
                stat = xml.stat()
                model_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(xml.relative_to(base).parent)) or xml.stem
                found.append(
                    Candidate(
                        id=model_id,
                        name=f"{xml.parent.name}/{xml.name}",
                        role=role,
                        dataset=str(xml.parent),
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                        steps=None,
                        train_loss=None,
                        val_loss=None,
                        source=str(xml.parent),
                        checkpoint=False,
                    )
                )
        return found

    @staticmethod
    def _sim(role: str, name: str, dataset: str, steps: int, created: str, checkpoint: bool = False) -> Candidate:
        return Candidate(
            id=f"sim-{name}",
            name=name,
            role=role,
            dataset=dataset,
            created_at=created,
            steps=steps or None,
            train_loss=0.19 if steps else None,
            val_loss=0.38 if steps else None,
            source="",
            checkpoint=checkpoint,
        )

    def _simulated_catalog(self, roles: dict[str, list[str]]) -> dict[str, list[Candidate]]:
        out: dict[str, list[Candidate]] = {role: [] for role in POLICY_ROLES}
        for role, names in roles.items():
            if not names:
                continue
            dataset = names[0]
            stem = f"act-{dataset}"
            out[role].append(self._sim(role, stem, dataset, 2699, "2026-11-14T10:05:57Z"))
            for step in (2500, 1500):
                out[role].append(self._sim(role, f"{stem}-step{step}", dataset, step, "2026-11-14T10:05:57Z", True))
        return out

    # Import ----------------------------------------------------------------------------------------------

    def import_model(self, role: str, model_id: str) -> dict[str, Any]:
        candidates = {c.id: c for c in self.catalog().get(role, [])}
        if model_id not in candidates:
            raise KeyError(f"No {role} candidate {model_id}")
        candidate = candidates[model_id]
        target = self._root / role / candidate.id
        export = target / "export"
        if not self._simulated:
            source = Path(candidate.source)
            if not source.exists():
                raise FileNotFoundError(f"Export folder {source} is missing")
            if export.exists():
                shutil.rmtree(export)
            shutil.copytree(source, export)
        target.mkdir(parents=True, exist_ok=True)
        provenance = {
            **asdict(candidate),
            "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "simulated": self._simulated,
        }
        for key in ("imported", "active", "recommended"):
            provenance.pop(key, None)
        (target / "provenance.json").write_text(json.dumps(provenance, indent=2))
        self._set_active(role, candidate.id)
        return provenance

    def best(self, role: str) -> Candidate | None:
        return pick_best(self.catalog().get(role, []))


def pick_best(candidates: list[Candidate]) -> Candidate | None:
    """Latest, most-trained model: the most recently trained family (name without ``-stepN``), then its most steps.

    A full run beats a checkpoint of the same step count; a family exported only as checkpoints still counts.
    """
    if not candidates:
        return None
    families: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        families.setdefault(_STEP.sub("", candidate.name), []).append(candidate)
    newest = max(families.values(), key=lambda items: max(c.created_at for c in items))
    return max(newest, key=lambda c: (c.steps or 0, not c.checkpoint, c.created_at))
