"""Recorded episode replay: the "present" step plays one Studio (LeRobot v3) episode's actions as recorded.

No model is involved: the joint targets come from the dataset's ``action`` column, in frame order, at its fps.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sortops.config import JOINTS

_ACTION_NAMES = [f"{joint}.pos" for joint in JOINTS]


@dataclass(frozen=True)
class Episode:
    actions: np.ndarray  # (frames, 6) joint targets in the follower's normalized units
    fps: float

    @property
    def duration_s(self) -> float:
        return len(self.actions) / self.fps


def resolve_dataset(name: str, datasets_dir: Path, studio_url: str, project: str) -> Path:
    """The folder of dataset ``name``: a folder of ``datasets_dir`` (Studio names them by id), an absolute path, or a
    dataset name looked up in the Studio project.

    Raises:
        FileNotFoundError: No such dataset.
    """
    direct = Path(name).expanduser()
    for candidate in (datasets_dir / name, direct):
        if candidate.is_absolute() and (candidate / "meta" / "info.json").exists():
            return candidate
    url = f"{studio_url.rstrip('/')}/api/projects/{project}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - local Studio API
            datasets = json.loads(response.read()).get("datasets", [])
    except OSError as error:
        raise FileNotFoundError(f"Dataset {name!r}: Studio did not answer ({error})") from error
    for item in datasets:
        if item.get("name") == name:
            folder = datasets_dir / str(item["id"])
            if (folder / "meta" / "info.json").exists():
                return folder
            raise FileNotFoundError(f"Dataset {name!r} has no files in {folder}")
    raise FileNotFoundError(f"No dataset named {name!r} in the Studio project")


def load_episode(folder: Path, episode: int) -> Episode:
    """Episode ``episode`` of the LeRobot dataset in ``folder``.

    Raises:
        ValueError: The dataset does not record the follower's 6 joint actions, or has no such episode.
    """
    import pyarrow.parquet as pq  # Studio's venv has it; only the replay step needs it

    info = json.loads((folder / "meta" / "info.json").read_text())
    names = (info.get("features", {}).get("action") or {}).get("names")
    if names != _ACTION_NAMES:
        raise ValueError(f"{folder.name}: actions are {names}, expected {_ACTION_NAMES}")
    rows: list[tuple[int, list[float]]] = []
    for path in sorted((folder / "data").glob("**/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action"]).to_pydict()
        rows.extend(
            (int(frame), action)
            for index, frame, action in zip(table["episode_index"], table["frame_index"], table["action"], strict=True)
            if int(index) == episode
        )
    if not rows:
        raise ValueError(f"{folder.name}: no episode {episode}")
    rows.sort(key=lambda row: row[0])
    return Episode(np.asarray([action for _, action in rows], np.float32), float(info["fps"]))


def drop_pauses(actions: np.ndarray, epsilon: float) -> np.ndarray:
    """The same path without its pauses: a frame is kept once some joint moved ``epsilon`` units from the last kept one.

    The first and last frames always stay, so the gesture starts and ends where it was recorded.
    """
    if epsilon <= 0 or len(actions) < 3:
        return actions
    kept = [0]
    for index in range(1, len(actions) - 1):
        if float(np.max(np.abs(actions[index] - actions[kept[-1]]))) >= epsilon:
            kept.append(index)
    kept.append(len(actions) - 1)
    return actions[kept]
