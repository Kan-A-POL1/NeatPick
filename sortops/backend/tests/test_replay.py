"""Present step: a recorded Studio episode replayed as is (no model) between the pick and the inspection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from sortops.cell import Cell, _PickWatch
from sortops.hw.robot import RobotState
from sortops.config import DEFAULTS, JOINTS, Settings
from sortops.pipeline.replay import load_episode, resolve_dataset
from sortops.safety import Mode

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

_POSE = [0.0, -30.0, 40.0, -10.0, -55.0, 5.0]  # the simulated inspection pose (sortops.hw.sim)


def _write_dataset(root: Path, episodes: dict[int, np.ndarray], fps: int = 30, names: list[str] | None = None) -> Path:
    folder = root / "ds-id"
    (folder / "meta").mkdir(parents=True)
    (folder / "data" / "chunk-000").mkdir(parents=True)
    names = names or [f"{joint}.pos" for joint in JOINTS]
    (folder / "meta" / "info.json").write_text(json.dumps({"fps": fps, "features": {"action": {"names": names}}}))
    rows: dict[str, list[object]] = {"episode_index": [], "frame_index": [], "action": []}
    for index, actions in episodes.items():
        for frame in reversed(range(len(actions))):  # stored out of order: the loader sorts by frame
            rows["episode_index"].append(index)
            rows["frame_index"].append(frame)
            rows["action"].append(actions[frame].tolist())
    pq.write_table(pa.table(rows), folder / "data" / "chunk-000" / "file-000.parquet")
    return folder


def _turn(seconds: float = 3.0, fps: int = 30) -> np.ndarray:
    """At the inspection pose, roll the wrist from one face to the other, then hold."""
    frames = []
    for k in range(int(seconds * fps)):
        pose = list(_POSE)
        pose[4] = -55.0 + 112.0 * k / (seconds * fps - 1)
        frames.append(pose)
    frames += [frames[-1]] * int(1.5 * fps)
    return np.asarray(frames, np.float32)


def test_load_episode_keeps_one_episode_in_frame_order(tmp_path: Path) -> None:
    first = np.arange(18, dtype=np.float32).reshape(3, 6)
    folder = _write_dataset(tmp_path, {0: first, 1: first + 100})
    episode = load_episode(folder, 0)
    assert episode.fps == 30
    np.testing.assert_allclose(episode.actions, first)
    np.testing.assert_allclose(load_episode(folder, 1).actions, first + 100)
    with pytest.raises(ValueError, match="no episode 2"):
        load_episode(folder, 2)


def test_load_episode_refuses_actions_of_other_joints(tmp_path: Path) -> None:
    folder = _write_dataset(tmp_path, {0: np.zeros((2, 6), np.float32)}, names=["a", "b", "c", "d", "e", "f"])
    with pytest.raises(ValueError, match="expected"):
        load_episode(folder, 0)


def test_resolve_dataset_by_folder_id_or_reports_a_missing_name(tmp_path: Path) -> None:
    folder = _write_dataset(tmp_path, {0: np.zeros((2, 6), np.float32)})
    assert resolve_dataset("ds-id", tmp_path, "http://127.0.0.1:9", "p") == folder
    with pytest.raises(FileNotFoundError, match="Studio did not answer"):
        resolve_dataset("present-cube-v1", tmp_path, "http://127.0.0.1:9", "p")


def test_present_rejects_an_empty_dataset_name(tmp_path: Path) -> None:
    settings = Settings(tmp_path)
    with pytest.raises(ValueError, match="present.dataset"):
        settings.update({"present": {"dataset": "  "}})


def _run(cell: Cell, seconds: float = 80) -> None:
    deadline = time.monotonic() + seconds
    while cell.cycle_running() and time.monotonic() < deadline:
        time.sleep(0.2)


def test_cycle_picks_then_replays_the_episode_then_places(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "datasets", {0: _turn()})
    settings = Settings(
        tmp_path,
        {
            "safety": {"watchdog_enabled": False},
            "run": {"max_cycles": 1},
            "hardware": {"studio_datasets_dir": str(tmp_path / "datasets")},
            "present": {"enabled": True, "dataset": "ds-id", "pause_epsilon": 0.0},
        },
    )
    cell = Cell(settings, tmp_path)
    sent: list[np.ndarray] = []
    original = cell.robot.send
    cell.robot.send = lambda goal: (sent.append(np.asarray(goal).copy()), original(goal))[1]  # type: ignore[method-assign]
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=False)
        _run(cell)
        (cycle,) = cell.cycles.cycles()
        assert cycle.outcome in {"good", "bad"}, (cycle.outcome, cycle.error)
        assert {"pick", "present", "inspect"} <= set(cycle.stages) and "pick_present" not in cycle.stages
        assert cycle.stages["present"] >= 4.0  # the 4.5 s episode is played in full, at its own pace
        assert cycle.frames_usable >= DEFAULTS["inspection"]["min_usable_frames"]
        # The wrist turn of the recording reached the arm (both faces), within the safety filter's step limit.
        rolls = [float(goal[4]) for goal in sent]
        assert min(rolls) <= -50 and max(rolls) >= 50
        steps = np.abs(np.diff(np.asarray(sent), axis=0))
        assert float(steps.max()) <= DEFAULTS["safety"]["max_step"] + 1e-3
    finally:
        cell.shutdown()


def test_estop_during_the_replay_holds_and_sends_nothing_more(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "datasets", {0: _turn(seconds=20.0)})
    settings = Settings(
        tmp_path,
        {
            "safety": {"watchdog_enabled": False},
            "run": {"max_cycles": 1},
            "hardware": {"studio_datasets_dir": str(tmp_path / "datasets")},
            "present": {"enabled": True, "dataset": "ds-id"},
        },
    )
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=False)
        deadline = time.monotonic() + 40
        while cell.phase != "present" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cell.phase == "present"
        time.sleep(1.0)
        cell.estop("test")
        _run(cell, 5)
        assert cell.cycles.cycles()[-1].outcome == "aborted"
        held = cell.robot.read().joints.copy()
        time.sleep(0.4)
        np.testing.assert_allclose(cell.robot.read().joints, held, atol=0.5)
    finally:
        cell.shutdown()


def test_missing_dataset_fails_the_cycle_on_the_real_cell_but_is_skipped_in_simulation(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {
            "safety": {"watchdog_enabled": False},
            "hardware": {"studio_datasets_dir": str(tmp_path / "none"), "studio_url": "http://127.0.0.1:9"},
            "present": {"enabled": True},
        },
    )
    cell = Cell(settings, tmp_path)
    assert cell._present_episode() is None  # noqa: SLF001 - simulation: skipped
    cell.world = None
    with pytest.raises(FileNotFoundError):
        cell._present_episode()  # noqa: SLF001 - real cell: the cycle fails and the arm holds


def test_drop_pauses_keeps_the_path_ends_and_motion_only() -> None:
    from sortops.pipeline.replay import drop_pauses

    still = np.zeros((10, 6), np.float32)
    moving = np.linspace(0, 10, 11, dtype=np.float32)[:, None].repeat(6, axis=1)
    actions = np.concatenate([still, moving, np.full((10, 6), 10.0, np.float32)])
    kept = drop_pauses(actions, 0.5)
    np.testing.assert_allclose(kept[0], actions[0])
    np.testing.assert_allclose(kept[-1], actions[-1])
    assert len(kept) == 12  # the first still frame, the 10 moving steps, the last frame
    assert drop_pauses(actions, 0).shape == actions.shape


def test_pick_ends_once_the_brick_is_held_not_at_the_timeout(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "datasets", {0: _turn(seconds=1.0)})
    settings = Settings(
        tmp_path,
        {
            "safety": {"watchdog_enabled": False},
            "run": {"max_cycles": 1},
            "hardware": {"studio_datasets_dir": str(tmp_path / "datasets")},
            "present": {"enabled": True, "dataset": "ds-id"},
        },
    )
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=False)
        _run(cell)
        (cycle,) = cell.cycles.cycles()
        # The simulated pick opens to 60 then closes on the brick at ~4 s: well before the 30 s timeout.
        assert cycle.stages["pick"] < 8.0, cycle.stages
    finally:
        cell.shutdown()


def test_release_frees_the_arm_until_taken_back(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}})
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        deadline = time.monotonic() + 5
        while not cell.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        cell.release_robot()
        assert not cell.connected and cell.released
        time.sleep(4.0)  # longer than the reconnection retry period
        assert not cell.connected, "a released arm must not be reconnected"
        assert cell.snapshot()["released"] is True
        with pytest.raises(RuntimeError, match="not connected"):
            cell.start_cycles(loop=False)
        cell.take_robot()
        deadline = time.monotonic() + 5
        while not cell.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cell.connected and not cell.released
    finally:
        cell.shutdown()


def test_release_is_refused_while_a_cycle_runs(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}})
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=True)
        with pytest.raises(RuntimeError, match="Stop the cycle"):
            cell.release_robot()
        assert cell.connected
    finally:
        cell.stop_cycle(immediate=True)
        cell.shutdown()


def _pick_watch(tmp_path: Path) -> _PickWatch:
    cell = Cell(Settings(tmp_path, {"safety": {"watchdog_enabled": False}}), tmp_path)
    return _PickWatch(cell, lambda state, elapsed: None)


def _grip(value: float) -> RobotState:
    return RobotState(joints=np.asarray([0, 0, 0, 0, 0, value], np.float32), ee_xyz=np.zeros(3, np.float32))


def test_pick_is_held_when_the_brick_stops_the_jaws_above_the_closed_value(tmp_path: Path, monkeypatch) -> None:
    watch = _pick_watch(tmp_path)
    clock = {"t": 100.0}
    monkeypatch.setattr("sortops.cell.time.monotonic", lambda: clock["t"])
    # Opens to 40, closes onto the brick which stops the jaws at 20 (never below pick_closed_below = 8).
    for value in (2, 20, 40, 35, 28, 22, 20.5, 20, 20.4):
        assert not watch(_grip(value), 0.0)
        clock["t"] += 0.1
    clock["t"] += 0.8
    assert watch(_grip(20.2), 0.0), "jaws stopped on the brick for pick_closed_s: held"


def test_pick_is_not_held_while_the_jaws_still_move_or_never_opened(tmp_path: Path, monkeypatch) -> None:
    watch = _pick_watch(tmp_path)
    clock = {"t": 100.0}
    monkeypatch.setattr("sortops.cell.time.monotonic", lambda: clock["t"])
    for value in (2, 3, 2, 4, 3):
        clock["t"] += 1.0
        assert not watch(_grip(value), 0.0), "never opened: not a grasp"
    assert "never opened" in watch.summary()
    for value in (40, 29, 26, 23, 20, 17, 14, 11):
        clock["t"] += 0.5
        assert not watch(_grip(value), 0.0), "still closing: wait for the jaws to stop"
    assert "opened to 40" in watch.summary()
