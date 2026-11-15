"""Behaviour of the decision policy, the safety layer, frame usability, model ranking and one simulated cycle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from sortops.cell import Cell
from sortops.config import DEFAULTS, Settings
from sortops.hw.robot import SimRobot
from sortops.metrics import CycleBook
from sortops.models.registry import ModelRegistry, _training_metrics
from sortops.pipeline.brick_crop import crop_brick
from sortops.pipeline.inspection import Verdict, aggregate, assess, decide
from sortops.safety import Mode, SafetyGuard

DECISION = DEFAULTS["decision"]
INSPECTION = DEFAULTS["inspection"]


# Decision policy ----------------------------------------------------------------------------------------


def test_clear_scores_decide_good_and_bad() -> None:
    assert decide([0.1, 0.2, 0.15], 3, 0, DECISION, "p90", 2).verdict == Verdict.GOOD
    assert decide([0.1, 0.9, 0.85], 3, 0, DECISION, "p90", 2).verdict == Verdict.BAD


def test_gray_zone_is_uncertain() -> None:
    decision = decide([0.5, 0.5], 2, 0, DECISION, "mean", 2)
    assert decision.verdict == Verdict.UNCERTAIN
    assert "gray zone" in decision.reason


def test_too_few_usable_frames_is_uncertain_even_with_a_clear_score() -> None:
    assert decide([0.05], 6, 0, DECISION, "p90", 2).verdict == Verdict.UNCERTAIN


def test_arm_presented_without_a_recognized_brick_goes_to_the_defect_bin() -> None:
    # No brick in any frame, or in at least no_brick_defect_ratio (0.5) of them: defect bin, flagged for the streak.
    for missing in (5, 3):
        decision = decide([0.1, 0.1], 5, missing, DECISION, "p90", 2)
        assert decision.verdict == Verdict.BAD and decision.no_brick, decision
    # A brick in most frames: scored as usual.
    assert decide([0.1, 0.1], 5, 2, DECISION, "p90", 2).verdict == Verdict.GOOD
    # Bricks seen but nothing usable: uncertain (recapture), not a no-brick defect.
    assert decide([], 5, 0, DECISION, "p90", 2).verdict == Verdict.UNCERTAIN
    # The pose was never reached: nothing was presented.
    assert decide([], 0, 0, DECISION, "p90", 2).verdict == Verdict.UNCERTAIN


def test_one_defective_face_dominates_with_max_but_not_mean() -> None:
    scores = [0.1, 0.1, 0.1, 0.9]
    assert aggregate(scores, "max") == pytest.approx(0.9)
    assert aggregate(scores, "mean") == pytest.approx(0.3)
    with pytest.raises(ValueError):
        aggregate(scores, "median")


# Safety --------------------------------------------------------------------------------------------------


def _guard() -> SafetyGuard:
    return SafetyGuard(lambda: dict(DEFAULTS["safety"]))


def test_filter_clamps_limits_and_rate() -> None:
    guard = _guard()
    current = np.zeros(6, np.float32)
    target = np.asarray([50, -50, 3, 200, -200, -10], np.float32)
    out = guard.filter(target, current)
    step = DEFAULTS["safety"]["max_step"]
    assert np.all(np.abs(out - current) <= step + 1e-6)
    assert out[2] == pytest.approx(3.0)
    assert out[5] >= 0.0
    assert guard.clamped_ticks == 1


def test_motion_needs_live_mode_and_no_estop() -> None:
    guard = _guard()
    assert not guard.may_move()
    guard.set_mode(Mode.LIVE)
    assert guard.may_move()
    reasons: list[str] = []
    guard.on_estop(reasons.append)
    guard.trigger("test")
    assert not guard.may_move()
    assert reasons == ["test"]
    with pytest.raises(RuntimeError):
        guard.set_mode(Mode.LIVE)
    guard.reset()
    assert guard.mode == Mode.SHADOW and not guard.estopped


def test_watchdog_and_temperature_trip_the_estop() -> None:
    guard = _guard()
    guard.set_mode(Mode.LIVE)
    guard.check(np.full(6, 30.0), cycle_running=True)
    assert guard.estopped, "no heartbeat ever received in live mode must stop a running cycle"
    guard.reset()
    guard.heartbeat()
    guard.check(np.full(6, 30.0), cycle_running=True)
    assert not guard.estopped
    assert not guard.estopped


def test_overheat_stop_uses_the_5_s_median_not_a_single_read(monkeypatch: pytest.MonkeyPatch) -> None:
    import sortops.safety as safety_module

    clock = {"t": 1000.0}
    monkeypatch.setattr(safety_module.time, "monotonic", lambda: clock["t"])
    guard = _guard()
    cool = np.full(6, 40.0, np.float32)
    hot = np.asarray([40, 40, 90, 40, 40, 40], np.float32)
    for i in range(60):  # 6 s of reads at 10 Hz, one spike in three on motor 3
        clock["t"] += 0.1
        guard.check(hot if i % 3 == 0 else cool, cycle_running=False)
    assert not guard.estopped, "isolated spikes must not stop the cell"
    assert guard.temp_median is not None and guard.temp_median[2] == pytest.approx(40.0)
    for _ in range(60):  # motor 3 now really hot for 6 s
        clock["t"] += 0.1
        guard.check(hot, cycle_running=False)
    assert guard.estopped and "motor 3" in guard.estop_reason
    assert len(guard.temp_history) >= 10


def test_motor_limit_accepts_values_above_80(tmp_path: Path) -> None:
    settings = Settings(tmp_path)
    settings.update({"safety": {"max_motor_temp_c": 95.0}})
    with pytest.raises(ValueError):
        settings.update({"safety": {"max_motor_temp_c": 150.0}})


def test_watchdog_ignored_in_shadow_mode() -> None:
    guard = _guard()
    guard.check(np.full(6, 30.0), cycle_running=True)
    assert not guard.estopped


# Usability -----------------------------------------------------------------------------------------------


def _brick_frame(blur: bool = False) -> np.ndarray:
    import cv2

    image = np.full((480, 640, 3), 110, np.uint8)
    cv2.rectangle(image, (260, 200), (380, 270), (30, 110, 220), -1)
    for x in (290, 350):  # studs and an edge line: a real brick is textured, a flat rectangle is not sharp
        cv2.circle(image, (x, 225), 12, (15, 60, 140), 2)
    cv2.line(image, (260, 250), (380, 250), (15, 60, 140), 2)
    cv2.putText(image, "texture", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return cv2.GaussianBlur(image, (0, 0), 12) if blur else image


def test_usability_accepts_a_sharp_still_brick_at_the_pose() -> None:
    pose = np.asarray(INSPECTION["pose_xyz"], np.float32)
    rated = assess(crop_brick(_brick_frame(), INSPECTION), pose, 0.0, INSPECTION)
    assert rated.usable, rated


@pytest.mark.parametrize(
    ("frame", "offset", "speed", "reason"),
    [
        ("blur", 0.0, 0.0, "blurred"),
        ("sharp", 0.3, 0.0, "off pose"),
        ("sharp", 0.0, 50.0, "moving"),
        ("empty", 0.0, 0.0, "no brick"),
    ],
)
def test_usability_rejects_with_a_reason(frame: str, offset: float, speed: float, reason: str) -> None:
    image = np.full((480, 640, 3), 110, np.uint8) if frame == "empty" else _brick_frame(blur=frame == "blur")
    if frame == "empty":
        image[::7, :, :] = 200  # texture so sharpness passes
    pose = np.asarray(INSPECTION["pose_xyz"], np.float32) + np.asarray([offset, 0, 0], np.float32)
    rated = assess(crop_brick(image, INSPECTION), pose, speed, INSPECTION)
    assert not rated.usable
    assert rated.reason == reason


# Models --------------------------------------------------------------------------------------------------


def test_training_metrics_reads_last_step_and_losses(tmp_path: Path) -> None:
    run = tmp_path / "version_0"
    run.mkdir()
    (run / "metrics.csv").write_text(
        "epoch,lr-AdamW/pg1,lr-AdamW/pg2,step,train/loss,val/loss\r\n"
        "0,,,99,0.9,\r\n19,,,2699,0.19,\r\n19,,,2699,,0.38\r\n"
    )
    assert _training_metrics(tmp_path) == (2699, 0.19, 0.38)
    assert _training_metrics(tmp_path / "missing") == (None, None, None)


def test_simulated_catalog_ranks_full_run_first_and_import_records_provenance(tmp_path: Path) -> None:
    settings = Settings(tmp_path)
    registry = ModelRegistry(settings, tmp_path, simulated=True)
    best = registry.best("pick_present")
    assert best is not None and not best.checkpoint and best.steps == 2699
    provenance = registry.import_model("pick_present", best.id)
    assert provenance["dataset"] == "001-onecube"
    assert registry.active()["pick_present"]["id"] == best.id
    assert json.loads((tmp_path / "models" / "pick_present" / best.id / "provenance.json").read_text())["steps"] == 2699
    with pytest.raises(KeyError):
        registry.import_model("pick_present", "nope")


# Metrics -------------------------------------------------------------------------------------------------


def test_confusion_matrix_from_operator_labels(tmp_path: Path) -> None:
    book = CycleBook(tmp_path)
    for outcome, truth in [("bad", "bad"), ("good", "good"), ("bad", "good"), ("good", "bad"), ("good", None)]:
        record = book.open()
        record.outcome = outcome
        book.close(record)
        if truth:
            book.label(record.id, truth)
    stats = book.stats()
    assert stats["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert stats["accuracy"] == 0.5 and stats["labelled"] == 4
    with pytest.raises(ValueError):
        book.label(1, "maybe")
    assert "outcome" in book.csv().splitlines()[0]


# Simulated cell ------------------------------------------------------------------------------------------


def test_sim_robot_holds_and_drops_without_torque() -> None:
    robot = SimRobot(max_speed=1000.0)
    robot.send(np.asarray([10, 0, 0, 0, 0, 50], np.float32))
    time.sleep(0.15)
    assert robot.read().joints[0] > 0
    robot.hold()
    held = robot.read().joints.copy()
    time.sleep(0.05)
    assert np.allclose(robot.read().joints, held, atol=1e-3)


def test_simulated_cycle_sorts_bricks_end_to_end(tmp_path: Path) -> None:
    # Both cameras scored by Anomalib (the default keeps it on front only).
    settings = Settings(
        tmp_path,
        {"safety": {"watchdog_enabled": False}, "run": {"max_cycles": 2}, "inspection": {"anomalib_cameras": ["front", "wrist"]}},
    )
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=True)
        deadline = time.monotonic() + 80
        while cell.cycle_running() and time.monotonic() < deadline:
            time.sleep(0.2)
        cycles = cell.cycles.cycles()
        assert [c.outcome in {"good", "bad"} for c in cycles] == [True, True], [(c.outcome, c.error) for c in cycles]
        for cycle in cycles:
            assert cycle.frames_usable >= DEFAULTS["inspection"]["min_usable_frames"]
            assert {"pick_present", "inspect"} <= set(cycle.stages)
        summary = cell.latency.summary()
        assert summary["anomaly_front"]["count"] >= 2 and summary["anomaly_wrist"]["count"] >= 2
        assert set(cycles[0].camera_scores) == {"front", "wrist"}
    finally:
        cell.shutdown()


def test_unrecognized_bricks_go_to_the_defect_bin_and_end_the_batch(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}, "run": {"stop_after_no_brick": 2}})
    cell = Cell(settings, tmp_path)
    assert cell.world is not None
    cell.world._draw_brick = lambda *args, **kwargs: None  # type: ignore[method-assign]  # noqa: SLF001
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=True)
        deadline = time.monotonic() + 80
        while cell.cycle_running() and time.monotonic() < deadline:
            time.sleep(0.2)
        cycles = cell.cycles.cycles()
        assert not cell.cycle_running()
        assert [c.outcome for c in cycles] == ["bad", "bad"], [(c.outcome, c.decision_reason, c.error) for c in cycles]
        assert all("no brick recognized" in c.decision_reason for c in cycles)
        assert all("place_bad" in c.stages for c in cycles)
    finally:
        cell.shutdown()


def test_estop_cancels_a_running_cycle(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}})
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=True)
        time.sleep(1.5)
        cell.estop("test")
        time.sleep(0.5)
        assert not cell.cycle_running()
        assert cell.cycles.cycles()[-1].outcome == "aborted"
        held = cell.robot.read().joints.copy()
        time.sleep(0.3)
        assert np.allclose(cell.robot.read().joints, held, atol=0.5), "arm must hold after e-stop"
        with pytest.raises(RuntimeError):
            cell.start_cycles(loop=False)
    finally:
        cell.shutdown()


# Settings validation --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "patch",
    [
        {"safety": {"max_step": 500}},
        {"safety": {"max_step": "6"}},
        {"safety": {"soft_max": [100, 100, 100, 100, 100]}},
        {"safety": {"soft_min": [-150, -95, -95, -95, -100, 0]}},
        {"safety": {"watchdog_enabled": 1}},
        {"decision": {"good_below": 0.7, "bad_above": 0.6}},
        {"devices": {"anomaly": "TPU"}},
    ],
)
def test_unsafe_settings_are_refused_and_not_persisted(tmp_path: Path, patch: dict[str, object]) -> None:
    settings = Settings(tmp_path)
    before = settings.get()
    with pytest.raises(ValueError):
        settings.update(patch)
    assert settings.get() == before
    assert not (tmp_path / "config.json").exists()


def test_valid_settings_update_persists(tmp_path: Path) -> None:
    settings = Settings(tmp_path)
    settings.update({"safety": {"max_step": 4.0}, "decision": {"good_below": 0.3}})
    assert Settings(tmp_path).section("safety")["max_step"] == 4.0


def test_stop_now_and_torque_off_leave_no_command_after_them(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}})
    cell = Cell(settings, tmp_path)
    sent_after: list[str] = []
    original_send = cell.robot.send
    stopped = {"flag": False}

    def spy_send(target: np.ndarray) -> None:
        if stopped["flag"]:
            sent_after.append("send")
        original_send(target)

    cell.robot.send = spy_send  # type: ignore[method-assign]
    cell.start()
    try:
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=True)
        time.sleep(1.0)
        cell.stop_cycle(immediate=True)
        stopped["flag"] = True
        time.sleep(0.4)
        assert not cell.cycle_running() and sent_after == []
        stopped["flag"] = False
        cell.start_cycles(loop=True)
        time.sleep(1.0)
        cell.set_torque(enabled=False)
        stopped["flag"] = True
        time.sleep(0.4)
        assert not cell.cycle_running() and sent_after == []
    finally:
        cell.shutdown()


def test_pick_best_takes_newest_family_then_most_steps() -> None:
    from sortops.models.registry import Candidate, pick_best

    def c(name: str, created: str, steps: int, checkpoint: bool) -> Candidate:
        return Candidate(name, name, "pick_present", "d", created, steps, None, None, "", checkpoint)

    items = [
        c("act-001-onecube", "2026-11-14T10:05:57Z", 2699, False),
        c("act-001-onecube-step2500", "2026-11-14T10:05:57Z", 2500, True),
        c("act-002-nodepth-step5000", "2026-11-14T11:41:25Z", 5000, True),
        c("act-002-nodepth-step7500", "2026-11-14T11:56:26Z", 7500, True),
        c("act-002-nodepth-lowloss-step10000", "2026-11-14T11:55:16Z", 10000, True),
    ]
    best = pick_best(items)
    assert best is not None and best.name == "act-002-nodepth-step7500"
    assert pick_best(items[:2]).name == "act-001-onecube"  # type: ignore[union-attr]
    assert pick_best([]) is None


def test_robot_connection_is_retried_until_the_port_is_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sortops.cell as cell_module

    monkeypatch.setattr(cell_module, "ROBOT_RETRY_S", 0.1)
    cell = Cell(Settings(tmp_path), tmp_path)
    attempts = {"n": 0}
    original = cell.robot.connect

    def flaky_connect() -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("port held by Physical AI Studio session (pid 1)")
        original()

    cell.robot.connect = flaky_connect  # type: ignore[method-assign]
    cell.start()
    try:
        assert not cell.connected and "Studio" in (cell.connect_error or "")
        deadline = time.monotonic() + 3
        while not cell.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cell.connected and cell.connect_error is None and attempts["n"] == 3
    finally:
        cell.shutdown()


def test_operator_verdict_replaces_the_anomaly_model(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}, "decision": {"verdict_source": "operator"}})
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        with pytest.raises(RuntimeError):
            cell.give_verdict("good")
        cell.set_mode(Mode.LIVE)
        cell.start_cycles(loop=True)
        for answer in ("bad", "no_piece"):
            deadline = time.monotonic() + 40
            while not cell.awaiting_verdict() and time.monotonic() < deadline:
                time.sleep(0.1)
            assert cell.awaiting_verdict(), "the cycle must stop and wait for the operator"
            assert cell.snapshot()["awaiting_verdict"] is True
            cell.give_verdict(answer)
            assert not cell.awaiting_verdict()
            with pytest.raises(RuntimeError):
                cell.give_verdict("good")
        deadline = time.monotonic() + 30
        while cell.cycle_running() and time.monotonic() < deadline:
            time.sleep(0.1)
        first, second = cell.cycles.cycles()[:2]
        assert first.outcome == "bad" and first.decision_reason == "operator: bad" and first.scores == []
        assert first.frames_usable > 0 and "await_operator" in first.stages
        assert second.outcome == "no_piece" and not cell.cycle_running()
        # Anomalib still gives its opinion on the frames (shown next to the decision), but never decides.
        assert first.camera_scores == {}
    finally:
        cell.shutdown()


def test_bad_verdict_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Settings(tmp_path).update({"decision": {"verdict_source": "coin"}})


def test_repeated_read_failures_drop_and_reopen_the_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sortops.cell as cell_module

    monkeypatch.setattr(cell_module, "ROBOT_RETRY_S", 0.2)
    cell = Cell(Settings(tmp_path), tmp_path)
    original_read = cell.robot.read
    broken = {"on": False}
    closes = {"n": 0}

    def flaky_read() -> object:
        if broken["on"]:
            raise ConnectionError("Sync read failed with comm result -1")
        return original_read()

    def counting_close() -> None:
        closes["n"] += 1

    cell.robot.read = flaky_read  # type: ignore[method-assign]
    cell.robot.close = counting_close  # type: ignore[method-assign]
    cell.start()
    try:
        assert cell.connected
        broken["on"] = True
        deadline = time.monotonic() + 6
        while cell.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not cell.connected and closes["n"] == 1 and "Sync read failed" in (cell.connect_error or "")
        broken["on"] = False
        deadline = time.monotonic() + 3
        while not cell.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cell.connected and cell.connect_error is None
    finally:
        cell.shutdown()


def test_live_crop_endpoint_serves_the_model_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from sortops.server import create_app

    monkeypatch.setenv("SORTOPS_DATA", str(tmp_path))
    with TestClient(create_app()) as client:
        deadline = time.monotonic() + 5
        response = client.get("/api/inspection/crop.jpg")
        while response.status_code != 200 and time.monotonic() < deadline:
            time.sleep(0.1)
            response = client.get("/api/inspection/crop.jpg")
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        import cv2

        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        size = DEFAULTS["inspection"]["crop_size"]
        assert image.shape[:2] == (size, size)


def test_calibration_puts_good_bricks_low_and_the_line_above_them() -> None:
    from sortops.models.runtime import calibrated, calibration_from

    goods = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 71.0]
    cal = calibration_from(goods)
    assert all(calibrated(g, cal) < 0.5 for g in goods)
    assert calibrated(cal["median"], cal) == pytest.approx(0.0, abs=0.05)
    assert calibrated(cal["threshold"], cal) == pytest.approx(0.5)
    assert calibrated(cal["threshold"] + cal["spread"], cal) == pytest.approx(1.0)
    flat = calibration_from([50.0] * 12)  # one still brick: the floor keeps a usable spread
    assert flat["spread"] > 0 and calibrated(50.0, flat) < 0.5 < calibrated(60.0, flat)


def test_home_button_moves_the_arm_home_only_in_live(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"safety": {"watchdog_enabled": False}, "home": {"speed": 120.0}})
    cell = Cell(settings, tmp_path)
    cell.start()
    try:
        cell.robot.send(np.asarray([30.0, -20.0, 40.0, 10.0, 20.0, 50.0], np.float32))
        time.sleep(1.0)
        with pytest.raises(RuntimeError, match="Live"):
            cell.go_home()
        cell.set_mode(Mode.LIVE)
        cell.go_home()
        deadline = time.monotonic() + 10
        while cell.cycle_running() and time.monotonic() < deadline:
            time.sleep(0.1)
        home = np.asarray(DEFAULTS["home"]["joints"], np.float32)
        assert np.allclose(cell.robot.read().joints[:5], home[:5], atol=2.5)
        assert cell.phase == "idle"
        with pytest.raises(ValueError):
            settings.update({"home": {"joints": [0, 0, 0, 0, 0]}})
    finally:
        cell.shutdown()



# Multi-camera fusion, manual defect line, migration ---------------------------------------------------------


def test_fusion_heuristics() -> None:
    from sortops.pipeline.inspection import decide_cameras, fuse

    scores = {"front": 0.2, "wrist": 0.9}
    assert fuse(scores, "max", {}) == 0.9
    assert fuse(scores, "min", {}) == 0.2
    assert fuse(scores, "mean", {}) == pytest.approx(0.55)
    assert fuse(scores, "weighted", {"front": 0.75, "wrist": 0.25}) == pytest.approx(0.375)
    cfg = {**DECISION, "fusion": "max"}
    both = {"front": [0.1, 0.12, 0.1], "wrist": [0.8, 0.85, 0.9]}
    assert decide_cameras(both, 6, True, cfg, "p90", 2).verdict == Verdict.BAD
    assert decide_cameras(both, 6, True, {**cfg, "fusion": "min"}, "p90", 2).verdict == Verdict.GOOD
    # A camera with too few usable frames does not vote; the other one decides alone.
    one = decide_cameras({"front": [0.1, 0.1], "wrist": [0.95]}, 3, True, cfg, "p90", 2)
    assert one.verdict == Verdict.GOOD and set(one.camera_scores) == {"front"}


def test_manual_defect_line_moves_the_decision(tmp_path: Path) -> None:
    from sortops.models.runtime import calibrated

    settings = Settings(tmp_path)
    cell = Cell(settings, tmp_path)
    cell.registry.import_model("anomaly_wrist", cell.registry.best("anomaly_wrist").id)  # type: ignore[union-attr]
    values = cell.set_defect_line("wrist", 70.0, 10.0)
    assert values["manual"] and cell.registry.calibration("anomaly_wrist")["threshold"] == 70.0
    assert calibrated(65.0, values) < 0.5 < calibrated(75.0, values)
    cell.set_defect_line("wrist", 60.0, None)  # spread kept from before
    assert cell.registry.calibration("anomaly_wrist")["spread"] == 10.0
    with pytest.raises(ValueError):
        cell.set_defect_line("wrist", 60.0, -1.0)


def test_single_anomaly_model_migrates_to_the_wrist_camera(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "anomaly" / "m1").mkdir(parents=True)
    (models / "anomaly" / "m1" / "provenance.json").write_text(json.dumps({"id": "m1", "name": "pc", "role": "anomaly", "simulated": True}))
    (models / "active.json").write_text(json.dumps({"anomaly": "m1"}))
    registry = ModelRegistry(Settings(tmp_path), tmp_path, simulated=True)
    active = registry.active()
    assert active["anomaly_wrist"]["id"] == "m1" and active["anomaly_front"] is None  # type: ignore[index]
    assert "anomaly" not in json.loads((models / "active.json").read_text())


def test_live_scores_cover_every_inspection_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from sortops.server import create_app

    monkeypatch.setenv("SORTOPS_DATA", str(tmp_path))
    with TestClient(create_app()) as client:
        deadline = time.monotonic() + 5
        live = client.get("/api/anomaly/live").json()
        while any("error" in v for v in live.values()) and time.monotonic() < deadline:
            time.sleep(0.2)
            live = client.get("/api/anomaly/live").json()
        assert set(live) == {"front", "wrist"} and all("score" in v for v in live.values())
        assert client.get("/api/inspection/front/crop.jpg").status_code == 200
        assert client.get("/api/inspection/nope/crop.jpg").status_code == 404
        assert client.post("/api/anomaly/nope/defect-line", json={"threshold": 1}).status_code == 404
