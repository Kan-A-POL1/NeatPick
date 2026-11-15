"""HTTP + WebSocket API of the cell, and the static web UI.

The UI connects to ``/api/ws``: the server pushes a state snapshot ~15 times a second and the UI answers with
heartbeats (the live-mode watchdog). Everything else is plain REST under ``/api``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from sortops.cell import Cell
from sortops.config import ANOMALY_ROLES, INSPECTION_CAMERAS, POLICY_ROLES, ROLES, Settings, data_dir_from_env
from sortops.hw.cameras import encode_jpeg
from sortops.pipeline.annotation import remove_quietly
from sortops.pipeline.brick_crop import crop_brick
from sortops.safety import Mode

WEB_DIST = Path(os.environ.get("SORTOPS_WEB", Path(__file__).resolve().parents[2] / "web" / "dist"))
DEVICES = ("CPU", "GPU", "NPU")


class StartBody(BaseModel):
    loop: bool = True
    forever: bool = False


class StopBody(BaseModel):
    immediate: bool = False


class ModeBody(BaseModel):
    mode: Mode


class TorqueBody(BaseModel):
    enabled: bool


class ReasonBody(BaseModel):
    reason: str = "operator"


class ImportBody(BaseModel):
    role: str
    model_id: str


class RoleBody(BaseModel):
    role: str


class BenchmarkBody(BaseModel):
    role: str
    devices: list[str] = list(DEVICES)
    runs: int = 20


class VerdictBody(BaseModel):
    verdict: str
    run: str
    cycle: int
    """The brick the answer is for (``snapshot.inspection``): a late answer for a previous brick is refused (409)."""
    annotation: dict[str, Any] | None = None
    """Optional drawing of the brick: ``{"cameras": {cam: {"shot", "piece", "defects"}}, "note"}`` (frame coords)."""


class TrainBody(BaseModel):
    runs: list[str]
    camera: str
    model: str = "patchcore"
    name: str | None = None


class CalibrateBody(BaseModel):
    seconds: float = 8.0


class DefectLineBody(BaseModel):
    threshold: float
    spread: float | None = None


class LabelBody(BaseModel):
    truth: str | None


def create_app(settings: Settings | None = None) -> FastAPI:
    data_dir = data_dir_from_env()
    settings = settings or Settings(data_dir)
    cell = Cell(settings, data_dir)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(cell.start)
        try:
            yield
        finally:
            await asyncio.to_thread(cell.shutdown)

    app = FastAPI(title="SortOps", lifespan=lifespan)
    app.state.cell = cell

    def fail(error: Exception, status: int = 409) -> HTTPException:
        return HTTPException(status_code=status, detail=str(error))

    # Live state ------------------------------------------------------------------------------------------

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return cell.snapshot()

    @app.websocket("/api/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()

        async def receive() -> None:
            while True:
                message = json.loads(await socket.receive_text())
                if message.get("type") == "heartbeat":
                    cell.safety.heartbeat()

        reader = asyncio.create_task(receive())
        try:
            while not reader.done():
                await socket.send_text(json.dumps({"type": "state", **cell.snapshot()}))
                await asyncio.sleep(1 / 15)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            reader.cancel()

    # Cycle & safety --------------------------------------------------------------------------------------

    @app.post("/api/cycle/start")
    def start(body: StartBody) -> dict[str, Any]:
        try:
            cell.start_cycles(loop=body.loop, forever=body.forever)
        except RuntimeError as error:
            raise fail(error) from error
        return cell.snapshot()

    @app.post("/api/cycle/stop")
    def stop(body: StopBody) -> dict[str, Any]:
        cell.stop_cycle(immediate=body.immediate)
        cell.log("info", "Stop requested" + (" (immediate)" if body.immediate else " (after this cycle)"))
        return cell.snapshot()

    @app.post("/api/estop")
    def estop(body: ReasonBody) -> dict[str, Any]:
        cell.estop(f"operator: {body.reason}")
        return cell.snapshot()

    @app.post("/api/estop/reset")
    def estop_reset() -> dict[str, Any]:
        if cell.cycle_running():
            raise fail(RuntimeError("Wait for the cycle to stop before resetting"))
        cell.safety.reset()
        cell.log("info", "E-stop reset (mode SHADOW)")
        return cell.snapshot()

    @app.post("/api/mode")
    def set_mode(body: ModeBody) -> dict[str, Any]:
        try:
            cell.set_mode(body.mode)
        except RuntimeError as error:
            raise fail(error) from error
        return cell.snapshot()

    @app.post("/api/torque")
    def torque(body: TorqueBody) -> dict[str, Any]:
        try:
            cell.set_torque(enabled=body.enabled)
        except Exception as error:  # noqa: BLE001 - serial errors surface to the operator
            raise fail(error) from error
        return cell.snapshot()

    @app.post("/api/robot/release")
    def release_robot() -> dict[str, Any]:
        """Free the arm port for Physical AI Studio; SortOps stays up (cameras, data) but drives nothing."""
        try:
            cell.release_robot()
        except RuntimeError as error:
            raise fail(error) from error
        return cell.snapshot()

    @app.post("/api/robot/take")
    def take_robot() -> dict[str, Any]:
        cell.take_robot()
        return cell.snapshot()

    @app.post("/api/verdict")
    def verdict(body: VerdictBody) -> dict[str, Any]:
        if body.verdict not in {"good", "bad", "no_piece"}:
            raise fail(ValueError("verdict must be good, bad or no_piece"), 422)
        try:
            cell.give_verdict(body.verdict, body.annotation, body.run, body.cycle)
        except RuntimeError as error:
            raise fail(error) from error
        return cell.snapshot()

    @app.post("/api/home")
    def home() -> dict[str, Any]:
        try:
            cell.go_home()
        except RuntimeError as error:
            raise fail(error) from error
        return cell.snapshot()

    @app.post("/api/home/teach")
    def teach_home() -> dict[str, Any]:
        try:
            return {"joints": cell.teach_home()}
        except (RuntimeError, ValueError) as error:
            raise fail(error) from error

    @app.post("/api/pose/teach")
    def teach() -> dict[str, Any]:
        state = cell.state()
        if state is None:
            raise fail(RuntimeError("No robot state yet"))
        xyz = [round(float(v), 4) for v in state.ee_xyz]
        settings.update({"inspection": {"pose_xyz": xyz}})
        cell.log("info", f"Inspection pose taught at {xyz}")
        return settings.get()

    # Settings --------------------------------------------------------------------------------------------

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return settings.get()

    @app.patch("/api/config")
    async def patch_config(request: Request) -> dict[str, Any]:
        patch = await request.json()
        if not isinstance(patch, dict):
            raise fail(ValueError("Body must be an object"), 422)
        if "hardware" in patch:
            raise fail(ValueError("Hardware settings are read at start: edit data/config.json and restart"), 422)
        for role, device in patch.get("devices", {}).items():
            if role not in ROLES or device not in DEVICES:
                raise fail(ValueError(f"Bad device {device!r} for {role!r}"), 422)
        try:
            values = settings.update(patch)
        except ValueError as error:
            raise fail(error, 422) from error
        if "devices" in patch:
            cell.invalidate_models()
        return values

    # Models ----------------------------------------------------------------------------------------------

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        try:
            catalog = {role: [asdict(c) for c in items] for role, items in cell.registry.catalog().items()}
            error = None
        except Exception as exc:  # noqa: BLE001 - Studio down: still show the active models
            catalog, error = {}, f"Physical AI Studio catalogue unavailable: {exc}"
        return {
            "catalog": catalog,
            "active": cell.registry.active(),
            "devices": settings.section("devices"),
            "errors": cell.model_errors,
            "catalog_error": error,
            "benchmarks": cell.benchmarks,
            "anomaly_calibration": {role: cell.registry.calibration(role) for role in ANOMALY_ROLES},
            "roles": list(ROLES),
        }

    @app.post("/api/models/import")
    def import_model(body: ImportBody) -> dict[str, Any]:
        if body.role not in ROLES:
            raise fail(ValueError(f"Unknown role {body.role}"), 422)
        if cell.cycle_running():
            raise fail(RuntimeError("Stop the cycle before changing models"))
        try:
            provenance = cell.registry.import_model(body.role, body.model_id)
        except (KeyError, FileNotFoundError) as error:
            raise fail(error, 404) from error
        cell.invalidate_models()
        cell.log("info", f"Imported {provenance['name']} for {body.role}")
        return provenance

    @app.post("/api/models/import-latest")
    def import_latest() -> dict[str, Any]:
        if cell.cycle_running():
            raise fail(RuntimeError("Stop the cycle before changing models"))
        imported: dict[str, Any] = {}
        for role in POLICY_ROLES:
            best = cell.registry.best(role)
            if best is not None:
                imported[role] = cell.registry.import_model(role, best.id)
                cell.log("info", f"Imported {best.name} ({best.steps} steps) for {role}")
        cell.invalidate_models()
        return imported

    def inspection_camera(camera: str) -> str:
        if camera not in INSPECTION_CAMERAS:
            raise fail(ValueError(f"Unknown inspection camera {camera}"), 404)
        return camera

    @app.post("/api/anomaly/{camera}/calibrate")
    async def calibrate(camera: str, body: CalibrateBody) -> dict[str, Any]:
        inspection_camera(camera)
        if not 2 <= body.seconds <= 60:
            raise fail(ValueError("seconds must be within [2, 60]"), 422)
        try:
            return await asyncio.to_thread(cell.calibrate_anomaly, camera, body.seconds)
        except RuntimeError as error:
            raise fail(error) from error

    @app.post("/api/anomaly/{camera}/defect-line")
    def defect_line(camera: str, body: DefectLineBody) -> dict[str, Any]:
        inspection_camera(camera)
        try:
            return cell.set_defect_line(camera, body.threshold, body.spread)
        except KeyError as error:
            raise fail(error, 404) from error
        except ValueError as error:
            raise fail(error, 422) from error

    @app.post("/api/anomaly/{camera}/calibration/reset")
    def reset_calibration(camera: str) -> dict[str, Any]:
        inspection_camera(camera)
        try:
            cell.registry.save_calibration(None, f"anomaly_{camera}")
        except KeyError as error:
            raise fail(error, 404) from error
        cell.invalidate_models()
        cell.log("info", f"{camera} anomaly score back to the export's own scale")
        return {"calibration": None}

    @app.get("/api/anomaly/live")
    async def anomaly_live() -> dict[str, Any]:
        """Each inspection camera's current crop scored now, to set a defect line by hand."""
        return await asyncio.to_thread(cell.live_scores)

    @app.post("/api/models/deactivate")
    def deactivate(body: RoleBody) -> dict[str, Any]:
        cell.registry.deactivate(body.role)
        cell.invalidate_models()
        return cell.registry.active()

    @app.post("/api/models/benchmark")
    async def benchmark(body: BenchmarkBody) -> dict[str, Any]:
        if body.role not in ROLES or any(d not in DEVICES for d in body.devices) or not 1 <= body.runs <= 200:
            raise fail(ValueError("Bad benchmark request"), 422)
        try:
            return await asyncio.to_thread(cell.benchmark, body.role, body.devices, body.runs)
        except RuntimeError as error:
            raise fail(error) from error

    # Metrics ---------------------------------------------------------------------------------------------

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        return {
            "latency": cell.latency.summary(),
            "latency_recent": {key: cell.latency.recent(key) for key in ("pick_present", "place_good", "place_bad", *ANOMALY_ROLES)},
            "system": cell.monitor.series(180),
            "motor_temps": list(cell.safety.temp_history)[-300:],
            "cycles": [{**asdict(c), "duration": c.duration} for c in cell.cycles.cycles()[-100:]],
            "stats": cell.cycles.stats(),
            "events": list(cell.events)[-80:],
            "shots": [
                {key: value for key, value in asdict(shot).items() if key not in {"jpeg", "heat_jpeg"}}
                | {"has_heat": bool(shot.heat_jpeg)}
                for shot in list(cell.shots)[-24:]
            ],
        }

    @app.post("/api/cycles/{cycle_id}/label")
    def label(cycle_id: int, body: LabelBody) -> dict[str, Any]:
        try:
            return asdict(cell.cycles.label(cycle_id, body.truth))
        except KeyError as error:
            raise fail(error, 404) from error
        except ValueError as error:
            raise fail(error, 422) from error

    @app.post("/api/runs/new")
    def new_run() -> dict[str, str]:
        if cell.cycle_running():
            raise fail(RuntimeError("Stop the cycle first"))
        return {"run": cell.cycles.new_run()}

    @app.get("/api/runs/export.csv")
    def export_csv() -> PlainTextResponse:
        return PlainTextResponse(
            cell.cycles.csv(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="sortops-{cell.cycles.run_id}.csv"'},
        )

    @app.get("/api/report.json")
    def report() -> JSONResponse:
        """Everything a judge may ask for, in one file: config, models, latencies, stats, cycles, telemetry."""
        return JSONResponse(
            {
                "run": cell.cycles.run_id,
                "config": settings.get(),
                "models": cell.registry.active(),
                "latency": cell.latency.summary(),
                "benchmarks": cell.benchmarks,
                "stats": cell.cycles.stats(),
                "cycles": [{**asdict(c), "duration": c.duration} for c in cell.cycles.cycles()],
                "system": cell.monitor.series(300),
            },
            headers={"Content-Disposition": f'attachment; filename="sortops-report-{cell.cycles.run_id}.json"'},
        )

    # Images ----------------------------------------------------------------------------------------------

    def live_crop(camera: str | None = None) -> bytes | None:
        """A camera's latest frame cut by the shared brick crop: exactly what its model would score."""
        name = camera or settings.section("inspection")["camera"]
        frame = cell.cameras.latest(name)
        if frame is None:
            return None
        crop = crop_brick(frame.image, cell.crop_settings(name))
        image = cell.model_image(name, frame.image, crop.image).copy()
        if not crop.found:
            # Red frame = no brick found: the fallback crop (search zone) is shown, it would not be scored as a brick.
            cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (230, 60, 50), 6)
        return encode_jpeg(image, 80)

    @app.get("/api/inspection/crop.jpg")
    def crop_jpeg() -> Response:
        data = live_crop()
        if data is None:
            raise HTTPException(404, "No inspection frame")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/inspection/{camera}/crop.jpg")
    def camera_crop_jpeg(camera: str) -> Response:
        data = live_crop(inspection_camera(camera))
        if data is None:
            raise HTTPException(404, "No inspection frame")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/inspection/crop.mjpg")
    async def crop_stream() -> StreamingResponse:
        async def frames() -> AsyncIterator[bytes]:
            while True:
                data = await asyncio.to_thread(live_crop)
                if data is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                await asyncio.sleep(1 / 8)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/cams/{key}.jpg")
    def cam_jpeg(key: str) -> Response:
        frame = cell.cameras.latest(key)
        if frame is None:
            raise HTTPException(404, f"No frame for {key}")
        return Response(encode_jpeg(frame.image), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cams/{key}.mjpg")
    async def cam_stream(key: str) -> StreamingResponse:
        async def frames() -> AsyncIterator[bytes]:
            last = 0.0
            while True:
                frame = cell.cameras.latest(key)
                if frame is not None and frame.timestamp != last:
                    last = frame.timestamp
                    jpeg = await asyncio.to_thread(encode_jpeg, frame.image, 70)
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1 / 12)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/shots/{shot_id}.jpg")
    def shot(shot_id: int, heat: bool = False) -> Response:
        for item in cell.shots:
            if item.id == shot_id:
                data = item.heat_jpeg if heat and item.heat_jpeg else item.jpeg
                return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})
        raise HTTPException(404, "Shot expired")

    # Inspection data (Data tab, Anomalib export) ----------------------------------------------------------

    store = cell.inspection_store

    @app.get("/api/inspection/runs")
    def inspection_runs() -> list[dict[str, Any]]:
        return store.runs()

    @app.get("/api/inspection/runs/{run}/cycles")
    def inspection_cycles(run: str) -> list[dict[str, Any]]:
        try:
            return store.cycles(run)
        except KeyError as error:
            raise fail(error, 404) from error

    @app.get("/api/inspection/runs/{run}/cycles/{cycle}")
    def inspection_cycle(run: str, cycle: int) -> dict[str, Any]:
        try:
            return store.cycle(run, cycle)
        except KeyError as error:
            raise fail(error, 404) from error

    @app.post("/api/inspection/runs/{run}/cycles/{cycle}/annotation")
    def inspection_annotate(run: str, cycle: int, body: dict[str, Any]) -> dict[str, Any]:
        """Correct a cycle's annotation afterwards (verdict, piece box, defects)."""
        try:
            store.cycle(run, cycle)
            return store.save_annotation(run, cycle, {**body, "source": "operator"})
        except KeyError as error:
            raise fail(error, 404) from error
        except ValueError as error:
            raise fail(error, 422) from error

    @app.get("/api/inspection/runs/{run}/review")
    def inspection_review(run: str) -> dict[str, Any]:
        """Every stored frame of the run, with its review or a proposal to confirm (Data -> Review)."""
        try:
            if run not in {item["run"] for item in store.runs()}:
                raise KeyError(run)
            return store.review_queue(run)
        except KeyError as error:
            raise fail(error, 404) from error

    @app.post("/api/inspection/runs/{run}/cycles/{cycle}/shots/{shot}/review")
    def inspection_review_shot(run: str, cycle: int, shot: int, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return store.save_review(run, cycle, shot, body)
        except KeyError as error:
            raise fail(error, 404) from error
        except (ValueError, TypeError) as error:
            raise fail(error, 422) from error

    @app.get("/api/anomaly/train")
    def anomaly_train_status() -> dict[str, Any]:
        return {"job": cell.anomaly_trainer.status()}

    @app.post("/api/anomaly/train")
    def anomaly_train(body: TrainBody) -> dict[str, Any]:
        try:
            return {"job": cell.anomaly_trainer.start(body.runs, body.camera, body.model, body.name)}
        except KeyError as error:
            raise fail(error, 404) from error
        except ValueError as error:
            raise fail(error, 422) from error
        except RuntimeError as error:
            raise fail(error) from error

    @app.post("/api/anomaly/train/cancel")
    def anomaly_train_cancel() -> dict[str, Any]:
        try:
            cell.anomaly_trainer.cancel()
        except RuntimeError as error:
            raise fail(error) from error
        return {"job": cell.anomaly_trainer.status()}

    @app.get("/api/inspection/runs/{run}/cycles/{cycle}/files/{name}")
    def inspection_file(run: str, cycle: int, name: str) -> FileResponse:
        try:
            path = store.file(run, cycle, name)
        except KeyError as error:
            raise HTTPException(404, "No such file") from error
        return FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/inspection/runs/{run}/dataset.zip")
    def inspection_zip(run: str, size: int = 256, only: str = "reviewed") -> FileResponse:
        """Anomalib-ready dataset of one run, or of every run with ``run = all`` (written to disk, then streamed)."""
        runs = [item["run"] for item in store.runs()] if run == "all" else [run]
        if run != "all" and run not in {item["run"] for item in store.runs()}:
            raise HTTPException(404, "No inspection data for this run")
        if not 64 <= size <= 1024:
            raise HTTPException(422, "size must be within [64, 1024]")
        exports = cell.data_dir / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        dest = exports / f"sortops-inspection-{run}-{os.getpid()}-{id(runs)}.zip"
        if only not in ("reviewed", "all"):
            raise HTTPException(422, "only must be 'reviewed' (Data -> Review) or 'all' (cycle annotations)")
        store.export_zip(runs, dest, size, reviewed_only=only == "reviewed")
        return FileResponse(
            dest,
            media_type="application/zip",
            filename=f"sortops-inspection-{run}.zip",
            background=BackgroundTask(remove_quietly, dest),
        )

    # 3D model & web UI -----------------------------------------------------------------------------------

    urdf_dir = Path(settings.section("hardware")["urdf_dir"])
    if urdf_dir.exists():
        app.mount("/urdf", StaticFiles(directory=urdf_dir), name="urdf")

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = (WEB_DIST / path).resolve()
            if path and candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")

    return app
