"""``python -m sortops.smoke`` — check the cell's models and cameras on the robot PC without touching the arm.

Imports the newest, most-trained Studio model of each role (unless ``--no-import``), compiles every policy on
each device, runs it on a synthetic observation of the right shapes and prints load/latency per device; with
``--cameras`` it also attaches to the Studio cameras for a few seconds and reports the frames received.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sortops.config import POLICY_ROLES, Settings, data_dir_from_env
from sortops.metrics import LatencyBook
from sortops.models.registry import ModelRegistry
from sortops.models.runtime import OpenVinoPolicy, manifest_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="CPU,GPU,NPU")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--no-import", action="store_true")
    parser.add_argument("--cameras", action="store_true")
    args = parser.parse_args()

    data = data_dir_from_env()
    settings = Settings(data)
    registry = ModelRegistry(settings, data, simulated=False)
    report: dict[str, object] = {}
    for role in POLICY_ROLES:
        if not args.no_import:
            best = registry.best(role)
            if best is None:
                report[role] = "no Studio model for this role"
                continue
            registry.import_model(role, best.id)
        export = registry.export_dir(role)
        if export is None:
            report[role] = "nothing imported"
            continue
        provenance = registry.active()[role] or {}
        inputs = manifest_images(export)
        observation_images = {name.removeprefix("images."): np.zeros((h, w, 3), np.uint8) for name, (h, w) in inputs.items()}
        per_device: dict[str, object] = {}
        for device in args.devices.split(","):
            book = LatencyBook()
            try:
                policy = OpenVinoPolicy(role, provenance.get("name", role), export, device, book)
                for _ in range(args.runs + 1):
                    policy.reset()
                    chunk = policy.predict(np.zeros(6, np.float32), observation_images, 0)
                stats = book.summary()
                per_device[device] = {
                    "load_ms": stats[f"load:{role}"]["last"],
                    "p50_ms": stats[role]["p50"],
                    "p95_ms": stats[role]["p95"],
                    "chunk": list(chunk.shape),
                }
            except Exception as error:  # noqa: BLE001 - report every device
                per_device[device] = f"failed: {str(error)[:200]}"
        report[role] = {"model": provenance.get("name"), "steps": provenance.get("steps"), "inputs": inputs, "devices": per_device}

    if args.cameras:
        from sortops.hw.cameras import StudioCameraHub  # noqa: PLC0415

        hardware = settings.section("hardware")
        hub = StudioCameraHub(hardware["studio_url"], hardware["studio_project"], hardware["studio_backend_src"], hardware["cameras"])
        hub.start()
        time.sleep(6)
        report["cameras"] = {
            key: list(frame.image.shape) for key in hub.keys() if (frame := hub.latest(key)) is not None
        } | {"errors": dict(hub.errors)}
        hub.stop()

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
