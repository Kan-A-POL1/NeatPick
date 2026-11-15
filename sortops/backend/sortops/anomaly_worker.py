"""Anomalib training run by SortOps' "Train" button, in Anomalib's own venv (no sortops import here).

    <anomalib python> anomaly_worker.py --root <dir> --model patchcore --out <dir> [--size 256]

``root`` holds the reviewed export of ONE camera: train/good, test/good, test/defect, ground_truth/defect. Trains,
tests on the held-out good frames + the defects (image AUROC/F1, pixel AUROC when masks exist), exports OpenVINO to
``out`` (model.xml/.bin, picked up by SortOps' anomaly model roots) and writes ``out/metrics.json``. Progress lines
on stdout start with ``PROGRESS <0-100> <message>``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

MODELS = ("patchcore", "padim", "efficient_ad", "fre", "dinomaly")


def progress(value: int, message: str) -> None:
    print(f"PROGRESS {value} {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="patchcore")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--work", type=Path, default=None)
    args = parser.parse_args()
    started = time.time()
    progress(2, "Loading Anomalib")
    from anomalib import models
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine

    root = args.root
    has = {d: any((root / d).glob("*.png")) for d in ("test/good", "test/defect", "ground_truth/defect")}
    if not any((root / "train/good").glob("*.png")):
        print("No reviewed good frame to train on", file=sys.stderr)
        return 2
    datamodule = Folder(
        name="sortops",
        root=root,
        normal_dir="train/good",
        abnormal_dir="test/defect" if has["test/defect"] else None,
        normal_test_dir="test/good" if has["test/good"] else None,
        mask_dir="ground_truth/defect" if has["ground_truth/defect"] and has["test/defect"] else None,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=4,
        test_split_mode="from_dir" if has["test/defect"] or has["test/good"] else "synthetic",
        seed=0,
    )
    model = {
        "patchcore": lambda: models.Patchcore(),
        "padim": lambda: models.Padim(),
        "efficient_ad": lambda: models.EfficientAd(),
        "fre": lambda: models.Fre(),
        "dinomaly": lambda: models.Dinomaly(),
    }[args.model]()
    work = args.work or (args.out.parent / f".{args.out.name}-work")
    epochs = {"patchcore": 1, "padim": 1, "fre": 20, "efficient_ad": 20, "dinomaly": 20}[args.model]
    engine = Engine(default_root_dir=str(work), max_epochs=epochs, logger=False, enable_progress_bar=False)
    progress(10, f"Training {args.model}")
    engine.fit(model=model, datamodule=datamodule)
    progress(70, "Testing on the held-out frames")
    metrics: dict[str, float] = {}
    if has["test/defect"] or has["test/good"]:
        results = engine.test(model=model, datamodule=datamodule, verbose=False) or [{}]
        metrics = {k: float(v) for k, v in results[0].items()}
    progress(85, "Exporting OpenVINO")
    args.out.mkdir(parents=True, exist_ok=True)
    exported = engine.export(model=model, export_type=ExportType.OPENVINO, export_root=work / "export",
                             input_size=(args.size, args.size))
    xml = next((work / "export").rglob("*.xml"), None) if exported is None or not Path(exported).exists() else Path(exported)
    if xml is None or not xml.exists():
        print("OpenVINO export produced no model.xml", file=sys.stderr)
        return 3
    for f in (xml, xml.with_suffix(".bin")):
        shutil.copy2(f, args.out / f"model{f.suffix}")
    counts = {d: len(list((root / d).glob("*.png"))) for d in ("train/good", "test/good", "test/defect")}
    (args.out / "metrics.json").write_text(json.dumps({
        "model": args.model, "size": args.size, "seconds": round(time.time() - started, 1),
        "images": counts, "metrics": metrics,
    }, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    progress(100, "Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
