# Anomaly dataset tools

SortOps flags defective bricks with Anomalib models trained on good bricks only. These tools build that training set,
check it before training, and fix two Anomalib Studio issues we hit along the way.

| File | What it does |
|---|---|
| `capture_app.py` | Web app (127.0.0.1:8105) to shoot and curate photos of good and defective bricks. |
| `dataset_report.py` | Report on a set of good photos: photo hygiene, framing, colour coverage, and a PatchCore learning curve. |
| `quick_eval.py` | Train → test check of a captured session: PatchCore fitted on the train goods, scored with SortOps' own score. |
| `extract_shots.py` | Extracts "brick presented to the camera" stills from a recorded Studio dataset, using joint states only. |
| `patches/` | Two fixes for [Anomalib Studio](https://github.com/open-edge-platform/anomalib) (`application/`), `main` at `437669f`. |

## Capture app

```bash
<studio>/application/backend/.venv/bin/python capture_app.py    # http://127.0.0.1:8105
```

- **Shoot.** Space or click takes 1 photo; holding it takes a burst. Each photo is labelled good or defect, with its
  camera and the brick colour. The Studio cameras are read through Studio's shared-camera transport, so capture works
  while Studio previews or records.
- **Same crop as SortOps.** Every photo is cut by `sortops.pipeline.brick_crop` with SortOps' live `inspection` settings
  (read from its API, or from its `data/config.json`). That is the same code that cuts the frames SortOps scores, so the
  model learns on exactly what it will see. A frame without a brick is not saved. The full frame is kept, so photos can
  be re-cut after a crop setting changes.
- **Stable split.** Good photos go to train or test by a hash of their file name (about 20 % test), and defects are
  always test. The split does not change when photos are added.
- **Evaluate.** Runs `quick_eval.py`.
- **Send.** Uploads the train goods, and the defects marked as anomalous, to an Anomalib Studio project. Each photo is
  sent once per project. Test goods stay local: they are what the evaluation measures.

## Dataset report

```bash
<anomalib venv>/bin/python dataset_report.py GOOD_DIR [--defects DIR] [--out REPORT_DIR]
```

Per camera:
- **Hygiene**: blur (Laplacian variance), under- or over-exposure, near-duplicates.
- **Framing**: position and size of the brick. Anomalib wants the brick at the same place and scale in every photo.
- **Colour coverage**: every brick colour seen at test time must be in the good set, otherwise it is flagged as an
  anomaly.
- **Model readiness**: PatchCore trained on growing subsets, with AUROC, recall at ~1 % false rejects, and false-reject
  rate. The curve tells you whether more photos still help or whether framing and lighting must be fixed.

Without photos of real defects, the defects are drawn on held-out good photos (marker strokes, dots, tape). The score
then answers one precise question: can a model trained on these photos catch a marker stroke. It does not replace
real defects.

## Anomalib Studio patches

```bash
cd anomalib && git apply ../NeatPick/anomaly/patches/*.patch
```

- **`media-anomalous.patch`**: the image upload endpoint accepts an `is_anomalous` form field.
  *Why:* uploads were always stored as normal, so the capture app could not send defective samples, and the defects
  ended up in the training set.
  *Gave us:* real defects uploaded as test anomalies, kept out of training.
- **`synthetic-spawn.patch`**: pickled copies of `SyntheticAnomalyDataset` never delete the shared temporary folder.
  *Why:* with spawned DataLoader workers, each worker's copy deleted the synthetic-anomaly folder when it exited, and
  training failed with `FileNotFoundError`.
  *Gave us:* Anomalib Studio trainings with synthetic validation anomalies run to the end.
