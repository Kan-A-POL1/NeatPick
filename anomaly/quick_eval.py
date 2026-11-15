#!/usr/bin/env python
"""Quick train → test check of an Anomalib dataset: PatchCore fitted on the train goods, scored on the held-out goods
and the defects, per camera. Run by capture_app.py ("Évaluer") with the Anomalib venv; reuses dataset_report.py.

Usage: quick_eval.py SPLIT_JSON --out DIR [--device cpu|xpu]
SPLIT_JSON = {"<camera>": {"train": [paths], "test": [paths of good photos], "defect": [paths]}}

The photos are already cut by SortOps' brick crop (capture_app.py), so this measures the model SortOps will run on
the frames it will see. Without real defects, synthetic ones (marker strokes, dots, tape) are drawn on the test goods.
Scores are raw PatchCore distances (not the 0..1 normalized scores of a Studio export): read them relative to each other.
The decision score is SortOps' own (sortops.pipeline.anomaly_score, imported from --sortops-src): localized peak (the
raw score) and whole-map mean in good-brick units, combined quadratically; the AUROC of each part is printed too. Its
good-brick references come from the test goods here (SortOps calibrates on live good bricks).
Prints per camera: AUROC, recall and false-reject rate at a threshold set on the test goods (their max: no good photo
rejected), then writes DIR/<camera>_test.jpg (test photos sorted by score, defects framed red, missed ones dashed)
and DIR/eval.json.
"""
import argparse, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_report import auroc, load_model, synth_defect  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("split"); p.add_argument("--out", required=True); p.add_argument("--device", default="cpu")
p.add_argument("--sortops-src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "sortops", "backend"))
a = p.parse_args()
sys.path.insert(0, a.sortops_src)
from sortops.config import DEFAULTS  # noqa: E402  (SortOps: the same score as the sorting cell)
from sortops.pipeline.anomaly_score import combined, map_terms, term_references  # noqa: E402

WEIGHT = DEFAULTS["inspection"]["score_peak_weight"]
FRACTION = DEFAULTS["inspection"]["score_peak_fraction"]


def fit_maps(model, torch, train, test, device, ratio=0.1):
    """PatchCore memory bank on ``train``; (raw image score, raw anomaly map) of every ``test`` image."""
    from dataset_report import to_batch
    model.train()
    store = getattr(model, "embedding_store", None)
    if store is not None:
        store.clear()
    embs, out = [], []
    with torch.no_grad():
        for i in range(0, len(train), 8):
            embs.append(model(to_batch(torch, train[i:i + 8], device)))
        if store is not None:
            model.subsample_embedding(sampling_ratio=ratio)
        else:
            model.subsample_embedding(torch.cat(embs), ratio)
        model.eval()
        for i in range(0, len(test), 8):
            r = model(to_batch(torch, test[i:i + 8], device))
            scores = (r.pred_score if hasattr(r, "pred_score") else r["pred_score"]).flatten().float().cpu().numpy()
            maps = (r.anomaly_map if hasattr(r, "anomaly_map") else r["anomaly_map"]).float().cpu().numpy()
            out.extend(zip(scores.tolist(), [m.squeeze() for m in maps]))
    return out


def sheet(items, path, thr):
    """items: (bgr, score, is_defect) sorted by score; red frame = defect, dashed = defect under the threshold."""
    th, cols, tiles = 150, 8, []
    for img, score, bad in items:
        t = cv2.resize(img, (th, th))
        if bad:
            cv2.rectangle(t, (0, 0), (th - 1, th - 1), (0, 0, 255), 5 if score > thr else 2)
        txt = f"{score:.2f}" + (" DEF" if bad else "")
        cv2.putText(t, txt, (4, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, txt, (4, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])


split = json.load(open(a.split))
os.makedirs(a.out, exist_ok=True)
model = torch = None
results = {}
for cam, s in sorted(split.items()):
    train = [i for i in (cv2.imread(x) for x in s["train"]) if i is not None]
    goods = [i for i in (cv2.imread(x) for x in s["test"]) if i is not None]
    defects = [i for i in (cv2.imread(x) for x in s["defect"]) if i is not None]
    print(f"=== {cam} : {len(train)} train, {len(goods)} bonnes de test, {len(defects)} défectueuses")
    if len(train) < 5 or len(goods) < 3:
        print("  pas assez de photos : il faut ≥ 5 bonnes en train et ≥ 3 en test (~20 % des bonnes vont en test)")
        continue
    source = "réels"
    if not defects:
        rng = np.random.default_rng(0)
        defects = [synth_defect(g, rng) for g in goods for _ in range(2)]
        source = "synthétiques (aucune photo défectueuse)"
    if model is None:
        model, torch = load_model(a.device)
    outs = fit_maps(model, torch, train, goods + defects, a.device)
    terms = [map_terms(m, FRACTION, raw) for raw, m in outs]
    ref = term_references([t[0] for t in terms[:len(goods)]], [t[1] for t in terms[:len(goods)]])
    scores = np.array([combined(pk, mn, ref, WEIGHT) for pk, mn in terms])
    g, b = scores[:len(goods)], scores[len(goods):]
    peak = np.array([t[0] for t in terms]); mean = np.array([t[1] for t in terms])
    parts = {"pic (score brut)": peak, "moyenne": mean, "combiné": scores}
    thr = float(g.max())  # strictest threshold that rejects no held-out good brick
    recall = float((b > thr).mean())
    # Threshold leaving ~10 % of the good bricks in the gray zone: the recall a recapture policy can count on.
    thr90 = float(np.percentile(g, 90))
    r = dict(n_train=len(train), n_test_good=len(goods), n_defect=len(defects), defects=source,
             auroc=round(auroc(g, b), 3), threshold=round(thr, 3), recall=round(recall, 3),
             recall_at_p90=round(float((b > thr90).mean()), 3),
             good_scores=[round(float(x), 3) for x in sorted(g)], defect_scores=[round(float(x), 3) for x in sorted(b)])
    results[cam] = r
    print(f"  défauts {source} — AUROC combiné {r['auroc']:.3f} (" + ", ".join(
        f"{k} {auroc(v[:len(goods)], v[len(goods):]):.3f}" for k, v in parts.items() if k != "combiné") + ")")
    print(f"  seuil = pire bonne de test ({thr:.2f}) : {recall:.0%} des défauts détectés, 0 bonne rejetée")
    print(f"  seuil au 90e percentile des bonnes ({thr90:.2f}) : {r['recall_at_p90']:.0%} détectés, ~10 % de bonnes rejetées")
    print(f"  scores bonnes {g.min():.2f}…{g.max():.2f} · défauts {b.min():.2f}…{b.max():.2f}")
    if r["auroc"] < 0.9:
        print("  → séparation faible : vérifiez le cadrage (cadre vert), la lumière, ou que le défaut se voit sous cet angle")
    items = sorted([(im, float(sc), False) for im, sc in zip(goods, g)] + [(im, float(sc), True) for im, sc in zip(defects, b)],
                   key=lambda t: t[1])
    sheet(items, os.path.join(a.out, f"{cam}_test.jpg"), thr)
json.dump(results, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
print(f"\nplanche : {a.out}/<caméra>_test.jpg (triée par score, défauts encadrés en rouge, ratés en trait fin)")
