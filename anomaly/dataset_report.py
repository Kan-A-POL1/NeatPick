#!/usr/bin/env python
"""Score an Anomalib training set (good bricks only) and say what to shoot next.

Usage:
  dataset_report.py GOOD_DIR [--defects DIR] [--out REPORT_DIR] [--colors N] [--no-model] [--device cpu|xpu]

GOOD_DIR holds the photos of good bricks, flat or one sub-folder per camera (front/, overhead/, wrist/: each
sub-folder is scored on its own, Anomalib needs one model per viewpoint). --defects is optional: photos of real
defective bricks, same layout. Without it, defects are synthesised (marker strokes, dots, tape) on held-out good
photos, so the score still means "can a model trained on these photos catch a marker stroke".

What it measures, per camera:
  photo hygiene    blur (Laplacian variance), under/over-exposure; near-duplicates (96x72 thumbnail diff) are counted
                   as a variety score, not rejected: a well-framed inspection set is meant to look alike
  framing          where the brick is and how big it is (largest saturated-colour blob): Anomalib wants the brick
                   at the same place and scale in every photo; a spread above ~5 % of the frame hurts
  colour coverage  the colour tag written by capture_app.py in the file name (<colour>__HHMMSS.jpg), else the hue of
                   the largest saturated blob in the central 70 % of the frame: every brick colour you will see at test
                   time must be in the good set, otherwise it is flagged as an anomaly
  model readiness  PatchCore (Anomalib's torch model) trained on growing subsets of the good photos; for each size:
                   AUROC good-vs-defect, recall at a threshold set from held-out good photos only (99th
                   percentile, i.e. ~1 % false rejects), false-reject rate. The learning curve tells you whether
                   more photos still help (keep shooting) or whether the model has plateaued (fix framing/lighting
                   or the defects are too subtle for this view).
Outputs: a table on stdout, REPORT_DIR/report.json, REPORT_DIR/<cam>_sheet.jpg (contact sheet, flagged photos framed
in red, label = reason), REPORT_DIR/<cam>_scores.jpg (highest-scoring held-out good photos = what the model finds odd).
"""
import argparse, glob, json, os, random, sys

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SIZE = 256  # model input side


CROPS = {}  # camera -> (x, y, w, h) as frame fractions, from --crop


def read(path, cam=None):
    img = cv2.imread(path)
    box = CROPS.get(cam) or CROPS.get("*")
    if img is not None and box:
        h, w = img.shape[:2]; x, y, bw, bh = box
        img = img[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]
    return img


def list_images(d):
    return sorted(f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if f.lower().endswith(IMG_EXT))


def camera_groups(root):
    subs = [s for s in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, s)) and list_images(os.path.join(root, s))]
    if subs and not [f for f in os.listdir(root) if f.lower().endswith(IMG_EXT)]:
        return {s: list_images(os.path.join(root, s)) for s in subs}
    return {os.path.basename(os.path.normpath(root)) or "all": list_images(root)}


# ---------------------------------------------------------------- photo hygiene
def thumb(gray):
    return cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA).astype(np.float32)


def brick_blob(bgr):
    """Largest saturated-colour blob (Duplo plastic vs white cloth / black arm). Returns (cx, cy, area, hue) in
    frame fractions, or None. Heuristic: the red gripper jaws or a coloured bowl can win, read it as a hint."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 90) & (hsv[..., 2] > 60)).astype(np.uint8)
    h0, w0 = mask.shape  # ignore the outer 15 % border: bowls and gripper jaws sit there more often than the brick
    border = np.ones_like(mask); border[int(.15 * h0):int(.85 * h0), int(.15 * w0):int(.85 * w0)] = 0
    if mask[border == 0].sum() > 0.002 * h0 * w0:
        mask[border == 1] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    h, w = mask.shape
    area = stats[i, cv2.CC_STAT_AREA] / (h * w)
    if area < 0.002:
        return None
    hue = float(np.median(hsv[..., 0][lab == i]))
    return cent[i][0] / w, cent[i][1] / h, area, hue


TAGGED_COLOURS = ("bleu", "rouge", "jaune", "vert", "orange", "blanc", "autre")


def hue_name(h):  # OpenCV hue 0..180
    for lim, name in ((8, "rouge"), (22, "orange"), (35, "jaune"), (85, "vert"), (130, "bleu"), (160, "violet"), (181, "rouge")):
        if h < lim:
            return name
    return "?"


def hygiene(paths, cam=None):
    rows, hashes = [], []
    for p in paths:
        bgr = read(p, cam)
        if bgr is None:
            rows.append(dict(path=p, unreadable=True)); hashes.append(None); continue
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        r = dict(path=p, blur=float(cv2.Laplacian(g, cv2.CV_64F).var()), mean=float(g.mean()),
                 dark=float((g < 15).mean()), sat=float((g > 245).mean()), w=bgr.shape[1], h=bgr.shape[0])
        b = brick_blob(bgr)
        if b:
            r.update(cx=b[0], cy=b[1], area=b[2], hue=b[3])
        base = os.path.basename(p)
        if "__" in base and base.split("__", 1)[0] in TAGGED_COLOURS:
            r["colour"] = base.split("__", 1)[0]  # tagged by capture_app: trust it over the hue guess
        rows.append(r); hashes.append(thumb(g))
    blurs = np.array([r["blur"] for r in rows if "blur" in r]) if rows else np.array([])
    blur_lim = 0.35 * float(np.median(blurs)) if len(blurs) else 0
    for i, r in enumerate(rows):
        why = []
        if r.get("unreadable"):
            why.append("illisible")
        else:
            if r["blur"] < blur_lim: why.append("flou")
            if r["dark"] > 0.25 or r["mean"] < 50: why.append("sombre")
            if r["sat"] > 0.10: why.append("surexposé")
            if "area" not in r: why.append("brique introuvable")
            for j in range(i):
                # near-identical frame: tiny mean change AND no local change (a small brick moving is a change)
                if hashes[j] is not None and hashes[i] is not None and \
                        np.abs(hashes[i] - hashes[j]).mean() < 1.5 and np.abs(hashes[i] - hashes[j]).max() < 25:
                    r["dup_of"] = os.path.basename(rows[j]["path"]); break
        r["flags"] = why
    return rows


# ---------------------------------------------------------------- synthetic defects
def synth_defect(bgr, rng, blob=None):
    img = bgr.copy(); h, w = img.shape[:2]
    if blob:
        cx, cy, s = blob[0] * w, blob[1] * h, max(12, np.sqrt(blob[2] * w * h) / 2)
    else:
        cx, cy, s = w / 2, h / 2, min(w, h) / 6
    pt = lambda: (int(cx + rng.uniform(-s, s)), int(cy + rng.uniform(-s, s)))
    kind = rng.choice(["stroke", "dots", "tape"])
    col = tuple(int(c) for c in rng.choice([(20, 20, 20), (30, 30, 160), (160, 60, 20), (240, 240, 240)]))
    if kind == "stroke":
        pts = np.array([pt() for _ in range(rng.integers(2, 5))], np.int32)
        cv2.polylines(img, [pts], False, col, int(rng.integers(2, 5)), cv2.LINE_AA)
    elif kind == "dots":
        for _ in range(rng.integers(3, 7)):
            cv2.circle(img, pt(), int(rng.integers(2, 4)), col, -1, cv2.LINE_AA)
    else:
        a, b = pt(), pt()
        cv2.rectangle(img, a, (a[0] + int(s * 0.6), a[1] + int(s * 0.35)), col, -1)
    return img


# ---------------------------------------------------------------- PatchCore (Anomalib torch model)
def load_model(device):
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    m = PatchcoreModel(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True)
    return m.to(device), torch


def to_batch(torch, imgs, device):
    x = np.stack([cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2RGB), (SIZE, SIZE)) for i in imgs]).astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32)).to(device)


def fit_score(model, torch, train, test, device, ratio=0.1):
    """Fit the PatchCore memory bank on `train` (BGR images), return image scores for `test`."""
    model.train()
    store = getattr(model, "embedding_store", None)  # anomalib >= 2.1 collects embeddings inside the model
    if store is not None:
        store.clear()
    embs = []
    with torch.no_grad():
        for i in range(0, len(train), 8):
            embs.append(model(to_batch(torch, train[i:i + 8], device)))
        if store is not None:
            model.subsample_embedding(sampling_ratio=ratio)
        else:  # anomalib 1.x / 2.0
            model.subsample_embedding(torch.cat(embs), ratio)
        model.eval(); scores = []
        for i in range(0, len(test), 8):
            out = model(to_batch(torch, test[i:i + 8], device))
            s = out.pred_score if hasattr(out, "pred_score") else out["pred_score"]
            scores.extend(s.flatten().float().cpu().numpy().tolist())
    return np.array(scores)


def auroc(neg, pos):
    allv = np.r_[neg, pos]; ranks = allv.argsort().argsort() + 1
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def readiness(goods, defects, device, cam=None, seed=0):
    rng = np.random.default_rng(seed)
    imgs = [read(r["path"], cam) for r in goods]
    idx = rng.permutation(len(imgs)); n_hold = max(5, len(imgs) // 4)
    hold, pool = idx[:n_hold], idx[n_hold:]
    if defects:
        bad = [read(p, cam) for p in defects]; bad_src = "réels"
    else:
        bad = [synth_defect(imgs[i], rng, (goods[i].get("cx") and (goods[i]["cx"], goods[i]["cy"], goods[i]["area"])))
               for i in hold for _ in range(2)]; bad_src = "synthétiques"
    model, torch = load_model(device)
    sizes = sorted({s for s in (10, 25, 50, 100, 200, 400) if s < len(pool)} | {len(pool)})
    curve = []
    for n in sizes:
        train = [imgs[i] for i in pool[:n]]
        sc = fit_score(model, torch, train, [imgs[i] for i in hold] + bad, device)
        g, b = sc[:len(hold)], sc[len(hold):]
        thr = float(np.percentile(g, 99))
        curve.append(dict(n_train=n, auroc=round(auroc(g, b), 3), recall_at_thr=round(float((b > thr).mean()), 3),
                          threshold=round(thr, 3)))
    hold_scores = list(zip([goods[i]["path"] for i in hold], g.tolist()))
    return dict(defects=bad_src, n_holdout=len(hold), n_defects=len(bad), curve=curve), hold_scores


# ---------------------------------------------------------------- report
def light(v, green, orange, higher_is_better=True):
    if v is None:
        return "?"
    ok = (v >= green) if higher_is_better else (v <= green)
    mid = (v >= orange) if higher_is_better else (v <= orange)
    return "VERT" if ok else ("ORANGE" if mid else "ROUGE")


def contact_sheet(rows, path, label=lambda r: ", ".join(r["flags"]), red=lambda r: bool(r["flags"]), cam=None):
    th, cols = 160, 8
    tiles = []
    for r in rows:
        im = read(r["path"], cam)
        if im is None:
            continue
        im = cv2.resize(im, (th * im.shape[1] // im.shape[0], th))
        im = cv2.resize(im, (th * 4 // 3, th))
        if red(r):
            cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), (0, 0, 255), 4)
        txt = (os.path.basename(r["path"])[:22] + " " + label(r))[:40]
        cv2.putText(im, txt, (4, th - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, txt, (4, th - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("good"); ap.add_argument("--defects"); ap.add_argument("--out", default="dataset-report")
    ap.add_argument("--colors", type=int, default=0, help="number of brick colours you expect at test time")
    ap.add_argument("--no-model", action="store_true"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--crop", action="append", default=[], metavar="[CAM=]X,Y,W,H",
                    help="crop every photo (of CAM, or all) to this box, in frame fractions, e.g. wrist=0.2,0.5,0.6,0.5; "
                         "try it to see whether framing the brick tighter raises the detection score")
    a = ap.parse_args()
    for c in a.crop:
        cam, _, box = c.rpartition("=")
        CROPS[cam or "*"] = tuple(float(v) for v in box.split(","))
    os.makedirs(a.out, exist_ok=True)
    defect_groups = camera_groups(a.defects) if a.defects else {}
    report = {}
    for cam, paths in camera_groups(a.good).items():
        rows = hygiene(paths, cam)
        ok = [r for r in rows if not r["flags"]]
        placed = [r for r in rows if "cx" in r]
        spread = float(np.hypot(np.std([r["cx"] for r in placed]), np.std([r["cy"] for r in placed]))) if len(placed) > 1 else None
        scale = float(np.std([np.sqrt(r["area"]) for r in placed]) / np.mean([np.sqrt(r["area"]) for r in placed])) if len(placed) > 1 else None
        colours = {}
        for r in rows:  # tagged colour first, else the hue of the central blob
            c = r.get("colour") or ("hue" in r and hue_name(r["hue"])) or None
            if c:
                colours[c] = colours.get(c, 0) + 1
        n_dup = sum(1 for r in rows if "dup_of" in r)
        res = dict(n_photos=len(rows), n_usable=len(ok), n_near_duplicates=n_dup, flagged={os.path.basename(r["path"]): r["flags"] for r in rows if r["flags"]},
                   position_spread=spread and round(spread, 3), scale_spread=scale and round(scale, 3), colours=colours)
        res["lights"] = dict(
            quantite=light(len(ok), 150, 60),
            proprete=light(len(ok) / max(1, len(rows)), 0.9, 0.75),
            variete=light(1 - n_dup / max(1, len(rows)), 0.6, 0.35),
            cadrage_position=light(spread, 0.05, 0.10, higher_is_better=False),
            cadrage_taille=light(scale, 0.10, 0.20, higher_is_better=False),
            couleurs=light(len(colours), a.colors, a.colors - 1) if a.colors else "?")
        contact_sheet(rows, os.path.join(a.out, f"{cam}_sheet.jpg"), cam=cam)
        if not a.no_model and len(ok) >= 12:
            try:
                res["model"], hold_scores = readiness(ok, defect_groups.get(cam) or (list_images(a.defects) if a.defects and len(defect_groups) == 1 else None), a.device, cam)
                c = res["model"]["curve"]
                res["lights"]["detection"] = light(c[-1]["auroc"], 0.95, 0.85)
                gain = c[-1]["auroc"] - c[-2]["auroc"] if len(c) > 1 else None
                if c[-1]["auroc"] >= 0.98 and c[-1]["recall_at_thr"] >= 0.95:
                    verdict = "prêt pour cette vue : passez à des défauts réels plus fins (--defects) et à d'autres couleurs"
                elif gain is not None and gain > 0.01:
                    verdict = "continuez à photographier : le score monte encore avec plus de photos"
                else:
                    verdict = "plateau : plus de photos n'aide plus, travaillez cadrage/lumière, ou la vue ne montre pas le défaut"
                res["model"]["verdict"] = verdict
                top = sorted(hold_scores, key=lambda t: -t[1])[:8]
                contact_sheet([dict(path=p, flags=[], score=s) for p, s in top], os.path.join(a.out, f"{cam}_scores.jpg"),
                              label=lambda r: f"score {r['score']:.2f}", red=lambda r: False, cam=cam)
            except Exception as e:  # report still useful without the model part
                res["model"] = dict(error=f"{type(e).__name__}: {e}")
        report[cam] = res
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2, ensure_ascii=False)

    for cam, r in report.items():
        print(f"\n=== {cam} : {r['n_usable']}/{r['n_photos']} photos utilisables")
        for k, v in r["lights"].items():
            print(f"  {k:<17} {v}")
        print(f"  position (écart-type, fraction du cadre) {r['position_spread']}  taille (écart relatif) {r['scale_spread']}")
        print(f"  couleurs vues : {r['colours']}   quasi-doublons : {r['n_near_duplicates']} (même vue répétée : rien à jeter, mais pas d'info nouvelle)")
        for name, why in list(r["flagged"].items())[:12]:
            print(f"  ! {name}: {', '.join(why)}")
        m = r.get("model")
        if m and "curve" in m:
            print(f"  PatchCore, défauts {m['defects']} ({m['n_defects']}), {m['n_holdout']} bonnes photos de contrôle :")
            for c in m["curve"]:
                print(f"    {c['n_train']:>4} photos -> AUROC {c['auroc']:.3f}  rappel au seuil 99 % {c['recall_at_thr']:.2f}")
            print(f"  => {m['verdict']}")
        elif m:
            print(f"  modèle : {m['error']}")
    print(f"\nrapport : {a.out}/report.json, planches : {a.out}/*_sheet.jpg, {a.out}/*_scores.jpg")


if __name__ == "__main__":
    sys.exit(main())
