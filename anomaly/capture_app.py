#!/usr/bin/env python
"""Photo capture + curation for Anomalib, on 127.0.0.1:8105 (stdlib HTTP + OpenCV, no other dependency).

Shoot: tap Space (or click the button) = 1 photo, hold it = burst at the chosen rate. Each photo is tagged with its
label (good / defect), its camera and the brick colour you selected.

Crop = SortOps' crop. Every photo is cut by ``sortops.pipeline.brick_crop.crop_brick`` (imported from the SortOps
backend, --sortops-src) with SortOps' ``inspection`` settings (live from its API, else its data/config.json): the
brick is found in the search zone, cut square and resized exactly as SortOps does before scoring. Studio cameras get
Studio's own preprocessing first (``sortops.hw.cameras.StudioFramePrep``: resize, undistort, Studio crop), as in
SortOps. The rectangle drawn on the preview is the search zone; for the inspection camera it IS ``inspection.roi`` of
SortOps. A frame without a brick is not saved. The full frame is kept in <session>/.raw/ so photos can be re-cut
after a crop setting changes (button "Recadrer à nouveau"; an older full-frame photo
becomes its own full frame on the first re-cut).

Split: each good photo is train or test, from a hash of its file name (stable, ~20 % test); defects are test.
Evaluate: quick_eval.py fits PatchCore on train, scores test goods + defects (synthetic ones if none).
Send: train goods (normal) and defects (is_anomalous, kept out of training) to an Anomalib Studio project, each once per
project (a manifest remembers it). Test goods never leave: they are what the evaluation measures.
Analyse: run dataset_report.py on the session.

Layout: ~/hackathon/captures/<session>/<good|defect>/<camera>/<colour>__HHMMSS_mmm.jpg   (trash: <session>/.trash/,
full frames: <session>/.raw/<good|defect>/<camera>/<same name>)

Cameras configured in Physical AI Studio (overhead, front, wrist) are read through the Studio's shared-camera
transport (iceoryx2 publishers): capture works while the Studio previews/records. Other cameras are opened directly.
Run with the Studio venv:  ~/hackathon/physical-ai-studio/application/backend/.venv/bin/python capture_app.py
"""
import argparse, glob, hashlib, json, os, re, shutil, socket, subprocess, sys, threading, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # webcams still work
    rs = None

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8105)
ap.add_argument("--root", default=os.path.expanduser("~/hackathon/captures"))
ap.add_argument("--studio", default="http://127.0.0.1:8100", help="Anomalib Studio base URL")
ap.add_argument("--report-python", default=os.path.expanduser("~/hackathon/anomalib/application/backend/.venv/bin/python"))
ap.add_argument("--report-script", default=os.path.join(HERE, "dataset_report.py"))
ap.add_argument("--physicalai", default="http://127.0.0.1:7860", help="Physical AI Studio backend (camera list)")
ap.add_argument("--studio-src", default=os.path.expanduser("~/hackathon/physical-ai-studio/application/backend/src"))
ap.add_argument("--sortops-src", default=os.path.join(HERE, os.pardir, "sortops", "backend"),
                help="SortOps backend: its brick crop and settings are used, never re-implemented here")
ap.add_argument("--sortops-data", default=os.path.join(HERE, os.pardir, "sortops", "data"))
ap.add_argument("--sortops", default="http://127.0.0.1:8200", help="SortOps API (live settings), file fallback")
ap.add_argument("--eval-script", default=os.path.join(HERE, "quick_eval.py"))
ap.add_argument("--test-pct", type=int, default=20, help="share of good photos held out for the evaluation")
A = ap.parse_args()
sys.path.insert(0, A.sortops_src)
from sortops.config import Settings  # noqa: E402  (SortOps backend, above)
from sortops.pipeline.brick_crop import CROP_DEFAULTS, crop_brick  # noqa: E402
ROOT = os.path.realpath(A.root)
IDLE_RELEASE = 20.0      # s without preview client nor shot before the camera is closed
HOLD_TIMEOUT = 1.2       # s without keep-alive before a burst stops (button released, tab closed, network drop)
KNOWN = {"243722063473": "overhead", "243622061121": "front", "HB20250300001": "wrist"}
LABELS = ("good", "defect")
COLOURS = ("bleu", "rouge", "jaune", "vert", "orange", "blanc", "autre")
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
SPLITS = ("train", "test")
LOCK = threading.RLock()


def safe(s, default=""):
    s = SAFE.sub("-", (s or "").strip())[:60].strip("-.")
    return s or default


def rel(path):
    return os.path.relpath(path, ROOT)


def inside(relpath):
    """Absolute path of a photo given relative to ROOT, or None if it escapes ROOT."""
    p = os.path.realpath(os.path.join(ROOT, relpath))
    return p if p.startswith(ROOT + os.sep) else None


# ------------------------------------------------------------------ settings: search zones of the non-inspection cameras
SETTINGS_PATH = os.path.join(ROOT, ".settings.json")


def load_settings():
    try:
        s = json.load(open(SETTINGS_PATH))
    except Exception:
        s = {}
    s.pop("crops", None)  # old manual crop boxes: replaced by SortOps' brick crop
    s.setdefault("zones", {})
    return s


SETTINGS = load_settings()


def save_settings():
    os.makedirs(ROOT, exist_ok=True)
    json.dump(SETTINGS, open(SETTINGS_PATH, "w"), indent=1)


# ------------------------------------------------------------------ SortOps inspection settings (the crop's parameters)
_INSPECTION = dict(t=0.0, cfg=None, source="")


def inspection():
    """SortOps' ``inspection`` section: live from its API when it runs (2 s cache), else its config.json + defaults."""
    if time.time() - _INSPECTION["t"] < 2 and _INSPECTION["cfg"]:
        return _INSPECTION["cfg"]
    # Defaults + config.json first, then the live values: a SortOps started before an upgrade lacks the newer keys.
    cfg = {**CROP_DEFAULTS, **Settings(__import__("pathlib").Path(A.sortops_data)).section("inspection")}
    src = "SortOps (config.json)"
    try:
        with urllib.request.urlopen(A.sortops + "/api/config", timeout=1) as r:
            cfg.update(json.loads(r.read())["inspection"]); src = "SortOps (API)"
    except Exception:
        pass
    _INSPECTION.update(t=time.time(), cfg=cfg, source=src)
    return cfg


def crop_cfg(cam):
    """Crop settings for ``cam``: SortOps' own for its inspection camera, its parameters + this app's zone otherwise."""
    cfg = dict(inspection())
    if cam != cfg["camera"]:
        cfg["roi"] = SETTINGS["zones"].get(cam) or [0.0, 0.0, 1.0, 1.0]
    return cfg


def set_zone(cam, box):
    """Search zone drawn on the preview. Inspection camera: written to SortOps (API, else its config.json)."""
    box = box or [0.0, 0.0, 1.0, 1.0]
    if cam == inspection()["camera"]:
        patch = json.dumps({"inspection": {"roi": box}}).encode()
        try:
            req = urllib.request.Request(A.sortops + "/api/config", data=patch, method="PATCH",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3).read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"SortOps refuse la zone : {e.read().decode(errors='replace')[:200]}")
        except Exception:
            Settings(__import__("pathlib").Path(A.sortops_data)).update({"inspection": {"roi": box}})
        _INSPECTION["t"] = 0.0
    elif box == [0.0, 0.0, 1.0, 1.0]:
        SETTINGS["zones"].pop(cam, None); save_settings()
    else:
        SETTINGS["zones"][cam] = box; save_settings()


def zones():
    out = dict(SETTINGS["zones"])
    out[inspection()["camera"]] = inspection()["roi"]
    return out


# ------------------------------------------------------------------ cameras
STUDIO_CAMS = {}  # "studio:<camera id>" -> camera row from the Physical AI Studio API


def studio_cams():
    out = []
    try:
        for proj in json.load(urllib.request.urlopen(A.physicalai + "/api/projects", timeout=3)):
            for row in json.load(urllib.request.urlopen(f"{A.physicalai}/api/projects/{proj['id']}/cameras", timeout=3)):
                cid = f"studio:{row['id']}"
                STUDIO_CAMS[cid] = row
                out.append(dict(id=cid, name=safe(row["name"], "cam"), label=f"{row['name']} — partagée avec le Studio",
                                serial=(row.get("fingerprint") or {}).get("serial")))
    except Exception:
        pass
    return out


def list_cams():
    cams = studio_cams()
    shared = {c.pop("serial") for c in cams if c.get("serial")}
    if rs is not None:
        try:
            for d in rs.context().query_devices():
                sn = d.get_info(rs.camera_info.serial_number)
                if sn not in shared:
                    n = KNOWN.get(sn, "realsense")
                    cams.append(dict(id=f"rs:{sn}", name=n, label=f"{n} — RealSense {sn} (directe)"))
        except Exception:
            pass
    for p in sorted(glob.glob("/dev/v4l/by-id/*-video-index0")):
        if "RealSense" in p or any(sn and sn in p for sn in shared):
            continue
        base = os.path.basename(p).replace("-video-index0", "")
        n = "laptop" if "HP_5MP" in base else next((v for k, v in KNOWN.items() if k in base), "webcam")
        cams.append(dict(id=f"v4l:{p}", name=n, label=f"{n} — {base[4:36]} (directe)"))
    return cams


class Camera:
    """One open camera, a reader thread keeping the latest frame (BGR)."""

    def __init__(self, cam_id):
        self.id, self.frame, self.t, self.err, self.alive = cam_id, None, 0.0, None, True
        self.last_use = time.time()
        self.cap = self.sc = self.pipe = self.prep = None
        if cam_id.startswith("studio:"):
            sys.path.insert(0, A.studio_src)
            from pydantic import TypeAdapter
            from schemas.project_camera import Camera as CameraSchema
            from utils.camera_factory import build_shared_camera
            row = STUDIO_CAMS.get(cam_id) or (studio_cams() and STUDIO_CAMS.get(cam_id))
            if not row:
                raise RuntimeError("caméra inconnue du Physical AI Studio (backend :7860 arrêté ?)")
            from sortops.hw.cameras import StudioFramePrep  # Studio's resize + undistort + crop, as in SortOps
            self.prep = StudioFramePrep(row)
            self.sc = build_shared_camera(TypeAdapter(CameraSchema).validate_python(row), idle_timeout=5.0)
            try:
                self.sc.connect()
            except Exception as e:
                raise RuntimeError(f"le Studio ne fournit pas cette caméra ({e}). Un autre programme la tient-il en direct "
                                   "(so101vision) ?")
        elif cam_id.startswith("rs:"):
            self.pipe, cfg = rs.pipeline(), rs.config()
            cfg.enable_device(cam_id[3:])
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipe.start(cfg)
        else:
            self.cap = cv2.VideoCapture(cam_id[4:], cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                raise RuntimeError("impossible d'ouvrir la caméra : déjà utilisée par un autre programme ?")
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while self.frame is None and self.err is None and time.time() - t0 < 6:
            time.sleep(0.05)
        if self.frame is None:
            self.close(); raise RuntimeError(self.err or "aucune image reçue de la caméra")

    def _grab(self):
        if self.sc is not None:
            f = self.sc.read_latest()
            time.sleep(1 / 30)  # read_latest does not block
            return cv2.cvtColor(self.prep(f.data), cv2.COLOR_RGB2BGR)
        if self.pipe is not None:
            f = self.pipe.wait_for_frames(2000).get_color_frame()
            return np.asanyarray(f.get_data()).copy() if f else None
        ok, img = self.cap.read()
        return img if ok else None

    def _loop(self):
        n = 0
        while self.alive:
            try:
                img = self._grab()
                if img is not None:
                    n += 1
                    if n > 3 or self.sc is not None:  # direct sensors give dark first frames
                        self.frame, self.t = img, time.time()
            except Exception as e:
                if self.sc is None or self.frame is not None:
                    self.err = str(e)
                time.sleep(0.05)

    def close(self):
        self.alive = False
        try:
            if self.sc is not None:
                self.sc.disconnect()
            elif self.cap is not None:
                self.cap.release()
            elif self.pipe is not None:
                self.pipe.stop()
        except Exception:
            pass


CAM = None          # current Camera
CAM_NAME = "cam"    # folder name of the current camera
BURST = None        # dict(session,label,colour,fps,until,count)
PUSH = dict(running=False, done=0, total=0, errors=0, msg="")
REPORT = dict(running=False, text="")
LAST_SHOT = dict(path=None, at=0.0, n=0, miss=0)
EVAL = dict(running=False, text="", sheet=None)


def open_cam(cam_id):
    global CAM, CAM_NAME
    with LOCK:
        if CAM and CAM.id == cam_id and CAM.alive:
            CAM.last_use = time.time(); return CAM
        if CAM:
            CAM.close(); CAM = None
        CAM = Camera(cam_id)
        CAM_NAME = next((c["name"] for c in list_cams() if c["id"] == cam_id), "cam")
        return CAM


def janitor():
    global CAM, BURST
    while True:
        time.sleep(1)
        with LOCK:
            if BURST and time.time() > BURST["until"]:
                BURST = None
            if CAM and not BURST and time.time() - CAM.last_use > IDLE_RELEASE:
                CAM.close(); CAM = None


def brick(bgr, cam):
    """SortOps' brick crop of a BGR frame (BGR in, BGR crop out)."""
    c = crop_brick(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), crop_cfg(cam))
    return c, cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR)


def raw_path(path):
    """<session>/<label>/<cam>/<name> -> <session>/.raw/<label>/<cam>/<name> (the full frame of a photo)."""
    parts = rel(path).split(os.sep)
    return os.path.join(ROOT, parts[0], ".raw", *parts[1:])


def save_shot(session, label, colour):
    """Save the current frame cut by SortOps' crop (+ the full frame in .raw/); returns its relative path."""
    cam = CAM
    if not cam or cam.frame is None:
        raise RuntimeError("aucune caméra ouverte")
    cam.last_use = time.time()
    frame = cam.frame
    c, img = brick(frame, CAM_NAME)
    if not c.found:
        LAST_SHOT.update(miss=LAST_SHOT.get("miss", 0) + 1)
        raise RuntimeError("pas de brique dans la zone de recherche : photo non prise")
    d = os.path.join(ROOT, session, label, CAM_NAME)
    os.makedirs(d, exist_ok=True)
    now = time.time()
    name = f"{colour}__{time.strftime('%H%M%S', time.localtime(now))}_{int(now * 1000) % 1000:03d}.jpg"
    path = os.path.join(d, name)
    os.makedirs(os.path.dirname(raw_path(path)), exist_ok=True)
    cv2.imwrite(raw_path(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    LAST_SHOT.update(path=rel(path), at=now, n=LAST_SHOT["n"] + 1)
    return rel(path)


def recrop(session):
    """Re-cut every photo of the session from its full frame with the current crop settings."""
    done = missing = lost = 0
    for path in glob.glob(os.path.join(ROOT, session, "*", "*", "*.jpg")):
        parts = rel(path).split(os.sep)
        if parts[1] not in LABELS:
            continue
        frame = cv2.imread(raw_path(path))
        adopt = False
        if frame is None:
            # Photo taken before the shared crop: it is its own full frame if it is not already a square crop.
            frame = cv2.imread(path)
            if frame is None or frame.shape[0] == frame.shape[1]:
                missing += 1; continue
            adopt = True
        c, img = brick(frame, parts[2])
        if not c.found:
            lost += 1; continue
        if adopt:
            os.makedirs(os.path.dirname(raw_path(path)), exist_ok=True)
            shutil.copy2(path, raw_path(path))
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95]); done += 1
    return dict(done=done, missing=missing, lost=lost)


def burst_loop():
    next_t = 0.0
    while True:
        b = BURST
        if not b or not CAM or CAM.frame is None:
            time.sleep(0.02); next_t = 0.0; continue
        now = time.time()
        if now < next_t:
            time.sleep(min(0.01, next_t - now)); continue
        next_t = max((next_t or now) + 1.0 / b["fps"], now)
        try:
            save_shot(b["session"], b["label"], b["colour"]); b["count"] += 1
        except Exception:
            time.sleep(0.1)


# ------------------------------------------------------------------ photos of a session
BLUR = {}  # path -> (mtime, sharpness)


def sharpness(path):
    m = os.path.getmtime(path)
    c = BLUR.get(path)
    if c and c[0] == m:
        return c[1]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    v = round(float(cv2.Laplacian(cv2.resize(img, (320, 240)), cv2.CV_64F).var()), 1) if img is not None else 0.0
    BLUR[path] = (m, v)
    return v


def split_of(label, name):
    """train / test, stable: a hash of the file name (~A.test_pct % of the good photos in test); defects are test."""
    if label != "good":
        return "test"
    return "test" if int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 100 < A.test_pct else "train"


def manifest(session):
    try:
        return json.load(open(os.path.join(ROOT, session, ".uploaded.json")))
    except Exception:
        return {}


def photos(session):
    sent = manifest(session)
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, session, "*", "*", "*.jpg"))):
        r = rel(path); parts = r.split(os.sep)
        if parts[1] not in LABELS:
            continue
        base = parts[3]
        colour = base.split("__", 1)[0] if "__" in base else "?"
        out.append(dict(path=r, label=parts[1], cam=parts[2], colour=colour, name=base, split=split_of(parts[1], base),
                        raw=os.path.isfile(raw_path(path)),
                        sharp=sharpness(path), sent=[p for p, lst in sent.items() if r in lst]))
    # sharpness relative to the median of the same camera: < 40 % = probably blurred
    for cam in {p["cam"] for p in out}:
        vals = sorted(p["sharp"] for p in out if p["cam"] == cam)
        med = vals[len(vals) // 2] or 1
        for p in out:
            if p["cam"] == cam:
                p["blurry"] = p["sharp"] < 0.4 * med
    return out


def sessions():
    out = []
    for s in sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []:
        if s.startswith(".") or not os.path.isdir(os.path.join(ROOT, s)):
            continue
        n = {lab: len(glob.glob(os.path.join(ROOT, s, lab, "*", "*.jpg"))) for lab in LABELS}
        out.append(dict(name=s, **n, mtime=os.path.getmtime(os.path.join(ROOT, s))))
    return sorted(out, key=lambda x: -x["mtime"])


TRASH_LOG = []  # list of batches [(trash_path, original_path), ...] for undo


def delete(paths):
    batch = []
    for r in paths:
        p = inside(r)
        if not p or not os.path.isfile(p):
            continue
        session = r.split(os.sep)[0]
        t = os.path.join(ROOT, session, ".trash", r.replace(os.sep, "~"))
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.move(p, t); batch.append((t, p))
        if os.path.isfile(raw_path(p)):
            shutil.move(raw_path(p), t + ".raw"); batch.append((t + ".raw", raw_path(p)))
    if batch:
        TRASH_LOG.append(batch)
    return len(batch)


def undo_delete():
    if not TRASH_LOG:
        return 0
    batch = TRASH_LOG.pop()
    for t, p in batch:
        if os.path.exists(t):
            os.makedirs(os.path.dirname(p), exist_ok=True); shutil.move(t, p)
    return len(batch)


def relabel(paths, label=None, colour=None):
    n = 0
    for r in paths:
        p = inside(r)
        if not p or not os.path.isfile(p):
            continue
        parts = r.split(os.sep)
        base = parts[3]
        stem = base.split("__", 1)[1] if "__" in base else base
        new_base = f"{colour}__{stem}" if colour else base
        new = os.path.join(ROOT, parts[0], label or parts[1], parts[2], new_base)
        if new != p:
            os.makedirs(os.path.dirname(new), exist_ok=True); shutil.move(p, new); n += 1
            if os.path.isfile(raw_path(p)):
                os.makedirs(os.path.dirname(raw_path(new)), exist_ok=True); shutil.move(raw_path(p), raw_path(new))
    return n


# ------------------------------------------------------------------ Anomalib Studio, report
def studio(path, data=None, headers=None, method=None, timeout=20):
    req = urllib.request.Request(A.studio + path, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def upload_one(project, path, anomalous=False):
    """POST one photo; ``anomalous`` needs Studio's is_anomalous form field (pc-side/anomalib-patches)."""
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"is_anomalous\"\r\n\r\n"
            f"{'true' if anomalous else 'false'}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    studio(f"/api/projects/{project}/images", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}, "POST", 60)


def push(session, project, paths):
    mpath = os.path.join(ROOT, session, ".uploaded.json")
    man = manifest(session)
    sent = set(man.get(project, []))
    rows = {p["path"]: p for p in photos(session)}
    if paths is None:
        paths = list(rows)
    # Test goods stay here: a model trained on them would make the evaluation meaningless.
    held = [r for r in paths if r in rows and rows[r]["label"] == "good" and rows[r]["split"] == "test"]
    paths = [r for r in paths if r in rows and r not in held]
    files = [r for r in paths if r not in sent and inside(r) and os.path.isfile(inside(r))]
    PUSH.update(running=True, done=0, total=len(files), errors=0, msg=f"envoi de {len(files)} photos…")
    lk = threading.Lock()

    def one(r):
        try:
            upload_one(project, inside(r), anomalous=rows[r]["label"] == "defect")
            with lk:
                sent.add(r); PUSH["done"] += 1
        except Exception as e:
            with lk:
                PUSH["errors"] += 1; PUSH["msg"] = f"erreur : {e}"

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, files))
    man[project] = sorted(sent)
    json.dump(man, open(mpath, "w"))
    skipped = len(paths) - len(files)
    PUSH.update(running=False, msg=f"{PUSH['done']} envoyée(s), {PUSH['errors']} erreur(s)"
                + (f", {skipped} déjà envoyée(s)" if skipped else "")
                + (f", {len(held)} bonne(s) de test gardée(s) ici" if held else ""))


def report(session, model):
    REPORT.update(running=True, text="analyse en cours…")
    good, defect = os.path.join(ROOT, session, "good"), os.path.join(ROOT, session, "defect")
    cmd = [A.report_python, A.report_script, good, "--out", os.path.join(ROOT, session, ".report"), "--device", "xpu"]
    if os.path.isdir(defect) and glob.glob(os.path.join(defect, "*", "*.jpg")):
        cmd += ["--defects", defect]
    if not model:
        cmd.append("--no-model")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        txt = "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                        if not re.search(r"warn|dinov2|^\s+from |subsample_embedding|^\s*$", l, re.I))
        REPORT["text"] = txt.strip() or "(pas de sortie)"
    except Exception as e:
        REPORT["text"] = f"échec : {e}"
    REPORT["running"] = False


def evaluate(session):
    """quick_eval.py (Anomalib venv): PatchCore fitted on the train goods, scored on test goods + defects, per camera."""
    EVAL.update(running=True, text="évaluation PatchCore en cours…", sheet=None)
    out = os.path.join(ROOT, session, ".eval")
    os.makedirs(out, exist_ok=True)
    split = {}
    for p in photos(session):
        key = "defect" if p["label"] == "defect" else p["split"]
        split.setdefault(p["cam"], {"train": [], "test": [], "defect": []})[key].append(inside(p["path"]))
    json.dump(split, open(os.path.join(out, "split.json"), "w"), indent=1)
    cmd = [A.report_python, A.eval_script, os.path.join(out, "split.json"), "--out", out, "--device", "xpu"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        txt = "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                        if not re.search(r"warn|dinov2|^\s+from |subsample_embedding|Coreset|^\s*$", l, re.I))
        EVAL["text"] = txt.strip() or "(pas de sortie)"
        sheets = sorted(glob.glob(os.path.join(out, "*_test.jpg")))
        EVAL["sheet"] = [rel(x) for x in sheets]
    except Exception as e:
        EVAL["text"] = f"échec : {e}"
    EVAL["running"] = False


# ------------------------------------------------------------------ HTTP
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", cache=False):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers(); self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/cams":
            return self.send(200, list_cams())
        if u.path == "/api/state":
            b = BURST
            return self.send(200, dict(cam=CAM and CAM.alive and CAM.id, cam_name=CAM_NAME,
                                       burst=b and dict(count=b["count"], fps=b["fps"]), last=LAST_SHOT, push=PUSH,
                                       report=REPORT, eval=EVAL, zones=zones(), crop_source=_INSPECTION["source"],
                                       inspection_cam=inspection()["camera"],
                                       crop={k: v for k, v in inspection().items() if k.startswith("crop_")}))
        if u.path == "/api/sessions":
            return self.send(200, sessions())
        if u.path == "/api/photos":
            return self.send(200, photos(safe(q.get("session"))))
        if u.path == "/api/projects":
            try:
                return self.send(200, studio("/api/projects")["projects"])
            except Exception as e:
                return self.send(502, dict(error=f"Anomalib Studio injoignable ({A.studio}) : {e}"))
        if u.path == "/stream.mjpg":
            try:
                cam = open_cam(q.get("cam", ""))
            except Exception as e:
                return self.send(409, dict(error=str(e)))
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
            try:
                while CAM is cam and cam.alive:
                    cam.last_use = time.time()
                    img = cam.frame.copy()
                    # What a photo would be: SortOps' crop box (green) or "no brick" (red), on the search zone.
                    try:
                        c, _ = brick(img, CAM_NAME)
                        h, w = img.shape[:2]; x, y, bw, bh = c.box
                        p0, p1 = (int(x * w), int(y * h)), (int((x + bw) * w) - 1, int((y + bh) * h) - 1)
                        if c.found:
                            cv2.rectangle(img, p0, p1, (60, 220, 60), 3)
                        else:
                            cv2.putText(img, "pas de brique", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 230), 3)
                    except Exception as e:  # noqa: BLE001 - the preview must keep streaming
                        cv2.putText(img, f"crop: {e}"[:60], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 230), 2)
                    flash = time.time() - LAST_SHOT["at"] < 0.15
                    if BURST or flash:
                        img = cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                                            (255, 255, 255) if flash else (0, 0, 255), 10)
                    # small preview: every request to this app shares the SSH tunnel with this stream, a heavy stream
                    # (640x480 @ 15 fps ~ 750 KB/s) starves them; 480x360 q60 @ 8 fps is ~120 KB/s. Photos stay full-res.
                    jpg = cv2.imencode(".jpg", cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA),
                                       [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                                     + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1 / 8)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if u.path.startswith("/photo/"):
            p = inside(unquote(u.path[7:]))
            if not p or not os.path.isfile(p):
                return self.send(404, dict(error="introuvable"))
            if "w" in q:
                img = cv2.imread(p); w = max(40, min(800, int(q["w"])))
                img = cv2.resize(img, (w, max(1, img.shape[0] * w // img.shape[1])), interpolation=cv2.INTER_AREA)
                return self.send(200, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), "image/jpeg", True)
            return self.send(200, open(p, "rb").read(), "image/jpeg", True)
        return self.send(404, dict(error="not found"))

    def do_POST(self):
        global BURST, CAM
        u = urlparse(self.path); b = self.body()
        session, label = safe(b.get("session"), "session"), b.get("label") if b.get("label") in LABELS else "good"
        colour = b.get("colour") if b.get("colour") in COLOURS else "autre"
        try:
            if u.path == "/api/shot":
                with LOCK:
                    return self.send(200, dict(path=save_shot(session, label, colour)))
            if u.path == "/api/hold":  # start or keep alive a burst; the page calls it every 300 ms while held
                with LOCK:
                    if not CAM:
                        return self.send(409, dict(error="ouvrez d'abord une caméra"))
                    if BURST is None:
                        BURST = dict(session=session, label=label, colour=colour, count=0,
                                     fps=max(0.5, min(15.0, float(b.get("fps") or 3))))
                    BURST["until"] = time.time() + HOLD_TIMEOUT
                    return self.send(200, dict(count=BURST["count"]))
            if u.path == "/api/release":
                with LOCK:
                    n = BURST["count"] if BURST else 0; BURST = None
                return self.send(200, dict(count=n))
            if u.path == "/api/close":
                with LOCK:
                    BURST = None
                    if CAM:
                        CAM.close(); CAM = None
                return self.send(200, dict(ok=True))
            if u.path == "/api/zone":
                cam = safe(b.get("cam"), "cam"); box = b.get("box")
                if box:
                    x, y, w, h = (max(0.0, min(1.0, float(v))) for v in box)
                    box = [round(x, 4), round(y, 4), round(min(w, 1 - x), 4), round(min(h, 1 - y), 4)]
                set_zone(cam, box)
                return self.send(200, zones())
            if u.path == "/api/recrop":
                return self.send(200, recrop(session))
            if u.path == "/api/eval":
                if EVAL["running"]:
                    return self.send(409, dict(error="évaluation déjà en cours"))
                threading.Thread(target=evaluate, args=(session,), daemon=True).start()
                return self.send(200, dict(ok=True))
            if u.path == "/api/delete":
                return self.send(200, dict(n=delete(b.get("paths") or [])))
            if u.path == "/api/undo":
                return self.send(200, dict(n=undo_delete()))
            if u.path == "/api/move":
                return self.send(200, dict(n=relabel(b.get("paths") or [], label=b.get("to") if b.get("to") in LABELS else None,
                                                     colour=b.get("colour") if b.get("colour") in COLOURS else None)))
            if u.path == "/api/free-so101vision":
                urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8092/close", data=b"", method="POST"), timeout=5)
                return self.send(200, dict(ok=True))
            if u.path == "/api/push":
                if PUSH["running"]:
                    return self.send(409, dict(error="envoi déjà en cours"))
                proj = b.get("project") or ""
                if not re.fullmatch(r"[0-9a-f-]{36}", proj):
                    return self.send(400, dict(error="choisissez un projet Anomalib Studio"))
                threading.Thread(target=push, args=(session, proj, b.get("paths")), daemon=True).start()
                return self.send(200, dict(ok=True))
            if u.path == "/api/project":
                return self.send(200, studio("/api/projects", json.dumps(dict(name=b.get("name") or "Briques")).encode(),
                                             {"Content-Type": "application/json"}, "POST"))
            if u.path == "/api/report":
                if REPORT["running"]:
                    return self.send(409, dict(error="analyse déjà en cours"))
                threading.Thread(target=report, args=(session, bool(b.get("model", True))), daemon=True).start()
                return self.send(200, dict(ok=True))
        except Exception as e:
            return self.send(500, dict(error=str(e)))
        return self.send(404, dict(error="not found"))


PAGE = r"""<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Capture Anomalib</title>
<style>
:root{--bg:#0f1115;--p:#181b22;--p2:#20242d;--b:#2c313c;--t:#e7e9ee;--m:#8b93a3;--a:#3b82f6;--g:#22c55e;--r:#ef4444;--y:#f59e0b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--t);font:14px system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;background:var(--p);border-bottom:1px solid var(--b);flex-wrap:wrap}
header h1{font-size:16px;margin:0 8px 0 0}select,input,button{font:inherit;color:var(--t);background:var(--p2);border:1px solid var(--b);border-radius:6px;padding:6px 9px}
button{cursor:pointer}button:hover{border-color:#4b5363}button.pri{background:var(--a);border-color:var(--a)}button.dan{color:#fca5a5}
button:disabled{opacity:.45;cursor:default}.chip{padding:3px 9px;border-radius:99px;font-size:12px;background:var(--p2);border:1px solid var(--b)}
.ok{color:var(--g)}.ko{color:var(--r)}.muted{color:var(--m)}main{display:grid;grid-template-columns:660px 1fr;gap:14px;padding:14px}
.card{background:var(--p);border:1px solid var(--b);border-radius:10px;padding:12px;margin-bottom:14px}.card h2{font-size:14px;margin:0 0 10px;color:var(--m);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
#stage{position:relative;width:640px;height:480px;background:#000;border-radius:8px;overflow:hidden;user-select:none}
#view{width:640px;height:480px;display:block}#view:not([src]){visibility:hidden}
#box{position:absolute;border:2px dashed var(--y);box-shadow:0 0 0 9999px rgba(0,0,0,.45);pointer-events:none;display:none}
#stagemsg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:30px;color:var(--m)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.seg{display:inline-flex;border:1px solid var(--b);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:8px 14px}.seg button.on{background:var(--a)}#lab-good.on{background:var(--g);color:#05290f}#lab-defect.on{background:var(--r)}
.dot{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:middle;border:1px solid #0006}
.colours button{padding:6px 10px}.colours button.on{outline:2px solid var(--t)}
#shoot{width:100%;height:76px;font-size:20px;font-weight:700;margin-top:10px;background:var(--g);border:0;color:#05290f;touch-action:none}
#shoot.burst{background:var(--r);color:#fff}kbd{background:rgba(255,255,255,.12);color:inherit;border:1px solid rgba(255,255,255,.25);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px}
#shoot kbd,#lab-good.on kbd{background:rgba(0,0,0,.15);border-color:rgba(0,0,0,.3)}
#gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;max-height:calc(100vh - 290px);overflow:auto;padding:2px}
.ph{position:relative;border-radius:8px;overflow:hidden;border:2px solid transparent;background:#000;cursor:pointer;aspect-ratio:4/3}
.ph img{width:100%;height:100%;object-fit:contain;display:block}.ph.sel{border-color:var(--a);box-shadow:0 0 0 2px var(--a)}
.ph .tags{position:absolute;left:4px;top:4px;display:flex;gap:4px}.ph .tag{font-size:11px;padding:1px 6px;border-radius:99px;background:#000b}
.ph .train{color:#93c5fd}.ph .test{color:#fcd34d}.ph .good{color:#86efac}.ph .defect{color:#fca5a5}.ph .warn{color:#fcd34d}.ph .sent{position:absolute;right:4px;top:4px;font-size:11px;background:#1d4ed8cc;padding:1px 6px;border-radius:99px}
.ph .nm{position:absolute;left:0;right:0;bottom:0;font-size:11px;padding:2px 5px;background:#000a;color:#ccc;white-space:nowrap;overflow:hidden}
#toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#111827;border:1px solid var(--b);padding:9px 16px;border-radius:8px;display:none;z-index:9}
#lb{position:fixed;inset:0;background:#000e;display:none;align-items:center;justify-content:center;flex-direction:column;z-index:8}#lb img{max-width:92vw;max-height:82vh}
pre{white-space:pre-wrap;font-size:12px;max-height:340px;overflow:auto;margin:0}.counts b{font-size:16px}
</style>
<header><h1>📷 Capture Anomalib</h1>
<label class="muted">Session</label><select id="session"></select><button id="newsess" title="Nouvelle session">＋</button>
<label class="muted">Caméra</label><select id="cam"></select><span id="camstate" class="chip muted">fermée</span>
<button id="closecam">Libérer la caméra</button><button id="freesv" title="so101vision garde parfois les caméras">Libérer so101vision</button>
<span style="flex:1"></span><label class="muted">Projet Anomalib Studio</label><select id="project"></select><button id="newproj" title="Créer un projet">＋</button>
</header>
<main><div>
<div class="card"><div id="stage"><img id="view" alt="" draggable="false"><div id="box"></div><div id="stagemsg">Choisissez une caméra.</div></div>
<div class="row" style="margin-top:8px"><span class="muted">Zone de recherche :</span><span id="cropinfo" class="muted">image entière</span>
<button id="cropclear">Image entière</button><button id="recrop" title="Recoupe toutes les photos de la session depuis leur image complète, avec les réglages actuels">Recadrer à nouveau</button></div>
<p class="muted" style="margin:6px 0 0">Cadre vert = la photo qui sera prise : recadrage <b>SortOps</b> (<span id="cropsrc">…</span>), identique à celui du tri.
Tracez un rectangle pour limiter la zone où la brique est cherchée<span id="inspnote"></span>.</p></div>
<div class="card"><h2>Prise de vue</h2>
<div class="row"><div class="seg"><button id="lab-good" class="on">Bonne <kbd>G</kbd></button><button id="lab-defect">Défectueuse <kbd>D</kbd></button></div>
<span class="muted">Rafale</span><input id="fps" type="number" min="0.5" max="15" step="0.5" value="3" style="width:64px"><span class="muted">img/s</span></div>
<div class="row colours" id="colours" style="margin-top:8px"></div>
<button id="shoot">Photo — <kbd>Espace</kbd> (maintenir = rafale)</button>
<p class="muted" style="margin:8px 0 0">Appui bref : 1 photo. Maintenu plus de 0,4 s : rafale tant que c'est maintenu. <kbd>Ctrl+Z</kbd> annule une suppression.</p></div>
<div class="card"><h2>Évaluation train → test</h2><div class="row"><button id="evalbtn" class="pri">Évaluer (PatchCore rapide)</button>
<span class="muted">entraîné sur les bonnes <b>train</b>, testé sur les bonnes <b>test</b> + les défectueuses</span></div>
<pre id="evaltxt" class="muted" style="margin-top:8px">Les bonnes photos sont réparties train / test une fois pour toutes (≈20 % test). Les défectueuses sont toujours en test.</pre>
<div id="evalsheet"></div></div>
<div class="card"><h2>Analyse de la session</h2><div class="row"><button id="repfast">Analyse rapide</button><button id="rep">Analyse complète (PatchCore)</button></div>
<pre id="report" class="muted" style="margin-top:8px">Quantité, netteté, cadrage, couleurs et courbe de détection : dit s'il faut plus de photos ou mieux cadrer.</pre></div>
</div>
<div><div class="card"><div class="row counts" style="justify-content:space-between"><div class="row">
<span>Bonnes <b id="nGood">0</b> <span class="muted">(<span id="nTrain">0</span> train · <span id="nTest">0</span> test)</span></span><span>Défectueuses <b id="nDef">0</b></span><span>Envoyées <b id="nSent">0</b></span><span id="sharpinfo" class="muted"></span></div>
<div class="row"><select id="fLabel"><option value="">toutes</option><option value="good">bonnes</option><option value="defect">défectueuses</option></select>
<select id="fSplit"><option value="">train + test</option><option value="train">train</option><option value="test">test</option></select>
<select id="fCam"><option value="">toutes caméras</option></select><select id="fColour"><option value="">toutes couleurs</option></select>
<label><input type="checkbox" id="fBlur"> floues</label><label><input type="checkbox" id="fNew"> non envoyées</label></div></div>
<div class="row" style="margin:10px 0"><span id="selinfo" class="muted">0 sélectionnée(s)</span><button id="selall">Tout <kbd>A</kbd></button><button id="selnone">Aucune <kbd>Échap</kbd></button>
<button id="toGood">→ Bonnes</button><button id="toDef">→ Défectueuses</button><select id="setColour"><option value="">couleur…</option></select>
<button id="del" class="dan">Supprimer <kbd>Suppr</kbd></button><button id="undo">Annuler</button>
<span style="flex:1"></span><button id="sendSel">Envoyer la sélection</button><button id="sendNew" class="pri" title="Bonnes de train (normales) + défectueuses (anomalies) pas encore envoyées. Les bonnes de test restent ici.">Envoyer vers Studio (train + défauts)</button></div>
<div id="pushmsg" class="muted" style="margin-bottom:8px"></div>
<div id="gal"></div></div></div></main>
<div id="lb"><img id="lbimg"><div id="lbtxt" class="muted" style="margin-top:8px"></div><div class="muted">← → naviguer · G/D étiqueter · Suppr supprimer · Échap fermer</div></div>
<div id="toast"></div>
<script>
const $=id=>document.getElementById(id);
const api=(u,b)=>fetch(u,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(async r=>{const j=await r.json().catch(()=>({}));if(!r.ok&&!j.error)j.error='HTTP '+r.status;return j});
const COLOURS={bleu:'#2563eb',rouge:'#dc2626',jaune:'#facc15',vert:'#16a34a',orange:'#f97316',blanc:'#f5f5f5',autre:'#9ca3af'};
let S={label:localStorage.label||'good',colour:localStorage.colour||'bleu',photos:[],sel:new Set(),last:null,cams:[],zones:{},camName:null,lb:-1,insp:null};
function toast(t,bad){const e=$('toast');e.textContent=t;e.style.borderColor=bad?'var(--r)':'var(--b)';e.style.display='block';clearTimeout(e._t);e._t=setTimeout(()=>e.style.display='none',2600)}
// ---------- header
async function loadSessions(pick){const ss=await api('/api/sessions');let cur=pick||localStorage.session;if(!ss.find(s=>s.name===cur))ss.unshift({name:cur||('briques-'+new Date().toISOString().slice(5,10)),good:0,defect:0});
 $('session').innerHTML=ss.map(s=>`<option value="${s.name}">${s.name} (${s.good} / ${s.defect})</option>`).join('');$('session').value=cur||ss[0].name;localStorage.session=$('session').value}
$('session').onchange=()=>{localStorage.session=$('session').value;S.sel.clear();refresh()};
$('newsess').onclick=()=>{const n=prompt('Nom de la nouvelle session (ex. briques-front-matin)');if(n){const s=n.trim().replace(/[^A-Za-z0-9_.-]+/g,'-');loadSessions(s).then(refresh)}};
async function loadCams(){S.cams=await api('/api/cams');$('cam').innerHTML='<option value="">— choisir —</option>'+S.cams.map(c=>`<option value="${c.id}">${c.label}</option>`).join('');if(localStorage.cam&&S.cams.find(c=>c.id===localStorage.cam)){$('cam').value=localStorage.cam;openCam()}}
function openCam(){const id=$('cam').value;localStorage.cam=id;const v=$('view');if(!id){v.removeAttribute('src');return}
 $('stagemsg').textContent='Ouverture de la caméra…';$('stagemsg').style.display='flex';S.camName=(S.cams.find(c=>c.id===id)||{}).name;drawBox();
 v.onload=()=>{$('stagemsg').style.display='none'};v.onerror=async()=>{const r=await fetch('/stream.mjpg?cam='+encodeURIComponent(id)).then(r=>r.json()).catch(()=>({error:'flux interrompu'}));$('stagemsg').textContent='⚠ '+(r.error||'flux interrompu');$('stagemsg').style.display='flex';v.removeAttribute('src')};
 v.src='/stream.mjpg?cam='+encodeURIComponent(id)+'&t='+Date.now()}
$('cam').onchange=openCam;$('closecam').onclick=async()=>{$('view').removeAttribute('src');await api('/api/close',{});$('stagemsg').textContent='Caméra libérée.';$('stagemsg').style.display='flex'};
$('freesv').onclick=async()=>{const r=await api('/api/free-so101vision',{});toast(r.error?'so101vision : '+r.error:'so101vision a rendu les caméras',r.error)};
async function loadProjects(pick){const p=await api('/api/projects');if(p.error){$('project').innerHTML=`<option value="">${p.error}</option>`;return}
 $('project').innerHTML=p.map(x=>`<option value="${x.id}">${x.name}</option>`).join('');const cur=pick||localStorage.project;if(cur&&p.find(x=>x.id===cur))$('project').value=cur;localStorage.project=$('project').value}
$('project').onchange=()=>{localStorage.project=$('project').value;render()};
$('newproj').onclick=async()=>{const n=prompt('Nom du projet Anomalib Studio');if(!n)return;const r=await api('/api/project',{name:n});if(r.error)return toast(r.error,1);loadProjects(r.id)};
// ---------- capture controls
function setLabel(l){S.label=localStorage.label=l;$('lab-good').classList.toggle('on',l==='good');$('lab-defect').classList.toggle('on',l==='defect')}
$('lab-good').onclick=()=>setLabel('good');$('lab-defect').onclick=()=>setLabel('defect');
$('colours').innerHTML='<span class="muted">Couleur de la brique</span>'+Object.keys(COLOURS).map((c,i)=>`<button data-c="${c}"><span class="dot" style="background:${COLOURS[c]}"></span> ${c} <kbd>${i+1}</kbd></button>`).join('');
function setColour(c){S.colour=localStorage.colour=c;document.querySelectorAll('#colours button').forEach(b=>b.classList.toggle('on',b.dataset.c===c))}
document.querySelectorAll('#colours button').forEach(b=>b.onclick=()=>setColour(b.dataset.c));
$('setColour').innerHTML='<option value="">couleur…</option>'+Object.keys(COLOURS).map(c=>`<option>${c}</option>`).join('');$('fps').value=localStorage.fps||3;$('fps').onchange=()=>localStorage.fps=$('fps').value;
const shotBody=()=>({session:$('session').value,label:S.label,colour:S.colour,fps:+$('fps').value});
let pressT=null,burstT=null,pressing=false;
async function press(){if(pressing)return;if(!$('view').getAttribute('src'))return toast('Ouvrez d\'abord une caméra',1);pressing=true;
 pressT=setTimeout(()=>{if(!pressing)return;$('shoot').classList.add('burst');$('shoot').textContent='Rafale en cours… (relâchez pour arrêter)';const k=async()=>{const r=await api('/api/hold',shotBody());if(r.error){toast(r.error,1);release()}};k();burstT=setInterval(k,300)},400);
 const r=await api('/api/shot',shotBody());if(r.error){toast(r.error,1);release();return}refreshSoon()}
async function release(){if(!pressing)return;pressing=false;clearTimeout(pressT);if(burstT){clearInterval(burstT);burstT=null;const r=await api('/api/release',{});toast(`Rafale : ${r.count||0} photo(s) de plus`)}
 $('shoot').classList.remove('burst');$('shoot').innerHTML='Photo — <kbd>Espace</kbd> (maintenir = rafale)';refresh()}
const sh=$('shoot');sh.onpointerdown=e=>{sh.setPointerCapture(e.pointerId);press()};sh.onpointerup=release;sh.onpointercancel=release;addEventListener('blur',release);
// ---------- crop box drawn on the preview
const stage=$('stage');let drag=null;
stage.ondragstart=e=>e.preventDefault();
stage.onpointerdown=e=>{if(!S.camName)return;e.preventDefault();const r=stage.getBoundingClientRect();drag={x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height};stage.setPointerCapture(e.pointerId)};
stage.onpointermove=e=>{if(!drag)return;const r=stage.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;drawBox([Math.min(x,drag.x),Math.min(y,drag.y),Math.abs(x-drag.x),Math.abs(y-drag.y)])};
stage.onpointerup=async e=>{if(!drag)return;const r=stage.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;const b=[Math.min(x,drag.x),Math.min(y,drag.y),Math.abs(x-drag.x),Math.abs(y-drag.y)];drag=null;
 if(b[2]<0.05||b[3]<0.05){drawBox();return}const z=await api('/api/zone',{cam:S.camName,box:b});if(z.error){toast(z.error,1);drawBox();return}S.zones=z;drawBox();
 toast(S.camName===S.insp?'Zone de recherche enregistrée dans SortOps (inspection.roi)':'Zone de recherche enregistrée')};
$('cropclear').onclick=async()=>{const r=await api('/api/zone',{cam:S.camName,box:null});if(r.error)return toast(r.error,1);S.zones=r;drawBox();toast('Zone de recherche : image entière')};
$('recrop').onclick=async()=>{if(!confirm('Recouper toutes les photos de la session avec les réglages actuels ?'))return;const r=await api('/api/recrop',{session:$('session').value});if(r.error)return toast(r.error,1);
 toast(`${r.done} recoupée(s)`+(r.lost?`, ${r.lost} sans brique (inchangées)`:'')+(r.missing?`, ${r.missing} sans image complète`:''),r.lost||r.missing);refresh()};
function drawBox(tmp){const b=tmp||S.zones[S.camName],e=$('box');const full=!b||(b[0]<=0&&b[1]<=0&&b[2]>=1&&b[3]>=1);if(full&&!tmp){e.style.display='none';$('cropinfo').textContent='image entière';return}
 Object.assign(e.style,{display:'block',left:b[0]*100+'%',top:b[1]*100+'%',width:b[2]*100+'%',height:b[3]*100+'%'});$('cropinfo').textContent=tmp?'…':`${Math.round(b[2]*100)} % × ${Math.round(b[3]*100)} % de l'image`}
// ---------- gallery
function visible(){const fl=$('fLabel').value,fs=$('fSplit').value,fc=$('fCam').value,fo=$('fColour').value,proj=$('project').value;return S.photos.filter(p=>(!fl||p.label===fl)&&(!fs||p.split===fs)&&(!fc||p.cam===fc)&&(!fo||p.colour===fo)&&(!$('fBlur').checked||p.blurry)&&(!$('fNew').checked||!p.sent.includes(proj))).reverse()}
async function refresh(){const ss=$('session').value;if(!ss)return;S.photos=await api('/api/photos?session='+encodeURIComponent(ss));const have=new Set(S.photos.map(p=>p.path));S.sel.forEach(p=>{if(!have.has(p))S.sel.delete(p)});
 const cams=[...new Set(S.photos.map(p=>p.cam))],cur=$('fCam').value;$('fCam').innerHTML='<option value="">toutes caméras</option>'+cams.map(c=>`<option>${c}</option>`).join('');$('fCam').value=cams.includes(cur)?cur:'';
 const cols=[...new Set(S.photos.map(p=>p.colour))],cc=$('fColour').value;$('fColour').innerHTML='<option value="">toutes couleurs</option>'+cols.map(c=>`<option>${c}</option>`).join('');$('fColour').value=cols.includes(cc)?cc:'';render()}
let rsT=null;const refreshSoon=()=>{clearTimeout(rsT);rsT=setTimeout(refresh,250)};
['fLabel','fSplit','fCam','fColour','fBlur','fNew'].forEach(k=>$(k).onchange=render);
function render(){const proj=$('project').value,v=visible();
 $('nGood').textContent=S.photos.filter(p=>p.label==='good').length;$('nTrain').textContent=S.photos.filter(p=>p.label==='good'&&p.split==='train').length;$('nTest').textContent=S.photos.filter(p=>p.label==='good'&&p.split==='test').length;$('nDef').textContent=S.photos.filter(p=>p.label==='defect').length;$('nSent').textContent=S.photos.filter(p=>p.sent.includes(proj)).length;
 const nb=S.photos.filter(p=>p.blurry).length;$('sharpinfo').textContent=nb?`⚠ ${nb} floue(s)`:'';
 $('gal').innerHTML=v.length?v.map((p,i)=>`<div class="ph${S.sel.has(p.path)?' sel':''}" data-p="${p.path}" data-i="${i}"><img loading="lazy" src="/photo/${encodeURI(p.path)}?w=240">
 <div class="tags"><span class="tag ${p.label}">${p.label==='good'?'bonne':'défect.'}</span><span class="tag ${p.split}">${p.split}</span><span class="tag"><span class="dot" style="background:${COLOURS[p.colour]||'#666'}"></span></span>${p.blurry?'<span class="tag warn">floue</span>':''}</div>
 ${p.sent.includes(proj)?'<span class="sent">envoyée</span>':''}<div class="nm">${p.cam} · ${p.name.replace(/^.*__/,'').replace('.jpg','')}</div></div>`).join(''):'<p class="muted">Aucune photo ici. Espace = 1 photo.</p>';
 $('selinfo').textContent=`${S.sel.size} sélectionnée(s) sur ${v.length} affichée(s)`;
 document.querySelectorAll('#gal .ph').forEach(el=>{el.onclick=e=>clickPhoto(e,el);el.ondblclick=()=>openLb(+el.dataset.i)})}
function clickPhoto(e,el){const p=el.dataset.p,v=visible(),i=+el.dataset.i;
 if(e.shiftKey&&S.last!==null){const [a,b]=[Math.min(S.last,i),Math.max(S.last,i)];for(let k=a;k<=b;k++)S.sel.add(v[k].path)}
 else if(e.ctrlKey||e.metaKey||S.sel.size){S.sel.has(p)?S.sel.delete(p):S.sel.add(p)}else{S.sel.add(p)}S.last=i;render()}
$('selall').onclick=()=>{visible().forEach(p=>S.sel.add(p.path));render()};$('selnone').onclick=()=>{S.sel.clear();render()};
async function act(url,body,msg){if(!S.sel.size)return toast('Sélectionnez des photos (clic, Maj+clic pour une plage)',1);const r=await api(url,Object.assign({paths:[...S.sel]},body));if(r.error)return toast(r.error,1);toast(msg(r.n));S.sel.clear();refresh()}
$('del').onclick=()=>act('/api/delete',{},n=>`${n} photo(s) à la corbeille — Ctrl+Z pour annuler`);
$('undo').onclick=async()=>{const r=await api('/api/undo',{});toast(r.n?`${r.n} photo(s) restaurée(s)`:'Rien à annuler');refresh()};
$('toGood').onclick=()=>act('/api/move',{to:'good'},n=>`${n} photo(s) → bonnes`);$('toDef').onclick=()=>act('/api/move',{to:'defect'},n=>`${n} photo(s) → défectueuses`);
$('setColour').onchange=()=>{const c=$('setColour').value;$('setColour').value='';if(c)act('/api/move',{colour:c},n=>`${n} photo(s) → ${c}`)};
async function send(paths){const r=await api('/api/push',{session:$('session').value,project:$('project').value,paths});if(r.error)toast(r.error,1)}
$('sendSel').onclick=()=>S.sel.size?send([...S.sel]):toast('Sélectionnez des photos',1);
$('sendNew').onclick=()=>send(null);
// ---------- lightbox
function openLb(i){S.lb=i;const v=visible();if(i<0||i>=v.length)return closeLb();const p=v[i];$('lbimg').src='/photo/'+encodeURI(p.path);$('lbtxt').textContent=`${i+1}/${v.length} · ${p.label==='good'?'bonne':'défectueuse'} · ${p.split} · ${p.cam} · ${p.colour} · netteté ${p.sharp}${p.blurry?' (floue)':''}`;$('lb').style.display='flex'}
function closeLb(){S.lb=-1;$('lb').style.display='none'}$('lb').onclick=e=>{if(e.target.id==='lb')closeLb()};
async function lbAct(url,body){const p=visible()[S.lb];if(!p)return;await api(url,Object.assign({paths:[p.path]},body));await refresh();openLb(Math.min(S.lb,visible().length-1))}
// ---------- keyboard
addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;
 if(S.lb>=0){if(e.key==='ArrowRight')openLb(S.lb+1);else if(e.key==='ArrowLeft')openLb(S.lb-1);else if(e.key==='Escape')closeLb();
  else if(e.key==='Delete'||e.key==='Backspace')lbAct('/api/delete',{});else if(e.key==='g')lbAct('/api/move',{to:'good'});else if(e.key==='d')lbAct('/api/move',{to:'defect'});e.preventDefault();return}
 if(e.code==='Space'){e.preventDefault();if(!e.repeat)press();return}
 if(e.key==='g')setLabel('good');else if(e.key==='d')setLabel('defect');else if(/^[1-7]$/.test(e.key))setColour(Object.keys(COLOURS)[+e.key-1]);
 else if(e.key==='Delete'||e.key==='Backspace')$('del').click();else if(e.key==='Escape'){S.sel.clear();render()}else if(e.key==='a'&&!e.ctrlKey){$('selall').click()}
 else if(e.key==='z'&&(e.ctrlKey||e.metaKey))$('undo').click()});
addEventListener('keyup',e=>{if(e.code==='Space')release()});
// ---------- polling
setInterval(async()=>{const s=await api('/api/state');$('camstate').textContent=s.cam?('ouverte : '+s.cam_name):'fermée';$('camstate').className='chip '+(s.cam?'ok':'muted');
 S.zones=s.zones||{};S.insp=s.inspection_cam;if(!drag)drawBox();
 $('cropsrc').textContent=`${s.crop.crop_mode}, ${s.crop.crop_size} px, marge ${Math.round(s.crop.crop_margin*100)} %, réglages ${s.crop_source}`;
 $('inspnote').textContent=S.camName===s.inspection_cam?` — pour ${s.inspection_cam}, c'est la zone d'inspection de SortOps (partagée)`:'';
 $('evaltxt').textContent=s.eval.text||$('evaltxt').textContent;$('evalbtn').disabled=s.eval.running;
 const sk=JSON.stringify(s.eval.sheet);if(sk!==S._sk){S._sk=sk;$('evalsheet').innerHTML=(s.eval.sheet||[]).map(x=>`<a href="/photo/${encodeURI(x)}" target="_blank"><img src="/photo/${encodeURI(x)}?w=800&t=${Date.now()}" style="width:100%;margin-top:8px;border-radius:6px"></a>`).join('')}
 if(s.burst&&pressing)$('shoot').textContent=`Rafale en cours : ${s.burst.count} photo(s) — relâchez pour arrêter`;
 $('pushmsg').textContent=s.push.running?`Envoi : ${s.push.done}/${s.push.total}…`:s.push.msg;if(!s.push.running&&s.push.msg&&S._pm!==s.push.msg){S._pm=s.push.msg;refresh()}
 $('report').textContent=s.report.text||$('report').textContent;$('rep').disabled=$('repfast').disabled=s.report.running;
 if(s.burst&&pressing)refreshSoon()},700);
$('evalbtn').onclick=async()=>{const r=await api('/api/eval',{session:$('session').value});if(r.error)toast(r.error,1)};
$('rep').onclick=()=>api('/api/report',{session:$('session').value,model:true});$('repfast').onclick=()=>api('/api/report',{session:$('session').value,model:false});
setLabel(S.label);setColour(S.colour);loadSessions().then(refresh);loadCams();loadProjects();
</script></html>"""


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    threading.Thread(target=janitor, daemon=True).start()
    threading.Thread(target=burst_loop, daemon=True).start()

    class Server6(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    ThreadingHTTPServer.daemon_threads = True
    servers = [ThreadingHTTPServer(("127.0.0.1", A.port), H)]
    try:
        servers.append(Server6(("::1", A.port), H))
    except OSError as e:
        print(f"no IPv6 loopback listener: {e}", flush=True)
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"capture app on http://127.0.0.1:{A.port}  photos -> {ROOT}", flush=True)
    servers[0].serve_forever()
