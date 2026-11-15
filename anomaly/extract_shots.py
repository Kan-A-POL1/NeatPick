#!/usr/bin/env python
"""Extract the 'brick presented to the camera' stills from a Physical AI Studio (LeRobot v3) dataset.

The presentation is found from the joint states only: gripper closed + arm motionless in the
presentation pose. Two holds per episode: view A (before the wrist roll) and view B (after).
Usage: extract_shots.py DATASET_DIR OUT_DIR [--per-hold N] [--cams front,overhead,wrist]
"""
import argparse, glob, json, os
import av, numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("dataset"); p.add_argument("out")
p.add_argument("--per-hold", type=int, default=3, help="stills kept per hold, spread over it")
p.add_argument("--cams", default="front,overhead,wrist")
p.add_argument("--grip-max", type=float, default=10, help="gripper.pos below this = closed")
p.add_argument("--still", type=float, default=1.0, help="max joint move per frame to count as motionless")
p.add_argument("--min-len", type=int, default=5, help="min hold length in frames")
p.add_argument("--pose-tol", type=float, default=15, help="max distance to the presentation pose, per joint (wrist roll excluded)")
a = p.parse_args()

D = a.dataset
names = json.load(open(f"{D}/meta/info.json"))["features"]["observation.state"]["names"]
J = {n.split(".")[0]: i for i, n in enumerate(names)}
df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/data/*/*.parquet"))).sort_values("index")

wanted = {}  # global frame index -> label
rows = []
for ep, g in df.groupby("episode_index"):
    s = np.stack(g["observation.state"].values); idx = g["index"].values
    speed = np.r_[0, np.abs(np.diff(s, axis=0)).max(axis=1)]
    # presentation pose = where the episode ends (the demo stops while showing view B)
    arm = [J[k] for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")]
    near = (np.abs(s[:, arm] - np.median(s[-10:, arm], axis=0)).max(axis=1) < a.pose_tol)
    ok = (s[:, J["gripper"]] < a.grip_max) & (speed < a.still) & near
    holds, start = [], None
    for i, v in enumerate(np.r_[ok, False]):
        if v and start is None: start = i
        if not v and start is not None:
            if i - start >= a.min_len: holds.append((start, i))
            start = None
    # merge holds whose wrist roll is the same (micro-moves split them)
    merged = []
    for h in holds:
        r = s[h[0]:h[1], J["wrist_roll"]].mean()
        if merged and abs(merged[-1][2] - r) < 30: merged[-1] = (merged[-1][0], h[1], merged[-1][2])
        else: merged.append((h[0], h[1], r))
    for k, (b, e, r) in enumerate(merged):
        view = "B" if abs(r - s[-1, J["wrist_roll"]]) < 30 else "A"
        picks = np.unique(np.linspace(b + 2, e - 3, a.per_hold).round().astype(int)) if e - b > 6 else [ (b + e) // 2 ]
        for n, i in enumerate(picks):
            wanted[int(idx[i])] = f"ep{ep:03d}_{view}_{n}"
        rows.append(dict(episode=ep, view=view, start=b, end=e, seconds=round((e - b) / 30, 2), wrist_roll=round(float(r), 1)))
print(pd.DataFrame(rows).to_string(index=False))

for cam in a.cams.split(","):
    os.makedirs(f"{a.out}/{cam}", exist_ok=True)
    i = 0
    for f in sorted(glob.glob(f"{D}/videos/observation.images.{cam}/*/*.mp4")):
        with av.open(f) as c:
            for fr in c.decode(video=0):
                if i in wanted: fr.to_image().save(f"{a.out}/{cam}/{wanted[i]}.png")
                i += 1
    print(cam, "->", len(os.listdir(f"{a.out}/{cam}")), "stills")
