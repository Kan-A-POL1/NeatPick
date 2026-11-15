import { useEffect, useState } from "react";
import { api, INSPECTION_CAMERAS, type InspectionRun, type TrainJob, usePoll } from "./api";

const MODELS = [
  ["patchcore", "PatchCore", "Learns from good bricks only, a few minutes; the safe default"],
  ["padim", "PaDiM", "Good bricks only, very fast, a little less sharp"],
  ["fre", "FRE", "Good bricks only, fast at inference"],
  ["efficient_ad", "EfficientAD", "Slower to train (20 epochs), very fast inference"],
  ["dinomaly", "Dinomaly", "Transformer, slow to train on this PC"],
] as const;

const metric = (metrics: Record<string, number>, name: string) => {
  const k = Object.keys(metrics).find((m) => m.toLowerCase().endsWith(name.toLowerCase()));
  return k === undefined ? null : metrics[k]!;
};

/** Data -> Train: an Anomalib model on the reviewed frames; it lands in the anomaly models, never activated by itself. */
export function TrainPanel({ runs, selected, notify }: { runs: InspectionRun[]; selected: string | null; notify: (m: string, e?: boolean) => void }) {
  const status = usePoll<{ job: TrainJob | null }>("/api/anomaly/train", 2000);
  const [chosen, setChosen] = useState<string[]>([]);
  const [camera, setCamera] = useState<string>("wrist");
  const [model, setModel] = useState<string>("patchcore");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!chosen.length && selected) setChosen([selected]);
  }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps

  const job = status.data?.job ?? null;
  const running = job?.status === "preparing" || job?.status === "training";
  // Training uses the selected camera only: count its kept reviewed frames.
  const reviewed = runs.filter((r) => chosen.includes(r.run)).reduce((n, r) => n + (r.kept_by_camera?.[camera] ?? 0), 0);

  const train = () => {
    setBusy(true);
    api<{ job: TrainJob }>("/api/anomaly/train", { body: { runs: chosen, camera, model, name: name.trim() || null } })
      .then(() => {
        notify(`Training ${model} on ${camera} started`);
        status.refresh();
      })
      .catch((error: Error) => notify(error.message, true))
      .finally(() => setBusy(false));
  };
  const cancel = () =>
    api("/api/anomaly/train/cancel", { body: {} })
      .then(() => status.refresh())
      .catch((error: Error) => notify(error.message, true));

  const auroc = job ? metric(job.metrics, "image_AUROC") : null;
  const pixel = job ? metric(job.metrics, "pixel_AUROC") : null;
  const f1 = job ? metric(job.metrics, "image_F1Score") : null;

  return (
    <section className="panel">
      <div className="panel-head">
        <div className="panel-title">Train an anomaly model (Anomalib → OpenVINO)</div>
      </div>
      <div className="panel-body stack">
        <div className="row wrap" style={{ gap: 12, alignItems: "flex-start" }}>
          <fieldset className="train-runs">
            <legend className="muted">Runs (reviewed frames only)</legend>
            {runs.map((r) => (
              <label key={r.run} className="row" style={{ fontSize: 12 }}>
                <input
                  type="checkbox"
                  checked={chosen.includes(r.run)}
                  onChange={(e) => setChosen((c) => (e.target.checked ? [...c, r.run] : c.filter((x) => x !== r.run)))}
                />
                <span className="mono">{r.run}</span>
                <span className="faint">
                  {r.kept_by_camera?.[camera] ?? 0} kept on {camera} · {r.reviewed} reviewed / {r.frames}
                </span>
              </label>
            ))}
          </fieldset>
          <div className="stack" style={{ gap: 8 }}>
            <label className="row" style={{ fontSize: 12 }}>
              Camera
              <select value={camera} onChange={(e) => setCamera(e.target.value)}>
                {INSPECTION_CAMERAS.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
            <label className="row" style={{ fontSize: 12 }}>
              Model
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {MODELS.map(([k, label]) => (
                  <option key={k} value={k}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <div className="faint" style={{ fontSize: 11, maxWidth: 320 }}>
              {MODELS.find(([k]) => k === model)?.[2]}
            </div>
            <label className="row" style={{ fontSize: 12 }}>
              Name
              <input className="annot-note" style={{ height: 28 }} placeholder={`${model}-<date>`} value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <div className="row">
              <button className="btn primary" disabled={busy || running || !chosen.length || reviewed < 5} onClick={train} title={reviewed >= 5 ? "" : `Review at least 5 frames of good bricks on ${camera} first`}>
                Train on {reviewed} reviewed {camera} frame{reviewed === 1 ? "" : "s"}
              </button>
              {running && (
                <button className="btn ghost" onClick={cancel}>
                  Cancel
                </button>
              )}
            </div>
          </div>
        </div>

        {job && (
          <div className={`train-job ${job.status}`}>
            <div className="row between">
              <b>
                {job.model} · {job.camera} · <span className="mono">{job.name}</span>
              </b>
              <span className="chip">{job.status}</span>
            </div>
            <div className="review-bar" style={{ marginTop: 6 }}>
              <div style={{ width: `${job.progress}%` }} />
            </div>
            <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
              {job.message}
              {Object.keys(job.images).length > 0 &&
                ` · train ${job.images["train/good"] ?? 0} good · test ${job.images["test/good"] ?? 0} good + ${job.images["test/defect"] ?? 0} defect`}
            </div>
            {job.status === "done" && (
              <div className="kv" style={{ marginTop: 6 }}>
                {auroc !== null && (
                  <span>
                    image AUROC <b className="mono">{auroc.toFixed(3)}</b>
                  </span>
                )}
                {f1 !== null && (
                  <span>
                    image F1 <b className="mono">{f1.toFixed(3)}</b>
                  </span>
                )}
                {pixel !== null && (
                  <span>
                    pixel AUROC <b className="mono">{pixel.toFixed(3)}</b>
                  </span>
                )}
                <span className="faint">→ Models → anomaly_{job.camera}: choose it, then “Calibrate on good bricks”</span>
              </div>
            )}
            {(job.status === "failed" || running) && job.log_tail.length > 0 && (
              <pre className="train-log">{job.log_tail.slice(-8).join("\n")}</pre>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
