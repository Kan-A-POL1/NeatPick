import { useCallback, useEffect, useState } from "react";
import { Annotator, annotationSummary } from "./Annotator";
import { Review } from "./Review";
import { TrainPanel } from "./TrainPanel";
import { api, type InspectionCycle, type InspectionCycleSummary, type InspectionRun, inspectionFile, type InspectionVerdict, usePoll } from "./api";

const size = (bytes: number) => (bytes > 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1e3)} kB`);

/** Inspection data of every run: browse, correct the annotations, download Anomalib-ready datasets. */
export function Data({ notify }: { notify: (m: string, e?: boolean) => void }) {
  const runs = usePoll<InspectionRun[]>("/api/inspection/runs", 5000);
  const [run, setRun] = useState<string | null>(null);
  const [cycles, setCycles] = useState<InspectionCycleSummary[]>([]);
  const [filter, setFilter] = useState<"all" | InspectionVerdict | "unlabeled">("all");
  const [open, setOpen] = useState<InspectionCycle | null>(null);
  const [heat, setHeat] = useState(true);
  const [saving, setSaving] = useState(false);
  const [reviewing, setReviewing] = useState(false);

  const selected = run ?? runs.data?.[0]?.run ?? null;
  const loadCycles = useCallback(() => {
    if (!selected) return;
    api<InspectionCycleSummary[]>(`/api/inspection/runs/${encodeURIComponent(selected)}/cycles`)
      .then(setCycles)
      .catch((error: Error) => notify(error.message, true));
  }, [selected, notify]);
  useEffect(loadCycles, [loadCycles]);

  const openCycle = (cycle: number) =>
    api<InspectionCycle>(`/api/inspection/runs/${encodeURIComponent(selected!)}/cycles/${cycle}`)
      .then(setOpen)
      .catch((error: Error) => notify(error.message, true));

  const shown = cycles.filter((c) => filter === "all" || (filter === "unlabeled" ? c.verdict === null : c.verdict === filter));
  const index = open ? shown.findIndex((c) => c.cycle === open.cycle) : -1;

  const save = (verdict: InspectionVerdict, annotation: { cameras: Record<string, unknown>; note: string }) => {
    if (!open) return;
    setSaving(true);
    api(`/api/inspection/runs/${encodeURIComponent(open.run)}/cycles/${open.cycle}/annotation`, { body: { verdict, ...annotation } })
      .then(() => {
        notify(`Cycle ${open.cycle} saved as ${verdict === "bad" ? "defect" : verdict.replace("_", " ")}`);
        loadCycles();
        runs.refresh();
        const next = shown[index + 1];
        if (next) openCycle(next.cycle);
        else setOpen(null);
      })
      .catch((error: Error) => notify(error.message, true))
      .finally(() => setSaving(false));
  };

  return (
    <div className="data-page">
      <section className="panel">
        <div className="panel-head">
          <div className="panel-title">Inspection runs</div>
          <div className="row">
            <a className="btn sm" href="/api/inspection/runs/all/dataset.zip?only=reviewed" download title="Reviewed frames only, cut to your piece boxes">
              Download all reviewed (zip)
            </a>
            <a className="btn sm ghost" href="/api/inspection/runs/all/dataset.zip?only=all" download title="Without review: one reference frame per cycle and its annotation, every usable good frame">
              Unreviewed zip
            </a>
          </div>
        </div>
        <div className="panel-body">
          {!runs.data?.length ? (
            <div className="faint">No inspection frame stored yet. Every cycle keeps its frames here, run by run.</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Cycles</th>
                  <th>Good</th>
                  <th>Defect</th>
                  <th>No brick</th>
                  <th>Unlabeled</th>
                  <th>Frames</th>
                  <th>Reviewed</th>
                  <th>Size</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.data.map((r) => (
                  <tr key={r.run} className={r.run === selected ? "on" : ""} onClick={() => { setRun(r.run); setOpen(null); setReviewing(false); }}>
                    <td className="mono">{r.run}</td>
                    <td className="num">{r.cycles}</td>
                    <td className="num good">{r.good}</td>
                    <td className="num bad">{r.bad}</td>
                    <td className="num">{r.no_piece}</td>
                    <td className="num">{r.unlabeled}</td>
                    <td className="num">{r.frames}</td>
                    <td className="num">
                      {r.reviewed}
                      {r.frames ? <span className="faint"> · {Math.round((100 * r.reviewed) / r.frames)}%</span> : null}
                    </td>
                    <td className="num">{size(r.bytes)}</td>
                    <td className="row">
                      <button className="btn sm primary" onClick={(e) => { e.stopPropagation(); setRun(r.run); setOpen(null); setReviewing(true); }}>
                        Review
                      </button>
                      <a className="btn sm" href={`/api/inspection/runs/${encodeURIComponent(r.run)}/dataset.zip?only=reviewed`} download onClick={(e) => e.stopPropagation()} title="Reviewed frames only">
                        Zip
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      {selected && (
        <section className="panel">
          <div className="panel-head">
            <div className="panel-title">
              Run <span className="mono">{selected}</span> · {shown.length} cycle{shown.length === 1 ? "" : "s"}
            </div>
            <div className="row">
              {!reviewing && (
                <button className="btn sm primary" onClick={() => { setOpen(null); setReviewing(true); }} title="Go through every frame: piece box, then defects">
                  Review frames
                </button>
              )}
              <div className="segmented" role="group" aria-label="Filter">
                {(["all", "good", "bad", "no_piece", "unlabeled"] as const).map((key) => (
                  <button key={key} className="shadow" style={{ height: 26, fontSize: 12 }} aria-pressed={filter === key} onClick={() => setFilter(key)}>
                    {{ all: "All", good: "Good", bad: "Defect", no_piece: "No brick", unlabeled: "Unlabeled" }[key]}
                  </button>
                ))}
              </div>
              <label className="row muted" style={{ fontSize: 12 }}>
                <input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} /> Anomaly maps
              </label>
            </div>
          </div>
          <div className="panel-body">
            {reviewing ? (
              <Review run={selected} notify={notify} onClose={() => { setReviewing(false); runs.refresh(); loadCycles(); }} />
            ) : open ? (
              <div>
                <div className="row between" style={{ marginBottom: 8 }}>
                  <div>
                    <b>Cycle {open.cycle}</b> <span className="muted">· {annotationSummary(open.annotation)}</span>
                  </div>
                  <div className="row">
                    <button className="btn sm" disabled={index <= 0} onClick={() => { const prev = shown[index - 1]; if (prev) openCycle(prev.cycle); }}>
                      ← Previous
                    </button>
                    <button className="btn sm" disabled={index < 0 || index >= shown.length - 1} onClick={() => { const next = shown[index + 1]; if (next) openCycle(next.cycle); }}>
                      Next →
                    </button>
                    <button className="btn sm" onClick={() => setOpen(null)}>
                      Back to the grid
                    </button>
                  </div>
                </div>
                <Annotator detail={open} purpose="edit" onSubmit={save} busy={saving} />
              </div>
            ) : (
              <div className="cycle-grid">
                {shown.map((c) => (
                  <button key={c.cycle} className={`cycle-card ${c.verdict ?? "unlabeled"}`} onClick={() => openCycle(c.cycle)}>
                    <div className="cycle-thumbs">
                      {Object.entries(c.best).map(([camera, b]) => (
                        <img key={camera} src={inspectionFile(selected, c.cycle, heat && b.heat ? b.heat : b.crop)} alt={`${camera} crop`} loading="lazy" title={camera} />
                      ))}
                    </div>
                    <div className="cycle-meta">
                      <b>#{c.cycle}</b>
                      <span className={`chip ${c.verdict === "bad" ? "bad" : c.verdict === "good" ? "good" : ""}`}>
                        {c.verdict === null ? "unlabeled" : c.verdict === "bad" ? "defect" : c.verdict.replace("_", " ")}
                      </span>
                      <span className="faint">
                        {c.source ?? ""}
                        {c.defects ? ` · ${c.defects} zone${c.defects > 1 ? "s" : ""}` : ""} · {c.frames} fr.
                      </span>
                    </div>
                  </button>
                ))}
                {!shown.length && <div className="faint">No cycle here.</div>}
              </div>
            )}
          </div>
        </section>
      )}

      {!!runs.data?.length && <TrainPanel runs={runs.data} selected={selected} notify={notify} />}
    </div>
  );
}
