import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type Annotation,
  type Box,
  type CameraAnnotation,
  type DefectShape,
  type InspectionCycle,
  type InspectionVerdict,
  inspectionFile,
  type StoredShot,
} from "./api";

/** Hooked by the open annotator so the global G / D / N keys answer the verdict WITH the drawing. */
export const verdictBridge: { submit?: (verdict: InspectionVerdict) => void } = {};

type Tool = "piece" | "rect" | "brush" | "erase";

interface Props {
  detail: InspectionCycle;
  /** "verdict": the arm waits for the answer; "edit": correcting a stored cycle (Data tab). */
  purpose: "verdict" | "edit";
  onSubmit: (verdict: InspectionVerdict, annotation: { cameras: Record<string, CameraAnnotation>; note: string }) => void;
  busy?: boolean;
}

const clamp = (v: number) => Math.min(1, Math.max(0, v));

function initial(detail: InspectionCycle): Record<string, CameraAnnotation> {
  const out: Record<string, CameraAnnotation> = {};
  const cameras = new Set([...Object.keys(detail.best), ...Object.keys(detail.annotation?.cameras ?? {})]);
  for (const camera of cameras) {
    const saved = detail.annotation?.cameras[camera];
    const best = detail.best[camera];
    out[camera] = {
      shot: saved?.shot ?? best?.shot ?? null,
      piece: saved?.piece ?? best?.crop_box ?? null,
      defects: saved?.defects ? saved.defects.map((d) => ({ ...d })) : [],
    };
  }
  return out;
}

/** The model's regions above the defect contour (crop coords) as defect boxes in frame coords. */
function modelRegions(shot: StoredShot | undefined): DefectShape[] {
  if (!shot?.heat_info) return [];
  const [bx, by, bw, bh] = shot.crop_box;
  return shot.heat_info.regions.slice(0, 5).map(([x, y, w, h]) => ({ type: "rect", box: [bx + x * bw, by + y * bh, w * bw, h * bh] as Box }));
}

export function Annotator({ detail, purpose, onSubmit, busy }: Props) {
  const cameraNames = useMemo(() => Object.keys(initial(detail)).sort(), [detail]);
  const [cams, setCams] = useState<Record<string, CameraAnnotation>>(() => initial(detail));
  const [history, setHistory] = useState<Record<string, CameraAnnotation>[]>([]);
  const [camera, setCamera] = useState<string>(() => (cameraNames.includes("wrist") ? "wrist" : cameraNames[0] ?? ""));
  const [tool, setTool] = useState<Tool>("rect");
  const [brush, setBrush] = useState(0.02);
  const [showHeat, setShowHeat] = useState(true);
  const [note, setNote] = useState(detail.annotation?.note ?? "");
  const [draft, setDraft] = useState<DefectShape | { type: "piece"; box: Box } | null>(null);
  const start = useRef<[number, number] | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    setCams(initial(detail));
    setHistory([]);
    setNote(detail.annotation?.note ?? "");
  }, [detail]);

  const current = cams[camera];
  const shots = detail.shots.filter((s) => s.camera === camera);
  const shot = shots.find((s) => s.shot === current?.shot) ?? detail.best[camera];
  const [fw, fh] = shot?.frame_size ?? [640, 480];

  const commit = useCallback(
    (next: Record<string, CameraAnnotation>) => {
      setHistory((h) => [...h.slice(-40), cams]);
      setCams(next);
    },
    [cams],
  );
  const update = (patch: Partial<CameraAnnotation>) => {
    const base = cams[camera];
    if (base) commit({ ...cams, [camera]: { ...base, ...patch } });
  };
  const undo = useCallback(() => {
    setHistory((h) => {
      const last = h[h.length - 1];
      if (!last) return h;
      setCams(last);
      return h.slice(0, -1);
    });
  }, []);

  const point = (event: React.PointerEvent): [number, number] => {
    const rect = svgRef.current!.getBoundingClientRect();
    return [clamp((event.clientX - rect.left) / rect.width), clamp((event.clientY - rect.top) / rect.height)];
  };

  const down = (event: React.PointerEvent) => {
    if (!current || busy) return;
    event.preventDefault();
    (event.target as Element).setPointerCapture?.(event.pointerId);
    const p = point(event);
    if (tool === "erase") {
      const keep = current.defects.filter((d) => !hits(d, p, fw, fh));
      if (keep.length !== current.defects.length) update({ defects: keep });
      return;
    }
    start.current = p;
    if (tool === "brush") setDraft({ type: "stroke", points: [p], width: brush });
    else setDraft({ type: tool === "piece" ? "piece" : "rect", box: [p[0], p[1], 0, 0] } as never);
  };
  const move = (event: React.PointerEvent) => {
    if (!draft || !start.current) return;
    const p = point(event);
    if (draft.type === "stroke") setDraft({ ...draft, points: [...draft.points, p] });
    else {
      const [x0, y0] = start.current;
      setDraft({ ...draft, box: [Math.min(x0, p[0]), Math.min(y0, p[1]), Math.abs(p[0] - x0), Math.abs(p[1] - y0)] });
    }
  };
  const up = () => {
    if (!draft || !current) return;
    if (draft.type === "piece") {
      if (draft.box[2] > 0.02 && draft.box[3] > 0.02) update({ piece: draft.box });
    } else if (draft.type === "rect") {
      if (draft.box[2] > 0.004 && draft.box[3] > 0.004) update({ defects: [...current.defects, draft] });
    } else {
      update({ defects: [...current.defects, draft] });
    }
    setDraft(null);
    start.current = null;
  };

  const submit = useCallback(
    (verdict: InspectionVerdict) => {
      if (busy) return;
      const cameras: Record<string, CameraAnnotation> = {};
      for (const [name, value] of Object.entries(cams)) cameras[name] = { ...value, defects: verdict === "bad" ? value.defects : [] };
      onSubmit(verdict, { cameras: verdict === "no_piece" ? {} : cameras, note });
    },
    [busy, cams, note, onSubmit],
  );

  // G / D / N go through the bridge (App's global handler); the drawing keys are local.
  useEffect(() => {
    // Only the live verdict annotator answers the waiting brick; an edit in the Data tab never does.
    if (purpose === "verdict") verdictBridge.submit = submit;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, select, textarea")) return;
      if ((event.ctrlKey || event.metaKey) && event.code === "KeyZ") {
        event.preventDefault();
        undo();
        return;
      }
      const tools: Record<string, Tool> = { KeyP: "piece", KeyR: "rect", KeyB: "brush", KeyE: "erase" };
      const picked = tools[event.code];
      if (picked) setTool(picked);
      else if (event.code === "KeyH") setShowHeat((v) => !v);
      else if (event.code === "Tab" && cameraNames.length > 1) {
        event.preventDefault();
        setCamera((c) => cameraNames[(cameraNames.indexOf(c) + 1) % cameraNames.length] ?? c);
      } else if (purpose === "edit") {
        const verdict = ({ KeyG: "good", KeyD: "bad", KeyN: "no_piece" } as const)[event.code as "KeyG" | "KeyD" | "KeyN"];
        if (verdict) {
          // Capture phase + stop: App's global G / D / N (live brick verdict) must not fire for an edit.
          event.stopPropagation();
          submit(verdict);
        }
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      if (verdictBridge.submit === submit) verdictBridge.submit = undefined;
    };
  }, [submit, undo, cameraNames, purpose]);

  if (!current) return <div className="faint">No inspection frame was stored for this brick.</div>;
  const heatShown = showHeat && shot?.heat;
  const suggestions = modelRegions(shot);
  const [cx, cy, cw, ch] = shot?.crop_box ?? [0, 0, 1, 1];

  return (
    <div className="annotator">
      <div className="row wrap between" style={{ gap: 8 }}>
        <div className="segmented" role="group" aria-label="Camera">
          {cameraNames.map((name) => (
            <button key={name} className="shadow" aria-pressed={camera === name} onClick={() => setCamera(name)} style={{ height: 28 }}>
              {name}
              {cams[name]?.defects.length ? ` · ${cams[name].defects.length}` : ""}
            </button>
          ))}
        </div>
        <div className="segmented" role="group" aria-label="Tool">
          {(
            [
              ["piece", "Piece box", "P"],
              ["rect", "Defect box", "R"],
              ["brush", "Brush", "B"],
              ["erase", "Erase", "E"],
            ] as const
          ).map(([key, label, k]) => (
            <button key={key} className="shadow" aria-pressed={tool === key} onClick={() => setTool(key)} style={{ height: 28 }} title={`${label} (${k})`}>
              {label} <kbd>{k}</kbd>
            </button>
          ))}
        </div>
        {tool === "brush" && (
          <label className="row muted" style={{ fontSize: 12 }}>
            width
            <input type="range" min={0.005} max={0.08} step={0.005} value={brush} onChange={(e) => setBrush(Number(e.target.value))} />
          </label>
        )}
        <div className="row">
          {shot?.heat && (
            <label className="row muted" style={{ fontSize: 12 }}>
              <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> Anomaly map <kbd>H</kbd>
            </label>
          )}
          <button className="btn sm" onClick={undo} disabled={!history.length} title="Undo (Ctrl+Z)">
            Undo
          </button>
          <button className="btn sm" onClick={() => update({ defects: [] })} disabled={!current.defects.length}>
            Clear defects
          </button>
        </div>
      </div>

      <div className="annot-stage" style={{ aspectRatio: `${fw} / ${fh}` }}>
        {shot && <img src={inspectionFile(detail.run, detail.cycle, shot.frame)} alt={`${camera} frame ${shot.shot}`} draggable={false} />}
        {heatShown && shot?.heat && (
          <img
            className="annot-heat"
            src={inspectionFile(detail.run, detail.cycle, shot.heat)}
            alt="anomaly map"
            draggable={false}
            style={{ left: `${cx * 100}%`, top: `${cy * 100}%`, width: `${cw * 100}%`, height: `${ch * 100}%` }}
          />
        )}
        <svg
          ref={svgRef}
          viewBox={`0 0 ${fw} ${fh}`}
          preserveAspectRatio="none"
          className={`annot-svg tool-${tool}`}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          onPointerCancel={up}
        >
          {current.piece && <BoxShape box={current.piece} fw={fw} fh={fh} className="piece" />}
          {current.defects.map((d, i) => (
            <Shape key={i} shape={d} fw={fw} fh={fh} />
          ))}
          {draft && (draft.type === "piece" ? <BoxShape box={draft.box} fw={fw} fh={fh} className="piece draft" /> : <Shape shape={draft} fw={fw} fh={fh} draft />)}
        </svg>
      </div>

      <div className="row wrap" style={{ gap: 6, marginTop: 6 }}>
        <span className="faint" style={{ fontSize: 11 }}>
          {shot ? `frame ${shot.shot} · ${shot.usability.usable ? "usable" : `not usable (${shot.usability.reason ?? "?"})`}` : ""}
          {shot?.heat_info ? ` · map ${shot.heat_info.max_units.toFixed(1)}× the defect line${shot.heat_info.calibrated ? "" : " (uncalibrated)"}` : ""}
        </span>
        {suggestions.length > 0 && (
          <button className="btn sm" onClick={() => update({ defects: [...current.defects, ...suggestions] })} title="Add the model's regions above the defect contour">
            + {suggestions.length} model region{suggestions.length > 1 ? "s" : ""}
          </button>
        )}
        <button className="btn sm" onClick={() => shot && update({ piece: shot.crop_box })} title="Reset the piece box to the automatic brick crop">
          Auto piece box
        </button>
      </div>

      {shots.length > 1 && (
        <div className="annot-shots" aria-label="Choose the reference frame">
          {shots.map((s) => (
            <button
              key={s.shot}
              className={`annot-thumb ${s.shot === current.shot ? "on" : ""} ${s.usability.usable ? "" : "unusable"}`}
              onClick={() => update({ shot: s.shot, piece: current.piece ?? s.crop_box })}
              title={`frame ${s.shot}${s.score != null ? ` · score ${s.score.toFixed(2)}` : ""}${s.usability.usable ? "" : " · not usable"}`}
            >
              <img src={inspectionFile(detail.run, detail.cycle, s.heat && showHeat ? s.heat : s.crop)} alt="" loading="lazy" />
            </button>
          ))}
        </div>
      )}

      <div className="row wrap between" style={{ marginTop: 10, gap: 8 }}>
        <input className="annot-note" placeholder="Note (defect type, marker colour…)" value={note} onChange={(e) => setNote(e.target.value)} />
        <div className="row">
          <button className="btn verdict-good" disabled={busy} onClick={() => submit("good")}>
            ✓ Good <kbd>G</kbd>
          </button>
          <button className="btn verdict-bad" disabled={busy} onClick={() => submit("bad")} title="Defect, with the shapes drawn on each camera">
            ✗ Defect{Object.values(cams).some((c) => c.defects.length) ? ` (${Object.values(cams).reduce((n, c) => n + c.defects.length, 0)} zones)` : ""} <kbd>D</kbd>
          </button>
          <button className="btn ghost" disabled={busy} onClick={() => submit("no_piece")}>
            No brick <kbd>N</kbd>
          </button>
        </div>
      </div>
      <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>
        Draw the piece box (P) tight around the brick, then the defects (R box, B brush, E erase) on each camera (Tab). {purpose === "verdict" ? "The arm waits for your answer. Space is still the e-stop." : "Saving replaces this cycle's annotation."}
      </div>
    </div>
  );
}

export function BoxShape({ box, fw, fh, className }: { box: Box | readonly number[]; fw: number; fh: number; className: string }) {
  const [x, y, w, h] = box;
  return <rect className={className} x={x * fw} y={y * fh} width={w * fw} height={h * fh} />;
}

export function Shape({ shape, fw, fh, draft }: { shape: DefectShape; fw: number; fh: number; draft?: boolean }) {
  if (shape.type === "rect") return <BoxShape box={shape.box} fw={fw} fh={fh} className={`defect ${draft ? "draft" : ""}`} />;
  const points = shape.points.map(([x, y]) => `${x * fw},${y * fh}`).join(" ");
  return <polyline className={`defect-stroke ${draft ? "draft" : ""}`} points={points} strokeWidth={Math.max(1, shape.width * fw)} />;
}

export function hits(shape: DefectShape, [px, py]: [number, number], fw: number, fh: number): boolean {
  if (shape.type === "rect") {
    const [x, y, w, h] = shape.box;
    return px >= x && px <= x + w && py >= y && py <= y + h;
  }
  const r = Math.max(shape.width, 0.01);
  return shape.points.some(([x, y]) => Math.hypot((x - px) * fw, (y - py) * fh) <= r * fw);
}

export function annotationSummary(annotation: Annotation | null): string {
  if (!annotation) return "not annotated";
  const zones = Object.values(annotation.cameras).reduce((n, c) => n + c.defects.length, 0);
  return `${annotation.verdict === "bad" ? "defect" : annotation.verdict.replace("_", " ")} · ${annotation.source}${zones ? ` · ${zones} zone${zones > 1 ? "s" : ""}` : ""}`;
}
