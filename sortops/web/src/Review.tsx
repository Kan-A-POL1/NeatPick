import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BoxShape, Shape, hits } from "./Annotator";
import { api, type Box, type DefectShape, inspectionFile, type ReviewItem, type ReviewQueue } from "./api";

type Tool = "piece" | "rect" | "brush" | "erase";
type Filter = "todo" | "all" | "bad";
interface Draft {
  piece: Box | null;
  defects: DefectShape[];
  excluded: boolean;
}

const clamp = (v: number) => Math.min(1, Math.max(0, v));
const same = (a: ReviewItem, b: ReviewItem) => a.cycle === b.cycle && a.camera === b.camera;

/** The proposal for a frame not reviewed yet: its own brick box (the brick moves), and on a defect brick the defects
 * of the previous frame of the same cycle and camera (the brick turns slowly: adjust them rather than redraw). */
function proposal(item: ReviewItem, previous: ReviewItem | undefined, drafts: Map<string, Draft>): Draft {
  if (item.reviewed) return { piece: item.piece, defects: item.defects, excluded: item.excluded };
  const prev = previous && same(previous, item) ? drafts.get(key(previous)) ?? previous : undefined;
  const defects = item.verdict === "bad" && !item.defects.length && prev ? prev.defects.map((d) => ({ ...d })) : item.defects;
  const piece = item.found ? item.piece : (prev?.piece ?? item.piece);
  return { piece, defects, excluded: item.excluded };
}
const key = (i: ReviewItem) => `${i.cycle}/${i.shot}`;

/** Data -> Review: every stored frame of a run, one by one. Confirm the piece box (the only zone analysed) and, on
 * defect bricks, draw this frame's defects. Only reviewed frames go to the export and to training. */
export function Review({ run, notify, onClose }: { run: string; notify: (m: string, e?: boolean) => void; onClose: () => void }) {
  const [queue, setQueue] = useState<ReviewQueue | null>(null);
  const [filter, setFilter] = useState<Filter>("todo");
  const [index, setIndex] = useState(0);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [history, setHistory] = useState<Draft[]>([]);
  const [tool, setTool] = useState<Tool>("piece");
  const [brush, setBrush] = useState(0.02);
  const [busy, setBusy] = useState(false);
  const [shape, setShape] = useState<DefectShape | { type: "piece"; box: Box } | null>(null);
  const start = useRef<[number, number] | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drafts = useRef(new Map<string, Draft>());

  const load = useCallback(() => {
    api<ReviewQueue>(`/api/inspection/runs/${encodeURIComponent(run)}/review`)
      .then(setQueue)
      .catch((error: Error) => notify(error.message, true));
  }, [run, notify]);
  useEffect(load, [load]);

  const items = useMemo(() => {
    const all = queue?.items ?? [];
    if (filter === "bad") return all.filter((i) => i.verdict === "bad");
    if (filter === "todo") return all.filter((i) => !i.reviewed);
    return all;
  }, [queue, filter]);
  const item = items[Math.min(index, Math.max(items.length - 1, 0))];
  const position = item ? (queue?.items ?? []).indexOf(item) : -1;
  const previous = position > 0 ? queue?.items[position - 1] : undefined;

  useEffect(() => {
    if (!item) return setDraft(null);
    setDraft(drafts.current.get(key(item)) ?? proposal(item, previous, drafts.current));
    setHistory([]);
  }, [item?.cycle, item?.shot, item?.reviewed]); // eslint-disable-line react-hooks/exhaustive-deps

  const change = (next: Draft) => {
    if (!draft || !item) return;
    setHistory((h) => [...h.slice(-40), draft]);
    setDraft(next);
    drafts.current.set(key(item), next);
  };
  const undo = useCallback(() => {
    const last = history[history.length - 1];
    if (!last || !item) return;
    setDraft(last);
    drafts.current.set(key(item), last); // revisiting the frame shows what was on screen, not the undone shape
    setHistory((h) => h.slice(0, -1));
  }, [history, item]);

  const go = useCallback((delta: number) => setIndex((i) => Math.min(Math.max(i + delta, 0), Math.max(items.length - 1, 0))), [items.length]);

  const save = useCallback(
    (excluded: boolean) => {
      if (!item || !draft || busy) return;
      if (!excluded && !draft.piece) return notify("Draw the piece box first (P), or exclude the image (X)", true);
      setBusy(true);
      const body = { piece: draft.piece, defects: item.verdict === "bad" ? draft.defects : [], excluded };
      api<Draft>(`/api/inspection/runs/${encodeURIComponent(run)}/cycles/${item.cycle}/shots/${item.shot}/review`, { body })
        .then((saved) => {
          drafts.current.delete(key(item));
          setQueue((q) => {
            if (!q) return q;
            const was = q.items.find((i) => i.cycle === item.cycle && i.shot === item.shot);
            const next = q.items.map((i) => (i === was ? { ...i, ...saved, reviewed: true } : i));
            const reviewed = next.filter((i) => i.reviewed).length;
            return { ...q, items: next, reviewed, kept: next.filter((i) => i.reviewed && !i.excluded).length };
          });
          // "To review" drops the saved frame from the list: the same index is the next one.
          if (filter !== "todo") go(1);
        })
        .catch((error: Error) => notify(error.message, true))
        .finally(() => setBusy(false));
    },
    [item, draft, busy, run, notify, filter, go],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, select, textarea")) return;
      if ((event.ctrlKey || event.metaKey) && event.code === "KeyZ") {
        event.preventDefault();
        return undo();
      }
      const tools: Record<string, Tool> = { KeyP: "piece", KeyR: "rect", KeyB: "brush", KeyE: "erase" };
      if (tools[event.code]) return setTool(tools[event.code]!);
      if (event.code === "Enter") {
        event.preventDefault();
        save(false);
      } else if (event.code === "KeyX") save(true);
      else if (event.code === "ArrowRight") go(1);
      else if (event.code === "ArrowLeft") go(-1);
      else return;
      // These keys belong to the review: the live G / D / N verdict handler never sees them.
      event.stopPropagation();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [save, go, undo]);

  if (!queue) return <div className="faint">Loading the frames of {run}…</div>;
  const [fw, fh] = item?.frame_size ?? [640, 480];
  const pct = queue.total ? Math.round((100 * queue.reviewed) / queue.total) : 0;

  const point = (event: React.PointerEvent): [number, number] => {
    const rect = svgRef.current!.getBoundingClientRect();
    return [clamp((event.clientX - rect.left) / rect.width), clamp((event.clientY - rect.top) / rect.height)];
  };
  const down = (event: React.PointerEvent) => {
    if (!draft || busy) return;
    event.preventDefault();
    (event.target as Element).setPointerCapture?.(event.pointerId);
    const p = point(event);
    if (tool === "erase") {
      const keep = draft.defects.filter((d) => !hits(d, p, fw, fh));
      if (keep.length !== draft.defects.length) change({ ...draft, defects: keep });
      return;
    }
    if (tool !== "piece" && item?.verdict !== "bad") return notify("Defects are drawn on defect bricks only (fix the cycle verdict in the grid)", true);
    start.current = p;
    if (tool === "brush") setShape({ type: "stroke", points: [p], width: brush });
    else setShape({ type: tool === "piece" ? "piece" : "rect", box: [p[0], p[1], 0, 0] } as never);
  };
  const move = (event: React.PointerEvent) => {
    if (!shape || !start.current) return;
    const p = point(event);
    if (shape.type === "stroke") setShape({ ...shape, points: [...shape.points, p] });
    else {
      const [x0, y0] = start.current;
      setShape({ ...shape, box: [Math.min(x0, p[0]), Math.min(y0, p[1]), Math.abs(p[0] - x0), Math.abs(p[1] - y0)] });
    }
  };
  const up = () => {
    if (!shape || !draft) return;
    if (shape.type === "piece") {
      if (shape.box[2] > 0.02 && shape.box[3] > 0.02) change({ ...draft, piece: shape.box, excluded: false });
    } else if (shape.type === "rect") {
      if (shape.box[2] > 0.004 && shape.box[3] > 0.004) change({ ...draft, defects: [...draft.defects, shape], excluded: false });
    } else change({ ...draft, defects: [...draft.defects, shape], excluded: false });
    setShape(null);
    start.current = null;
  };

  return (
    <div className="review">
      <div className="row between wrap" style={{ gap: 8 }}>
        <div className="review-progress" title={`${queue.reviewed} of ${queue.total} frames reviewed, ${queue.kept} kept`}>
          <div className="review-bar">
            <div style={{ width: `${pct}%` }} />
          </div>
          <span>
            <b>{queue.reviewed}</b>/{queue.total} reviewed · {queue.kept} kept · {pct}%
          </span>
        </div>
        <div className="row">
          <div className="segmented" role="group" aria-label="Frames">
            {(["todo", "all", "bad"] as const).map((f) => (
              <button key={f} className="shadow" style={{ height: 26, fontSize: 12 }} aria-pressed={filter === f} onClick={() => { setFilter(f); setIndex(0); }}>
                {{ todo: "To review", all: "All frames", bad: "Defect bricks" }[f]}
              </button>
            ))}
          </div>
          <button className="btn sm" onClick={onClose}>
            Back to the grid
          </button>
        </div>
      </div>

      {!item || !draft ? (
        <div className="review-done">
          {filter === "todo" ? "Every frame of this run is reviewed. Download the zip or train a model below." : "No frame here."}
        </div>
      ) : (
        <>
          <div className="row between wrap" style={{ gap: 8, marginTop: 10 }}>
            <div>
              <b>Cycle {item.cycle}</b> · {item.camera} · frame {item.shot}{" "}
              <span className={`chip ${item.verdict === "bad" ? "bad" : item.verdict === "good" ? "good" : ""}`}>
                {item.verdict === "bad" ? "defect" : item.verdict ?? "no verdict"}
              </span>{" "}
              <span className="faint" style={{ fontSize: 12 }}>
                {item.reviewed ? (item.excluded ? "reviewed · excluded" : "reviewed") : "proposal"} · {item.usable ? "usable" : "flagged unusable"} ·{" "}
                {Math.min(index, items.length - 1) + 1}/{items.length}
              </span>
            </div>
            <div className="segmented" role="group" aria-label="Tool">
              {(
                [
                  ["piece", "Piece box", "P"],
                  ["rect", "Defect box", "R"],
                  ["brush", "Brush", "B"],
                  ["erase", "Erase", "E"],
                ] as const
              ).map(([k, label, hotkey]) => (
                <button
                  key={k}
                  className="shadow"
                  style={{ height: 28 }}
                  aria-pressed={tool === k}
                  disabled={k !== "piece" && item.verdict !== "bad"}
                  onClick={() => setTool(k)}
                  title={k !== "piece" && item.verdict !== "bad" ? "Defects only on defect bricks" : `${label} (${hotkey})`}
                >
                  {label} <kbd>{hotkey}</kbd>
                </button>
              ))}
            </div>
            {tool === "brush" && (
              <label className="row muted" style={{ fontSize: 12 }}>
                width
                <input type="range" min={0.005} max={0.08} step={0.005} value={brush} onChange={(e) => setBrush(Number(e.target.value))} />
              </label>
            )}
          </div>

          <div className={`annot-stage review-stage ${draft.excluded ? "excluded" : ""}`} style={{ aspectRatio: `${fw} / ${fh}` }}>
            <img src={inspectionFile(run, item.cycle, item.frame)} alt={`${item.camera} frame ${item.shot}`} draggable={false} />
            <svg ref={svgRef} viewBox={`0 0 ${fw} ${fh}`} preserveAspectRatio="none" className={`annot-svg tool-${tool}`} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
              {draft.piece && <BoxShape box={draft.piece} fw={fw} fh={fh} className="piece" />}
              {draft.defects.map((d, i) => (
                <Shape key={i} shape={d} fw={fw} fh={fh} />
              ))}
              {shape && (shape.type === "piece" ? <BoxShape box={shape.box} fw={fw} fh={fh} className="piece draft" /> : <Shape shape={shape} fw={fw} fh={fh} draft />)}
            </svg>
            {draft.excluded && <div className="review-excluded">Excluded</div>}
          </div>

          <div className="row wrap between" style={{ gap: 8, marginTop: 10 }}>
            <div className="row">
              <button className="btn sm" onClick={() => go(-1)} disabled={index <= 0}>
                ← <kbd>←</kbd>
              </button>
              <button className="btn sm" onClick={() => go(1)} disabled={index >= items.length - 1}>
                <kbd>→</kbd> →
              </button>
              <button className="btn sm" onClick={undo} disabled={!history.length} title="Undo (Ctrl+Z)">
                Undo
              </button>
              <button className="btn sm" onClick={() => change({ ...draft, piece: item.crop_box, excluded: false })} title="The automatic brick box of this frame">
                Auto piece box
              </button>
              {item.verdict === "bad" && (
                <button className="btn sm" onClick={() => change({ ...draft, defects: [] })} disabled={!draft.defects.length}>
                  Clear defects
                </button>
              )}
            </div>
            <div className="row">
              <button className="btn ghost" disabled={busy} onClick={() => save(true)} title="Blurry, empty, brick hidden: keep it out of the dataset">
                Exclude <kbd>X</kbd>
              </button>
              <button className="btn primary" disabled={busy || !draft.piece} onClick={() => save(false)}>
                Keep{item.verdict === "bad" ? ` · ${draft.defects.length} defect zone${draft.defects.length === 1 ? "" : "s"}` : ""} & next <kbd>Enter</kbd>
              </button>
            </div>
          </div>
          <div className="faint" style={{ fontSize: 11, marginTop: 4 }}>
            The yellow box is the only zone analysed: drag (P) to fit it tight around the brick. On a defect brick, mark everything wrong on THIS frame (R box, B
            brush, E erase); frames where no defect is visible are kept as “defect, no mask”. Space is still the e-stop.
          </div>
        </>
      )}
    </div>
  );
}
