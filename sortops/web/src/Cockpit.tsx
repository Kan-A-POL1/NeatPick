import { useEffect, useState } from "react";
import { Annotator } from "./Annotator";
import { ArmView } from "./ArmView";
import {
  api,
  type CameraAnnotation,
  type Config,
  type CycleRecord,
  type DecisionNote,
  fmt,
  type InspectionCycle,
  type InspectionVerdict,
  type Metrics,
  type ModelsResponse,
  type Shot,
  type Snapshot,
  type Transition,
} from "./api";
import { DeviceTag, Dot, LiveImage, Panel, Sparkline } from "./ui";

const STAGES = [
  { key: "pick_present", label: "Pick & present", role: "pick_present" },
  { key: "present", label: "Present · replay", role: null },
  { key: "inspect", label: "Inspect & score", role: "anomaly_wrist" },
  { key: "decide", label: "Decide", role: null },
  { key: "place", label: "Place in bin", role: null },
] as const;

const STAGE_COLORS: Record<string, string> = {
  pick_present: "oklch(0.72 0.13 235)",
  pick: "oklch(0.72 0.13 235)",
  present: "oklch(0.75 0.12 205)",
  inspect: "oklch(0.8 0.11 300)",
  recapture: "oklch(0.8 0.14 80)",
  reinspect: "oklch(0.8 0.14 80)",
  place_good: "oklch(0.76 0.14 165)",
  place_bad: "oklch(0.68 0.19 25)",
};

function activeStage(phase: string): string {
  if (phase === "pick_present" || phase === "pick") return "pick_present";
  if (phase === "inspect" || phase === "recapture" || phase === "reinspect" || phase === "await_operator") return "inspect";
  if (phase === "place_good" || phase === "place_bad") return "place";
  return phase;
}

interface Props {
  snapshot: Snapshot | null;
  metrics: Metrics | null;
  config: Config | null;
  models: ModelsResponse | null;
  notify: (message: string, error?: boolean) => void;
  refreshMetrics: () => void;
  refreshConfig: () => void;
}

export function Cockpit({ snapshot, metrics, config, refreshConfig, models, notify, refreshMetrics }: Props) {
  const [showGhost, setShowGhost] = useState(true);
  const [heat, setHeat] = useState(true);
  const robot = snapshot?.robot;
  const running = snapshot?.running ?? false;
  const stage = activeStage(snapshot?.phase ?? "idle");
  // With the present step on, the pick model only picks and a recorded episode presents the brick.
  const presenting = config?.present?.enabled ?? false;
  const stages = STAGES.filter((s) => presenting || s.key !== "present").map((s) =>
    presenting && s.key === "pick_present" ? { ...s, label: "Pick" } : s,
  );
  const devices = config?.devices;

  const act = (path: string, body: unknown = {}) =>
    api(path, { body })
      .then(() => refreshMetrics())
      .catch((error: Error) => notify(error.message, true));

  const lastCycle = metrics?.cycles.filter((c) => c.ended !== null).at(-1);
  const decision = snapshot?.last_decision;
  const cfgDecision = config?.decision;

  return (
    <div className="cockpit">
      {/* Column 1: the arm --------------------------------------------------------------------------- */}
      <div className="col">
        <Panel
          title="Cycle"
          actions={
            <div className="row">
              {running ? (
                <>
                  <button className="btn sm" onClick={() => act("/api/cycle/stop", { immediate: false })}>
                    Stop after this cycle
                  </button>
                  <button className="btn sm danger" onClick={() => act("/api/cycle/stop", { immediate: true })}>
                    Stop now
                  </button>
                </>
              ) : (
                <>
                  {snapshot?.released ? (
                    <button className="btn sm" title="Reconnect SortOps to the arm (close the Studio session first)" onClick={() => act("/api/robot/take")}>
                      Take the arm back
                    </button>
                  ) : (
                    <button
                      className="btn sm"
                      disabled={!snapshot?.connected || snapshot.simulated}
                      title="Close the arm port so Physical AI Studio can open it; the arm goes limp"
                      onClick={() => {
                        if (window.confirm("Release the arm for Physical AI Studio? Its motors go limp: hold it or park it first.")) {
                          act("/api/robot/release");
                        }
                      }}
                    >
                      Release arm for Studio
                    </button>
                  )}
                  <label className="chip" title="After the pick, replay the recorded presentation episode (Settings → Present)">
                    <input
                      type="checkbox"
                      checked={presenting}
                      disabled={!config}
                      onChange={(e) =>
                        api("/api/config", { method: "PATCH", body: { present: { enabled: e.target.checked } } })
                          .then(() => refreshConfig())
                          .catch((error: Error) => notify(error.message, true))
                      }
                    />{" "}
                    Present replay
                  </label>
                  <button
                    className="btn sm"
                    disabled={!snapshot?.connected || snapshot.estop.latched || snapshot.mode !== "live"}
                    title={snapshot?.mode !== "live" ? "Switch to Live to move the arm" : "Move the arm to its home pose"}
                    onClick={() => act("/api/home")}
                  >
                    ⌂ Home
                  </button>
                  <button className="btn sm" disabled={!snapshot?.connected || snapshot.estop.latched} onClick={() => act("/api/cycle/start", { loop: false })}>
                    One cycle
                  </button>
                  <button
                    className="btn sm primary"
                    disabled={!snapshot?.connected || snapshot.estop.latched}
                    onClick={() => act("/api/cycle/start", { loop: true })}
                  >
                    Start sorting
                  </button>
                  <button
                    className="btn sm primary"
                    disabled={!snapshot?.connected || snapshot.estop.latched}
                    title="Pick → present → inspect → place → home, again and again: never stops by itself (Stop to end)"
                    onClick={() => act("/api/cycle/start", { loop: true, forever: true })}
                  >
                    ∞ Loop
                  </button>
                </>
              )}
            </div>
          }
        >
          <div className="pipeline">
            {stages.map((item, index) => {
              const order = stages.findIndex((s) => s.key === stage);
              const isActive = running && item.key === stage;
              const done = running && order > index;
              const operator = item.key === "inspect" && snapshot?.verdict_source === "operator";
              const device = operator ? "you" : item.role ? devices?.[item.role] : item.key === "place" ? devices?.place_good : "CPU";
              const stageKey =
                item.key === "place"
                  ? snapshot?.phase === "place_bad"
                    ? "place_bad"
                    : "place_good"
                  : operator
                    ? "await_operator"
                    : item.key === "pick_present" && presenting
                      ? "pick"
                      : item.key;
              const elapsed = snapshot?.cycle?.stages[stageKey];
              return (
                <div key={item.key} className={`stage ${isActive ? "active" : ""} ${done ? "done" : ""}`}>
                  {device && <span className="device">{device}</span>}
                  <div className="name">{operator ? "Inspect · operator" : item.label}</div>
                  <div className="meta num">
                    {isActive ? `${snapshot?.phase_s.toFixed(1)} s` : elapsed !== undefined ? `${elapsed.toFixed(1)} s` : "—"}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="row wrap" style={{ marginTop: 10 }}>
            <span className="chip">
              Cycle <b className="num">{snapshot?.cycle?.id ?? lastCycle?.id ?? "—"}</b>
            </span>
            <span className="chip">{fmt.role(snapshot?.phase ?? "idle")}</span>
            {snapshot?.execution && (
              <span className="chip num" title="Chunk player of the running policy">
                {snapshot.execution.mode} · {snapshot.execution.hz} Hz · next at {Math.round(snapshot.execution.request_threshold * 100)} % · queue {snapshot.execution.queue}/{snapshot.execution.chunk} · {snapshot.execution.requests} chunks
              </span>
            )}
            {snapshot?.cycle && (
              <span className="chip num">
                frames {snapshot.cycle.frames_usable}/{snapshot.cycle.frames_seen} usable
              </span>
            )}
            {snapshot?.simulated && snapshot.piece && (
              <span className="chip" title="Simulation ground truth of the brick being handled">
                sim brick #{snapshot.piece.index} · {snapshot.piece.defective ? "defective" : "good"}
              </span>
            )}
          </div>
          {snapshot?.transition && <TransitionView transition={snapshot.transition} />}
          <DecisionLog
            notes={snapshot?.cycle?.decisions ?? lastCycle?.decisions ?? []}
            title={snapshot?.cycle ? `Cycle ${snapshot.cycle.id} · decisions` : lastCycle ? `Cycle ${lastCycle.id} · decisions` : "Decisions"}
          />
        </Panel>

        <Panel
          title="Arm · live"
          actions={
            <label className="row muted" style={{ fontSize: 12 }}>
              <input type="checkbox" checked={showGhost} onChange={(e) => setShowGhost(e.target.checked)} />
              Policy target
            </label>
          }
        >
          <div style={{ position: "relative" }}>
            <ArmView names={robot?.names} joints={robot?.joints} ghost={snapshot?.ghost} showGhost={showGhost} inspectionPose={config?.inspection.pose_xyz} />
            <div className="legend">
              <span>
                <i className="swatch" style={{ background: "#d9dde6" }} /> measured
              </span>
              <span>
                <i className="swatch" style={{ background: "#ffc35a", opacity: 0.6 }} /> policy target
              </span>
              <span>
                <i className="swatch" style={{ background: "#5fb4ff", opacity: 0.7, borderRadius: 5 }} /> inspection pose
              </span>
            </div>
          </div>
          <Joints snapshot={snapshot} />
        </Panel>
      </div>

      {/* Column 2: perception & decision ------------------------------------------------------------- */}
      <div className="col">
        <Panel title="Cameras">
          <div className="cams">
            {(config?.hardware.cameras ?? ["front", "overhead", "wrist"]).map((name, index) => (
              <div key={name} className={`cam ${index === 0 ? "main" : ""}`}>
                <LiveImage src={`/api/cams/${name}.jpg`} alt={`${name} camera`} />
                <span className="cam-label">
                  {name}
                  {config?.inspection.cameras.includes(name) ? " · inspection" : ""}
                </span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel
          title="Inspection"
          actions={
            <div className="row">
              <span className="faint" style={{ fontSize: 11 }}>
                Verdict
              </span>
              <div className="segmented" role="group" aria-label="Verdict source">
                {(
                  [
                    ["model", "anomalib", "Anomalib"],
                    ["cv", "algorithm", "Algorithm"],
                    ["operator", "manual", "Manual"],
                  ] as const
                ).map(([source, key, label]) => {
                  const opinion = decision?.opinions?.[key];
                  // Pastille: this method's verdict on the last brick; grey when it gave none (Manual not selected).
                  const color = !opinion
                    ? "var(--faint, #888)"
                    : opinion.verdict === "good"
                      ? "oklch(0.76 0.14 165)"
                      : opinion.verdict === "bad"
                        ? "oklch(0.68 0.19 25)"
                        : "oklch(0.8 0.14 80)";
                  return (
                    <button
                      key={source}
                      className="shadow"
                      style={{ height: 26, fontSize: 12 }}
                      aria-pressed={snapshot?.verdict_source === source}
                      disabled={snapshot?.awaiting_verdict}
                      title={opinion ? `${label}: ${opinion.verdict}${opinion.score != null ? ` (${opinion.score.toFixed(2)})` : ""}` : `${label}: no opinion`}
                      onClick={() =>
                        api("/api/config", { method: "PATCH", body: { decision: { verdict_source: source } } })
                          .then(() => notify(`${label} decides`))
                          .catch((error: Error) => notify(error.message, true))
                      }
                    >
                      <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 9, background: color, marginRight: 6 }} />
                      {label}
                    </button>
                  );
                })}
              </div>
              {snapshot?.verdict_source !== "operator" && (
                <label className="row muted" style={{ fontSize: 12 }}>
                  <input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} />
                  Anomaly map
                </label>
              )}
            </div>
          }
        >
          {snapshot?.awaiting_verdict && <OperatorPrompt cycle={snapshot.cycle?.id} inspection={snapshot.inspection} notify={notify} />}
          <div className="verdict with-crop">
            <div className="live-crops">
              {(snapshot?.inspection_cameras ?? config?.inspection.cameras ?? ["wrist"]).map((camera) => (
                <figure key={camera} className="live-crop" title={`Live: the ${camera} camera cut by the brick crop, exactly what its model scores. Red frame = no brick found.`}>
                  <LiveImage src={`/api/inspection/${camera}/crop.jpg`} alt={`Live brick crop of the ${camera} camera`} intervalMs={250} />
                  <figcaption>
                    {camera}
                    {decision?.camera_scores?.[camera] != null ? ` · ${decision.camera_scores[camera].toFixed(2)}` : " · live"}
                  </figcaption>
                </figure>
              ))}
            </div>
            <div className={`verdict-badge ${decision?.verdict ?? ""}`}>
              <div>
                <div className="big num">{decision?.score != null ? decision.score.toFixed(2) : "—"}</div>
                {decision ? (decision.verdict === "bad" ? "defect" : decision.verdict.replace("_", " ")) : "waiting"}
              </div>
            </div>
            <div>
              <div className="muted" style={{ fontSize: 12 }}>
                {decision ? `Cycle ${decision.cycle}: ${decision.reason}` : "No decision yet. Start a cycle."}
              </div>
              {snapshot?.verdict_source === "operator" ? (
                <div className="faint" style={{ fontSize: 11, marginTop: 8 }}>
                  Operator mode: no anomaly model, you decide each brick. Frames are still rated and kept.
                </div>
              ) : (
                <>
                  {cfgDecision && <ScoreScale scores={snapshot?.cycle?.scores ?? lastCycle?.scores ?? []} aggregate={decision?.score ?? null} good={cfgDecision.good_below} bad={cfgDecision.bad_above} />}
                  <div className="faint" style={{ fontSize: 11 }}>
                    {config &&
                `${config.inspection.aggregate} per camera · fusion ${config.decision.fusion} of ${config.inspection.cameras.join(" + ")} · gray zone → ${config.decision.recapture_frames} recaptures → defect bin`}
                  </div>
                </>
              )}
            </div>
          </div>
          <ShotStrip shots={metrics?.shots ?? []} heat={heat} />
        </Panel>

        <Panel title="Events">
          <div className="events">
            {(metrics?.events ?? [])
              .slice()
              .reverse()
              .map((event) => (
                <div key={`${event.t}-${event.msg}`} className={event.level}>
                  <span className="faint">{new Date(event.t * 1000).toLocaleTimeString()}</span> {event.msg}
                </div>
              ))}
          </div>
        </Panel>
      </div>

      {/* Column 3: evidence ------------------------------------------------------------------------- */}
      <div className="col">
        <Kpis metrics={metrics} />
        <Latency metrics={metrics} />
        <MotorTemps metrics={metrics} snapshot={snapshot} />
        <SystemPanel metrics={metrics} snapshot={snapshot} />
        <Accuracy metrics={metrics} />
      </div>

      <div className="span-all">
        <CycleTable cycles={metrics?.cycles ?? []} models={models} notify={notify} refresh={refreshMetrics} />
      </div>
    </div>
  );
}

/** Answer the brick waiting for a verdict; ``target`` names it so a late answer can never land on the next brick. */
/** What step 1 waits for before the inspection: every gate with its measured value and target, live. */
function TransitionView({ transition }: { transition: Transition }) {
  const { zone } = transition;
  const passed = transition.gates.filter((g) => g.ok).length;
  // Distance scale: the widest the zone can get, or the arm if it is farther.
  const scale = Math.max(zone.radius_m, zone.base_m * 1.5, zone.distance_m ?? 0) * 1.15;
  const pct = (m: number) => `${Math.min(100, (m / scale) * 100)}%`;
  const inside = zone.distance_m !== null && zone.distance_m <= zone.radius_m;
  return (
    <div className="transition" aria-label="Transition criteria">
      <div className="transition-head">
        <span className="transition-title">
          {fmt.role(transition.stage)} → {fmt.role(transition.next)}
        </span>
        <span className="faint num">
          attempt {transition.attempt}
          {transition.attempts ? `/${transition.attempts}` : ""} · {transition.elapsed_s.toFixed(1)} s
        </span>
        <div className="spacer" />
        <span className={`chip ${passed === transition.gates.length ? "good" : ""} num`}>
          {passed}/{transition.gates.length} criteria
        </span>
      </div>
      <ul className="gates">
        {transition.gates.map((gate) => (
          <li key={gate.key} className={gate.ok ? "ok" : ""}>
            <span className="gate-icon" aria-label={gate.ok ? "met" : "not met"}>
              {gate.ok ? "✓" : "…"}
            </span>
            <span className="gate-label">{gate.label}</span>
            <span className="gate-value num">{gate.value}</span>
            <span className="gate-target num">{gate.target}</span>
            <span className="gate-detail faint">{gate.detail}</span>
          </li>
        ))}
      </ul>
      <div className="zone" title="Distance of the gripper to the inspection pose, against the admission radius">
        <div className="zone-track">
          <i className="zone-fill" style={{ width: pct(zone.radius_m) }} />
          <i className="zone-base" style={{ left: pct(zone.base_m) }} />
          {zone.distance_m !== null && <i className={`zone-arm ${inside ? "in" : ""}`} style={{ left: pct(zone.distance_m) }} />}
        </div>
        <div className="zone-legend faint num">
          <span>
            zone {(zone.radius_m * 100).toFixed(1)} cm
            {zone.growth > 0 ? ` (base ${(zone.base_m * 100).toFixed(0)} cm +${Math.round(zone.growth * 100)} %)` : " (base)"}
          </span>
          <span>
            {zone.admissible} admissible frames
            {zone.next_growth_at !== null ? ` · next widening at ${zone.next_growth_at}` : " · widest"}
          </span>
        </div>
      </div>
      <div className="timers">
        {transition.timers.map((timer) => {
          const fraction = timer.elapsed === null ? 0 : Math.min(1, timer.elapsed / timer.limit);
          return (
            <div key={timer.label} className={`timer ${timer.active ? "active" : ""}`}>
              <span>{timer.label}</span>
              <div className="timer-bar">
                <i style={{ width: `${fraction * 100}%`, background: fraction > 0.75 ? "var(--warn)" : "var(--accent)" }} />
              </div>
              <span className="num faint">
                {timer.elapsed === null ? "—" : timer.elapsed.toFixed(1)} / {timer.limit.toFixed(0)} s
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Every transition of the cycle and why, in order. */
function DecisionLog({ notes, title }: { notes: DecisionNote[]; title: string }) {
  if (notes.length === 0) return null;
  return (
    <div className="decision-log" aria-label={title}>
      <div className="faint" style={{ fontSize: 11, marginBottom: 4 }}>
        {title}
      </div>
      <ol>
        {notes.map((note, index) => (
          <li key={`${note.t}-${index}`} className={note.ok ? "ok" : "ko"}>
            <span className="num faint">+{note.t.toFixed(1)} s</span>
            <span className="step">
              {fmt.role(note.stage)} → {fmt.role(note.next)}
            </span>
            <span className="why">{note.text}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function sendVerdict(
  verdict: InspectionVerdict,
  target: { run: string; cycle: number } | null,
  notify: (m: string, e?: boolean) => void,
  annotation?: { cameras: Record<string, CameraAnnotation>; note: string },
) {
  if (!target) return Promise.resolve(notify("No brick is waiting for a verdict", true));
  return api("/api/verdict", { body: { verdict, run: target.run, cycle: target.cycle, annotation: annotation ?? null } }).catch(
    (error: Error) => notify(error.message, true),
  );
}

function OperatorPrompt({
  cycle,
  inspection,
  notify,
}: {
  cycle: number | undefined;
  inspection: { run: string; cycle: number } | null;
  notify: (m: string, e?: boolean) => void;
}) {
  // The loaded detail is tagged with the brick it belongs to: the annotator of a previous brick is never shown
  // (nor submitted) for the next one while its detail loads.
  const [loaded, setLoaded] = useState<{ key: string; detail: InspectionCycle | null } | null>(null);
  const [sending, setSending] = useState(false);
  const key = inspection ? `${inspection.run}#${inspection.cycle}` : "";
  const detail = loaded && loaded.key === key ? loaded.detail : null;
  const loading = !!inspection && loaded?.key !== key;
  useEffect(() => {
    setLoaded(null);
    if (!inspection) return;
    let stop = false;
    const tag = `${inspection.run}#${inspection.cycle}`;
    api<InspectionCycle>(`/api/inspection/runs/${encodeURIComponent(inspection.run)}/cycles/${inspection.cycle}`)
      .then((d) => !stop && setLoaded({ key: tag, detail: d }))
      .catch(() => !stop && setLoaded({ key: tag, detail: null }));
    return () => {
      stop = true;
    };
  }, [inspection?.run, inspection?.cycle]); // eslint-disable-line react-hooks/exhaustive-deps

  const answer = (verdict: InspectionVerdict, annotation?: { cameras: Record<string, CameraAnnotation>; note: string }) => {
    if (sending || loading) return;
    setSending(true);
    sendVerdict(verdict, inspection, notify, annotation).finally(() => setSending(false));
  };

  return (
    <div className="prompt annotate" role="alertdialog" aria-label="Operator verdict">
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontWeight: 600 }}>Is the presented brick good? Show the piece and, if defective, where.</div>
        <div className="muted" style={{ fontSize: 12 }}>
          Cycle {cycle ?? "—"} · the arm holds it in front of the camera until you answer · saved with the run (Data tab)
        </div>
      </div>
      {detail && Object.keys(detail.best).length ? (
        <Annotator detail={detail} purpose="verdict" busy={sending} onSubmit={(verdict, annotation) => answer(verdict, annotation)} />
      ) : loading ? (
        <div className="faint" style={{ fontSize: 12 }}>Loading this brick's frames…</div>
      ) : (
        <div className="row">
          <span className="faint" style={{ fontSize: 12 }}>No inspection frame stored for this brick: verdict only.</span>
          <button className="btn verdict-good" onClick={() => answer("good")}>
            ✓ Good <kbd>G</kbd>
          </button>
          <button className="btn verdict-bad" onClick={() => answer("bad")}>
            ✗ Defect <kbd>D</kbd>
          </button>
          <button className="btn ghost" onClick={() => answer("no_piece")}>
            No brick <kbd>N</kbd>
          </button>
        </div>
      )}
    </div>
  );
}

function Joints({ snapshot }: { snapshot: Snapshot | null }) {
  const robot = snapshot?.robot;
  if (!robot) return <div className="faint" style={{ marginTop: 10 }}>Waiting for the arm…</div>;
  return (
    <div className="joints" style={{ marginTop: 10 }}>
      {robot.names.map((name, index) => {
        const value = robot.joints[index] ?? 0;
        const target = snapshot?.ghost?.[index];
        const gripper = name === "gripper";
        const toPct = (v: number) => (gripper ? v : (v + 100) / 2);
        return (
          <div key={name} style={{ display: "contents" }}>
            <span className="muted">{name.replace("_", " ")}</span>
            <div className="bar">
              <i style={gripper ? { left: 0, width: `${value}%` } : { left: `${50 + Math.min(0, value) / 2}%`, width: `${Math.abs(value) / 2}%` }} />
              {target !== undefined && <b style={{ left: `${Math.min(100, Math.max(0, toPct(target)))}%` }} />}
            </div>
            <span className="mono num" style={{ textAlign: "right", minWidth: 44 }}>
              {value.toFixed(1)}
            </span>
            <span className="mono num faint" style={{ textAlign: "right", minWidth: 40 }} title="motor temperature">
              {robot.temp[index]?.toFixed(0)}°
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ScoreScale({ scores, aggregate, good, bad }: { scores: number[]; aggregate: number | null; good: number; bad: number }) {
  const pct = (v: number) => `${Math.min(100, Math.max(0, v * 100))}%`;
  return (
    <div>
      <div className="scale" style={{ "--g": pct(good), "--b": pct(bad) } as React.CSSProperties}>
        <div className="track" />
        {scores.map((score, index) => (
          <span key={index} className="tick" style={{ left: pct(score), opacity: 0.55 }} title={`frame score ${score.toFixed(3)}`} />
        ))}
        {aggregate !== null && <span className="tick agg" style={{ left: pct(aggregate) }} title={`decision score ${aggregate.toFixed(3)}`} />}
      </div>
      <div className="scale-labels">
        <span>0 · good</span>
        <span>
          {good.toFixed(2)} — {bad.toFixed(2)} gray zone
        </span>
        <span>defect · 1</span>
      </div>
    </div>
  );
}

function ShotStrip({ shots, heat }: { shots: Shot[]; heat: boolean }) {
  if (shots.length === 0) return <div className="faint" style={{ marginTop: 12 }}>Inspection frames appear here when the arm presents a brick.</div>;
  const latestCycle = shots[shots.length - 1]?.cycle;
  const visible = shots.filter((shot) => shot.cycle === latestCycle).slice(-12);
  return (
    <div style={{ marginTop: 12 }}>
      <div className="row between faint" style={{ fontSize: 11, marginBottom: 6 }}>
        <span>Cycle {latestCycle} · brick crops, one per period, rated before scoring</span>
        <span>
          {visible.filter((s) => s.usability.usable).length}/{visible.length} usable
        </span>
      </div>
      <div className="shots">
        {visible.map((shot) => (
          <figure
            key={shot.id}
            className={`shot ${shot.usability.usable ? "" : "unusable"}`}
            style={{ margin: 0 }}
            title={`${shot.camera} · ${shot.phase} · usability ${shot.usability.score} (${shot.usability.reason}) · sharpness ${shot.usability.sharpness} · pose error ${(shot.usability.pose_error_m * 100).toFixed(1)} cm`}
          >
            <img src={`/api/shots/${shot.id}.jpg${heat && shot.has_heat ? "?heat=true" : ""}`} alt={`inspection frame ${shot.id}`} loading="lazy" />
            <figcaption className="tag">
              <span style={{ color: shot.usability.usable ? "var(--text-2)" : "var(--warn)" }}>
                {shot.camera[0]?.toUpperCase()} {shot.usability.usable ? `u ${shot.usability.score.toFixed(2)}` : shot.usability.reason}
              </span>
              {shot.score !== null && <span style={{ color: shot.score >= 0.5 ? "var(--bad)" : "var(--good)" }}>{shot.score.toFixed(2)}</span>}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

function Kpis({ metrics }: { metrics: Metrics | null }) {
  const stats = metrics?.stats;
  return (
    <Panel
      title="Run"
      actions={
        <div className="row">
          <a className="btn sm ghost" href="/api/runs/export.csv">
            CSV
          </a>
          <a className="btn sm ghost" href="/api/report.json">
            Report
          </a>
        </div>
      }
    >
      <div className="kpis">
        <div className="kpi">
          <div className="label">Sorted</div>
          <div className="value num">{stats?.sorted ?? 0}</div>
          <div className="sub num">
            {stats?.good ?? 0} good · {stats?.bad ?? 0} defect
          </div>
        </div>
        <div className="kpi">
          <div className="label">Accuracy</div>
          <div className="value num">{fmt.pct(stats?.accuracy)}</div>
          <div className="sub num">{stats?.labelled ?? 0} labelled</div>
        </div>
        <div className="kpi">
          <div className="label">Cycle time</div>
          <div className="value num">
            {fmt.num(stats?.cycle_s_mean)}
            <small>s</small>
          </div>
          <div className="sub num">p95 {fmt.num(stats?.cycle_s_p95)} s</div>
        </div>
        <div className="kpi">
          <div className="label">Throughput</div>
          <div className="value num">
            {fmt.num(stats?.throughput_per_min)}
            <small>/min</small>
          </div>
          <div className="sub">bricks sorted</div>
        </div>
        <div className="kpi">
          <div className="label">Energy / cycle</div>
          <div className="value num">
            {fmt.num(stats?.energy_j_mean, 0)}
            <small>J</small>
          </div>
          <div className="sub">package, RAPL</div>
        </div>
        <div className="kpi">
          <div className="label">Recoveries</div>
          <div className="value num">{stats?.recaptured ?? 0}</div>
          <div className="sub num">{stats?.failed ?? 0} failed/aborted</div>
        </div>
      </div>
    </Panel>
  );
}

function Latency({ metrics }: { metrics: Metrics | null }) {
  const rows = ["pick_present", "anomaly_front", "anomaly_wrist", "place_good", "place_bad"] as const;
  return (
    <Panel title="Inference latency">
      <div className="lat-row faint" style={{ borderTop: 0, paddingTop: 0, fontSize: 11 }}>
        <span>Model</span>
        <span />
        <span>last 60</span>
        <span className="v">p50</span>
        <span className="v">p95</span>
      </div>
      {rows.map((key) => {
        const stat = metrics?.latency[key];
        return (
          <div key={key} className="lat-row">
            <span>{fmt.role(key)}</span>
            {stat ? <DeviceTag device={stat.device} /> : <span />}
            <Sparkline values={metrics?.latency_recent[key] ?? []} />
            <span className="v">{fmt.ms(stat?.p50)}</span>
            <span className="v faint">{fmt.ms(stat?.p95)}</span>
          </div>
        );
      })}
      <div className="faint" style={{ fontSize: 11, marginTop: 6 }}>
        ms per call · ACT returns a 100-step action chunk per call
      </div>
    </Panel>
  );
}

function SystemPanel({ metrics, snapshot }: { metrics: Metrics | null; snapshot: Snapshot | null }) {
  const series = metrics?.system ?? [];
  const latest = snapshot?.system;
  const items = [
    { label: "Package power", unit: "W", value: latest?.package_w, values: series.map((s) => s.package_w), color: "oklch(0.8 0.14 80)" },
    { label: "Package temp", unit: "°C", value: latest?.package_temp_c, values: series.map((s) => s.package_temp_c), color: "oklch(0.68 0.19 25)", max: 105 },
    { label: "CPU", unit: "%", value: latest?.cpu_pct, values: series.map((s) => s.cpu_pct), color: "oklch(0.72 0.13 235)", max: 100 },
    { label: "iGPU", unit: "%", value: latest?.gpu_pct, values: series.map((s) => s.gpu_pct), color: "oklch(0.72 0.13 235)", max: 100 },
    { label: "NPU", unit: "%", value: latest?.npu_pct, values: series.map((s) => s.npu_pct), color: "oklch(0.8 0.11 300)", max: 100 },
    { label: "RAM", unit: "%", value: latest?.ram_pct, values: series.map((s) => s.ram_pct), color: "oklch(0.76 0.14 165)", max: 100 },
  ];
  return (
    <Panel title="Core Ultra · platform">
      <div className="sys-grid">
        {items.map((item) => (
          <div key={item.label} className="sys-item">
            <div className="top">
              <span>{item.label}</span>
              <b className="num">
                {item.value == null ? "n/a" : item.value.toFixed(item.unit === "W" ? 1 : 0)}
                {item.value == null ? "" : ` ${item.unit}`}
              </b>
            </div>
            <Sparkline values={item.values.slice(-90)} color={item.color} max={item.max} />
          </div>
        ))}
      </div>
    </Panel>
  );
}

function MotorTemps({ metrics, snapshot }: { metrics: Metrics | null; snapshot: Snapshot | null }) {
  const names = snapshot?.robot?.names ?? ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
  const limit = snapshot?.temp_limit ?? 65;
  const history = (metrics?.motor_temps ?? []).slice(-120);
  const hottest = Math.max(...(snapshot?.temp_median ?? [0]));
  return (
    <Panel
      title="Motor temperatures"
      actions={
        <span className="faint" style={{ fontSize: 11 }}>
          median 5 s · stop at {limit.toFixed(0)} °C
        </span>
      }
    >
      <div className="temps">
        {names.map((name, index) => {
          const median = snapshot?.temp_median?.[index];
          const raw = snapshot?.robot?.temp[index];
          const state = median == null ? "" : median >= limit ? "bad" : median >= limit - 10 ? "warn" : "";
          return (
            <div key={name} className="temp-row">
              <span className="muted">{name.replace("_", " ")}</span>
              <Sparkline values={history.map((h) => h.median[index] ?? null)} color={state === "bad" ? "var(--bad)" : state === "warn" ? "var(--warn)" : "oklch(0.76 0.14 165)"} min={20} max={Math.max(limit + 5, hottest + 5)} threshold={limit} />
              <b className={`mono num ${state}`} title={raw != null ? `last read ${raw.toFixed(0)} °C` : undefined}>
                {median == null ? "—" : `${median.toFixed(0)}°`}
              </b>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

function Accuracy({ metrics }: { metrics: Metrics | null }) {
  const stats = metrics?.stats;
  const c = stats?.confusion ?? { tp: 0, tn: 0, fp: 0, fn: 0 };
  return (
    <Panel title="Detection quality">
      <div className="row" style={{ alignItems: "flex-start", gap: 14 }}>
        <div className="confusion" style={{ flex: 1 }}>
          <span />
          <span className="h">truth good</span>
          <span className="h">truth defect</span>
          <span className="h">sorted good</span>
          <span className="cell ok num">{c.tn}</span>
          <span className="cell ko num">{c.fn}</span>
          <span className="h">sorted defect</span>
          <span className="cell ko num">{c.fp}</span>
          <span className="cell ok num">{c.tp}</span>
        </div>
        <div className="stack" style={{ gap: 4, fontSize: 12, minWidth: 110 }}>
          <span className="row between">
            <span className="muted">Precision</span> <b className="mono">{fmt.pct(stats?.precision)}</b>
          </span>
          <span className="row between">
            <span className="muted">Recall</span> <b className="mono">{fmt.pct(stats?.recall)}</b>
          </span>
          <span className="faint" style={{ fontSize: 11 }}>
            Label each sorted brick below (✓ good / ✗ defect).
          </span>
        </div>
      </div>
    </Panel>
  );
}

function CycleTable({ cycles, models, notify, refresh }: { cycles: CycleRecord[]; models: ModelsResponse | null; notify: (m: string, e?: boolean) => void; refresh: () => void }) {
  const label = (cycle: CycleRecord, truth: "good" | "bad") =>
    api(`/api/cycles/${cycle.id}/label`, { body: { truth: cycle.truth === truth ? null : truth } })
      .then(refresh)
      .catch((error: Error) => notify(error.message, true));
  const recent = cycles.slice(-12).reverse();
  return (
    <Panel
      title="Cycles"
      actions={
        <span className="faint" style={{ fontSize: 11 }}>
          models: {(["pick_present", "place_good", "place_bad", "anomaly_front", "anomaly_wrist"] as const).map((role) => models?.active[role]?.name ?? "sim").join(" · ")}
        </span>
      }
    >
      {recent.length === 0 ? (
        <div className="faint">No cycle yet. Put a brick on the table and press Start sorting.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Outcome</th>
              <th className="r">Score</th>
              <th className="r">Frames</th>
              <th>Stages</th>
              <th className="r">Cycle</th>
              <th className="r">Energy</th>
              <th>Reason</th>
              <th>Truth</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((cycle) => {
              const total = Object.values(cycle.stages).reduce((a, b) => a + b, 0) || 1;
              const outcomeClass = cycle.outcome === "good" ? "good" : cycle.outcome === "bad" ? "bad" : cycle.outcome === "running" ? "accent" : "warn";
              return (
                <tr key={`${cycle.run}-${cycle.id}`}>
                  <td className="mono">{cycle.id}</td>
                  <td>
                    <span className={`chip ${outcomeClass}`}>
                      {cycle.outcome === "bad" ? "defect" : cycle.outcome.replace("_", " ")}
                      {cycle.recaptures > 0 ? " · recaptured" : ""}
                    </span>
                  </td>
                  <td className="r mono">{cycle.score?.toFixed(2) ?? "—"}</td>
                  <td className="r mono">
                    {cycle.frames_usable}/{cycle.frames_seen}
                  </td>
                  <td>
                    <div className="stagebar" title={Object.entries(cycle.stages).map(([k, v]) => `${fmt.role(k)} ${v.toFixed(1)} s`).join(" · ")}>
                      {Object.entries(cycle.stages).map(([key, value]) => (
                        <i key={key} style={{ width: `${(value / total) * 100}%`, background: STAGE_COLORS[key] ?? "var(--text-3)" }} />
                      ))}
                    </div>
                  </td>
                  <td className="r mono">{cycle.duration != null ? `${cycle.duration.toFixed(1)} s` : "—"}</td>
                  <td className="r mono">{cycle.energy_j != null ? `${cycle.energy_j.toFixed(0)} J` : "—"}</td>
                  <td
                    className="muted"
                    style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                    title={
                      (cycle.decisions ?? []).map((d) => `+${d.t.toFixed(1)} s ${fmt.role(d.stage)} → ${fmt.role(d.next)}: ${d.text}`).join("\n") ||
                      (cycle.error ?? cycle.decision_reason)
                    }
                  >
                    {(cycle.pick_attempts ?? 1) > 1 && <span className="chip warn" style={{ marginRight: 6 }}>{cycle.pick_attempts} picks</span>}
                    {cycle.error ?? cycle.decision_reason}
                  </td>
                  <td>
                    {cycle.outcome === "good" || cycle.outcome === "bad" ? (
                      <span className="label-btns">
                        <button className="g" aria-pressed={cycle.truth === "good"} aria-label={`Cycle ${cycle.id} was a good brick`} onClick={() => label(cycle, "good")}>
                          ✓
                        </button>
                        <button className="b" aria-pressed={cycle.truth === "bad"} aria-label={`Cycle ${cycle.id} was a defective brick`} onClick={() => label(cycle, "bad")}>
                          ✗
                        </button>
                      </span>
                    ) : (
                      <span className="faint">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

export function StatusChips({ snapshot, online }: { snapshot: Snapshot | null; online: boolean }) {
  if (!online || !snapshot) {
    return (
      <span className="chip bad">
        <Dot state="err" /> backend offline
      </span>
    );
  }
  return (
    <>
      <span className="chip" title={snapshot.connect_error ?? undefined}>
        <Dot state={snapshot.connected ? "ok" : snapshot.released ? "warn" : "err"} />
        {snapshot.simulated ? "Simulation" : snapshot.released ? "SO-101 released" : "SO-101"}
      </span>
      {snapshot.read_only && <span className="chip warn" title="hardware.arm_access = read_only: models run, motors are never written">read-only arm</span>}
      {!snapshot.torque && <span className="chip bad">torque off</span>}
      {snapshot.busy && <span className="chip accent">{snapshot.busy}…</span>}
    </>
  );
}
