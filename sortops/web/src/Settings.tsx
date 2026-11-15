import { useEffect, useState } from "react";
import { api, type Config, INSPECTION_CAMERAS, type Snapshot } from "./api";
import { Panel } from "./ui";

type Section = "decision" | "inspection" | "safety" | "release" | "present" | "pick" | "run";

interface FieldSpec {
  key: string;
  label: string;
  hint?: string;
  step?: number;
  kind?: "number" | "bool" | "select" | "text";
  options?: string[];
}

const SECTIONS: { section: Section; title: string; fields: FieldSpec[] }[] = [
  {
    section: "decision",
    title: "Decision policy",
    fields: [
      { key: "verdict_source", label: "Verdict from", kind: "select", options: ["model", "cv", "operator"], hint: "model = Anomalib, cv = Algorithm (stains / dark marks threshold), operator = Manual" },
      { key: "good_below", label: "Good if score ≤", step: 0.01, hint: "Anomalib normalizes scores; 0.5 is its own threshold" },
      { key: "bad_above", label: "Defect if score ≥", step: 0.01, hint: "Between the two: gray zone" },
      { key: "recapture_frames", label: "Recapture frames", step: 1, hint: "Fresh frames taken when uncertain" },
      { key: "recapture_period_s", label: "Recapture period (s)", step: 0.05 },
      { key: "reinspect_enabled", label: "Re-inspect from another angle", kind: "bool", hint: "Needs a re-inspect model" },
    ],
  },
  {
    section: "inspection",
    title: "Inspection",
    fields: [
      { key: "camera", label: "Capture-app camera", kind: "select", options: ["wrist", "front"], hint: "Its search zone (roi) is the one the capture app edits" },
      { key: "period_s", label: "Frame period (s)", step: 0.05 },
      { key: "window_s", label: "Window from 1st frame (s)", step: 0.1, hint: "Covers the 180° wrist turn" },
      { key: "pose_tolerance_m", label: "Pose tolerance (m)", step: 0.005 },
      { key: "best_frame_only", label: "One frame per camera", kind: "bool", hint: "Only the most usable frame of each camera is scored and kept" },
      { key: "min_usable_frames", label: "Min usable frames", step: 1, hint: "When one frame per camera is off" },
      { key: "usability_threshold", label: "Usability threshold", step: 0.05 },
      { key: "min_sharpness", label: "Min sharpness (Laplacian var.)", step: 5 },
      { key: "min_brick_fraction", label: "Min brick share of ROI", step: 0.005 },
      { key: "max_speed_for_capture", label: "Max arm speed for capture", step: 0.5 },
      { key: "aggregate", label: "Aggregate scores", kind: "select", options: ["p90", "max", "mean", "p75"] },
    ],
  },
  {
    section: "pick",
    title: "Step 1 → inspection",
    fields: [
      { key: "require_grasp", label: "Require a held brick", kind: "bool", hint: "No grasp read on the gripper: no frame, no inspection" },
      { key: "min_load", label: "Grasp: gripper load ≥", step: 1, hint: "Holding a brick reads 17-20, closed on nothing 2-5" },
      { key: "min_closed_pos", label: "Grasp: gripper ≥ (not fully closed)", step: 0.5, hint: "Holding reads ~3, fully closed ~1" },
      { key: "open_below", label: "Grasp: gripper < (closed)", step: 1 },
      { key: "hold_s", label: "Grasp confirmed after (s)", step: 0.05 },
      { key: "lost_s", label: "Brick lost after (s) without grip", step: 0.05 },
      { key: "min_admissible_in_zone", label: "Admissible frames in zone", step: 1, hint: "A brick seen by an inspection camera, arm in the admission zone" },
      { key: "zone_step_frames", label: "Widen the zone every (frames)", step: 1, hint: "Admissible frames seen anywhere" },
      { key: "zone_growth", label: "… by (0.5 = +50 %)", step: 0.05 },
      { key: "zone_max_growth", label: "Widest (2 = +200 %)", step: 0.1 },
      { key: "present_max_s", label: "Held, inspect anyway after (s)", step: 1 },
      { key: "retry_wait_s", label: "No brick: wait at home (s)", step: 0.1, hint: "Time to grasp = segments.pick_present.timeout_s" },
    ],
  },
  {
    section: "run",
    title: "Run",
    fields: [
      { key: "pick_attempts", label: "Pick attempts per brick", step: 1, hint: "Home between two; 0 = no limit" },
      { key: "stop_after_no_brick", label: "Stop after N cycles without brick", step: 1 },
    ],
  },
  {
    section: "release",
    title: "Release reflex (place)",
    fields: [
      { key: "enabled", label: "Release reflex", kind: "bool", hint: "When the place model starts opening over the bowl, SortOps opens wide and moves on" },
      { key: "place_good", label: "On place good", kind: "bool" },
      { key: "place_bad", label: "On place bad", kind: "bool" },
      { key: "trigger", label: "Trigger at gripper command ≥", step: 1, hint: "0 = closed, 100 = open; the demos open to ~20-28" },
      { key: "open", label: "Open to", step: 5 },
      { key: "open_tolerance", label: "Counts as open within", step: 1 },
      { key: "hold_s", label: "Wait once open (s)", step: 0.05, hint: "Time for the brick to fall" },
      { key: "max_open_s", label: "Give up opening after (s)", step: 0.1 },
      { key: "min_s", label: "Not before (s into the segment)", step: 0.1 },
    ],
  },
  {
    section: "present",
    title: "Present (replay)",
    fields: [
      { key: "enabled", label: "Present step", kind: "bool", hint: "After the pick, replay a recorded episode as is (no model), then inspect and place" },
      { key: "dataset", label: "Studio dataset", kind: "text", hint: "Name of the dataset holding the presentation episode" },
      { key: "episode", label: "Episode", step: 1 },
      { key: "approach_speed", label: "Approach speed (units/s)", step: 5, hint: "Glide from the end of the pick to the episode's first pose" },
      { key: "playback_speed", label: "Playback speed", step: 0.05, hint: "1 = as recorded, up to 3" },
      { key: "pause_epsilon", label: "Skip pauses below (units)", step: 0.1, hint: "Frames where no joint moves this much are skipped; 0 = replay every frame" },
      { key: "pick_open_above", label: "Pick: gripper opened above", step: 1 },
      { key: "pick_closed_below", label: "Pick: then closed below", step: 1, hint: "Holding the brick reads ~3" },
      { key: "pick_closed_s", label: "Pick: closed for (s)", step: 0.1, hint: "Then the pick ends and the replay starts" },
    ],
  },
  {
    section: "safety",
    title: "Safety",
    fields: [
      { key: "max_step", label: "Max joint step per tick", step: 0.5, hint: "Normalized units; ×fps = speed limit" },
      { key: "max_motor_temp_c", label: "E-stop above motor °C", step: 1, hint: "On the per-motor median; 30 to 120" },
      { key: "temp_window_s", label: "Temperature median window (s)", step: 1 },
      { key: "watchdog_s", label: "Watchdog timeout (s)", step: 0.1 },
      { key: "watchdog_enabled", label: "Watchdog in live mode", kind: "bool", hint: "E-stop if this page stops answering" },
    ],
  },
];

type Values = Record<string, number | boolean | string>;

function SectionForm({ title, section, fields, config, notify, onSaved }: { title: string; section: Section; fields: FieldSpec[]; config: Config; notify: (m: string, e?: boolean) => void; onSaved: () => void }) {
  const source = config[section] as unknown as Values;
  const [draft, setDraft] = useState<Values>(source);
  useEffect(() => setDraft(config[section] as unknown as Values), [config, section]);
  const dirty = fields.some((field) => draft[field.key] !== source[field.key]);

  const save = () =>
    api("/api/config", { method: "PATCH", body: { [section]: Object.fromEntries(fields.map((f) => [f.key, draft[f.key]])) } })
      .then(() => {
        notify(`${title} saved`);
        onSaved();
      })
      .catch((error: Error) => notify(error.message, true));

  return (
    <Panel
      title={title}
      actions={
        <button className="btn sm primary" disabled={!dirty} onClick={save}>
          Save
        </button>
      }
    >
      <div className="form-grid">
        {fields.map((field) => {
          const id = `${section}-${field.key}`;
          const value = draft[field.key];
          return (
            <div key={field.key} className="field">
              <label htmlFor={id}>{field.label}</label>
              {field.kind === "bool" ? (
                <select id={id} value={String(value)} onChange={(e) => setDraft({ ...draft, [field.key]: e.target.value === "true" })}>
                  <option value="true">On</option>
                  <option value="false">Off</option>
                </select>
              ) : field.kind === "select" ? (
                <select id={id} value={String(value)} onChange={(e) => setDraft({ ...draft, [field.key]: e.target.value })}>
                  {field.options?.map((option) => (
                    <option key={option}>{option}</option>
                  ))}
                </select>
              ) : field.kind === "text" ? (
                <input id={id} type="text" value={String(value)} onChange={(e) => setDraft({ ...draft, [field.key]: e.target.value })} />
              ) : (
                <input
                  id={id}
                  type="number"
                  step={field.step}
                  value={String(value)}
                  onChange={(e) => setDraft({ ...draft, [field.key]: e.target.value === "" ? "" : Number(e.target.value) })}
                />
              )}
              {field.hint && <span className="hint">{field.hint}</span>}
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

function CameraFusion({ config, notify, onSaved }: { config: Config; notify: (m: string, e?: boolean) => void; onSaved: () => void }) {
  const [cameras, setCameras] = useState<string[]>(config.inspection.cameras);
  const [fusion, setFusion] = useState(config.decision.fusion);
  const [weights, setWeights] = useState<Record<string, number>>(config.decision.weights);
  const [inputs, setInputs] = useState<Record<string, "crop" | "frame">>(config.inspection.model_input);
  useEffect(() => {
    setInputs(config.inspection.model_input);
    setCameras(config.inspection.cameras);
    setFusion(config.decision.fusion);
    setWeights(config.decision.weights);
  }, [config]);
  const dirty =
    cameras.join() !== config.inspection.cameras.join() ||
    fusion !== config.decision.fusion ||
    INSPECTION_CAMERAS.some((cam) => weights[cam] !== config.decision.weights[cam] || inputs[cam] !== config.inspection.model_input[cam]);
  const save = () =>
    api("/api/config", { method: "PATCH", body: { inspection: { cameras }, decision: { fusion, weights } } })
      .then(() => {
        notify(`Inspection: ${cameras.join(" + ")}, fusion ${fusion}`);
        onSaved();
      })
      .catch((error: Error) => notify(error.message, true));
  return (
    <Panel
      title="Inspection cameras & fusion"
      actions={
        <button className="btn sm primary" disabled={!dirty || cameras.length === 0} onClick={save}>
          Save
        </button>
      }
    >
      <div className="stack">
        <div className="row">
          {INSPECTION_CAMERAS.map((cam) => (
            <label key={cam} className="row muted" style={{ fontSize: 12 }}>
              <input
                type="checkbox"
                checked={cameras.includes(cam)}
                onChange={(e) => setCameras(e.target.checked ? INSPECTION_CAMERAS.filter((c) => c === cam || cameras.includes(c)) : cameras.filter((c) => c !== cam))}
              />
              {cam} (its own model)
            </label>
          ))}
        </div>
        <div className="form-grid">
          {INSPECTION_CAMERAS.map((cam) => (
            <div key={cam} className="field">
              <label htmlFor={`in-${cam}`}>{cam} model sees</label>
              <select id={`in-${cam}`} value={inputs[cam] ?? "crop"} onChange={(e) => setInputs({ ...inputs, [cam]: e.target.value as "crop" | "frame" })}>
                <option value="crop">the brick crop</option>
                <option value="frame">the whole image</option>
              </select>
              <span className="hint">Match how its training photos were taken</span>
            </div>
          ))}
          <div className="field">
            <label htmlFor="fusion">Defect if</label>
            <select id="fusion" value={fusion} onChange={(e) => setFusion(e.target.value as Config["decision"]["fusion"])}>
              <option value="max">any camera sees one (max)</option>
              <option value="mean">the average says so (mean)</option>
              <option value="min">every camera sees one (min)</option>
              <option value="weighted">the weighted average says so</option>
            </select>
            <span className="hint">Each camera: {config.inspection.aggregate} of its usable frames, then this rule</span>
          </div>
          {fusion === "weighted" &&
            INSPECTION_CAMERAS.map((cam) => (
              <div key={cam} className="field">
                <label htmlFor={`w-${cam}`}>Weight {cam}</label>
                <input id={`w-${cam}`} type="number" step={0.05} min={0} max={1} value={weights[cam] ?? 0} onChange={(e) => setWeights({ ...weights, [cam]: Number(e.target.value) })} />
              </div>
            ))}
        </div>
      </div>
    </Panel>
  );
}

export function Settings({ config, snapshot, notify, refresh }: { config: Config | null; snapshot: Snapshot | null; notify: (m: string, e?: boolean) => void; refresh: () => void }) {
  if (!config) return <div className="faint">Loading settings…</div>;
  const teach = () =>
    api("/api/pose/teach", { body: {} })
      .then(() => {
        notify("Inspection pose taught from the current gripper position");
        refresh();
      })
      .catch((error: Error) => notify(error.message, true));
  const torque = (enabled: boolean) =>
    api("/api/torque", { body: { enabled } })
      .then(() => notify(enabled ? "Torque restored: the arm holds its position" : "Torque cut"))
      .catch((error: Error) => notify(error.message, true));
  const pose = config.inspection.pose_xyz;
  const ee = snapshot?.robot?.ee;
  return (
    <div className="settings">
      <Panel
        title="Inspection pose"
        actions={
          <button className="btn sm primary" disabled={!ee} onClick={teach}>
            Teach from current pose
          </button>
        }
      >
        <div className="stack">
          <div className="muted" style={{ fontSize: 12 }}>
            Hold the brick in front of the inspection camera ({config.inspection.camera}), then teach. Frames are taken while the gripper is within{" "}
            {(config.inspection.pose_tolerance_m * 200).toFixed(0)} cm of this point.
          </div>
          <div className="kv">
            <span>
              taught <b className="mono">{pose.map((v) => v.toFixed(3)).join(", ")} m</b>
            </span>
            {ee && (
              <span>
                gripper now <b className="mono">{ee.map((v) => v.toFixed(3)).join(", ")} m</b>
              </span>
            )}
          </div>
        </div>
      </Panel>
      <Panel
        title="Home pose"
        actions={
          <button
            className="btn sm"
            disabled={!snapshot?.robot}
            onClick={() =>
              api("/api/home/teach", { body: {} })
                .then(() => {
                  notify("Home pose taught from the current arm position");
                  refresh();
                })
                .catch((error: Error) => notify(error.message, true))
            }
          >
            Teach from current pose
          </button>
        }
      >
        <div className="stack">
          <div className="muted" style={{ fontSize: 12 }}>
            The Home button (Cell tab, Live only) moves the arm here at {config.home.speed} units/s; STOP interrupts it.
          </div>
          <div className="kv">
            <span>
              home <b className="mono">{config.home.joints.map((v) => v.toFixed(1)).join(", ")}</b>
            </span>
            {snapshot?.robot && (
              <span>
                arm now <b className="mono">{snapshot.robot.joints.map((v) => v.toFixed(1)).join(", ")}</b>
              </span>
            )}
          </div>
        </div>
      </Panel>
      <Panel title="Motors">
        <div className="stack">
          <div className="muted" style={{ fontSize: 12 }}>
            STOP holds the arm where it is with torque on. Cutting torque lets the arm fall: support it first.
          </div>
          <div className="row">
            <button
              className="btn danger"
              disabled={!snapshot?.connected || snapshot.read_only || !snapshot.torque}
              onClick={() => {
                if (window.confirm("Cut motor torque? The arm will fall under gravity.")) torque(false);
              }}
            >
              Cut torque
            </button>
            <button className="btn" disabled={!snapshot?.connected || snapshot.read_only || snapshot.torque} onClick={() => torque(true)}>
              Restore torque (hold here)
            </button>
          </div>
        </div>
      </Panel>
      <CameraFusion config={config} notify={notify} onSaved={refresh} />
      {SECTIONS.map((spec) => (
        <SectionForm key={spec.section} {...spec} config={config} notify={notify} onSaved={refresh} />
      ))}
    </div>
  );
}
