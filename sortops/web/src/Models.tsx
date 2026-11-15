import { useEffect, useState } from "react";
import { api, type Candidate, type Config, DEVICES, type Device, type Execution, fmt, type LiveScore, type ModelsResponse, type Role } from "./api";
import { DeviceTag, Panel } from "./ui";

const ROLE_HINT: Record<Role, string> = {
  pick_present: "ACT · picks the brick and presents both faces to the inspection camera",
  place_good: "ACT · drops the brick in the good bin",
  place_bad: "ACT · drops the brick in the defect bin",
  anomaly_front: "Anomalib · scores the front camera's brick crops (train it on front photos)",
  anomaly_wrist: "Anomalib · scores the wrist camera's brick crops (train it on wrist photos)",
  reinspect: "Next step · views the brick from another angle when the score stays uncertain",
};

interface Props {
  config: Config | null;
  refreshConfig: () => void;
  models: ModelsResponse | null;
  refresh: () => void;
  notify: (message: string, error?: boolean) => void;
  running: boolean;
}

export function Models({ config, refreshConfig, models, refresh, notify, running }: Props) {
  const [busy, setBusy] = useState<string | null>(null);

  const run = async (key: string, action: () => Promise<unknown>, done: string) => {
    setBusy(key);
    try {
      await action();
      notify(done);
      refresh();
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), true);
    } finally {
      setBusy(null);
    }
  };

  if (!models) return <div className="faint">Loading the model catalogue…</div>;

  return (
    <div className="stack" style={{ maxWidth: 1800, margin: "0 auto" }}>
      <div className="panel">
        <div className="panel-body row between wrap" style={{ gap: 12 }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Models from Physical AI Studio</div>
            <div className="muted" style={{ fontSize: 12 }}>
              Each role takes the most recently trained model family of its datasets (pick also accepts any act-00x-… model), at its highest step count. Importing copies the OpenVINO export next to the cell and records where it came from.
            </div>
            {models.catalog_error && <div style={{ color: "var(--bad)", fontSize: 12, marginTop: 4 }}>{models.catalog_error}</div>}
          </div>
          <button
            className="btn primary"
            disabled={running || busy !== null}
            onClick={() => run("latest", () => api("/api/models/import-latest", { body: {} }), "Latest trained models imported")}
          >
            {busy === "latest" ? "Importing…" : "Import latest trained models"}
          </button>
        </div>
      </div>
      <div className="models">
        {models.roles.map((role) => {
          const execution = config?.execution?.[role];
          return (
            <RoleCard key={role} role={role} models={models} busy={busy} running={running} run={run}>
              {execution && <ExecutionForm role={role} value={execution} running={running} notify={notify} onSaved={refreshConfig} />}
            </RoleCard>
          );
        })}
      </div>
    </div>
  );
}

function Calibration({
  role,
  models,
  busy,
  running,
  run,
}: {
  role: Role;
  models: ModelsResponse;
  busy: string | null;
  running: boolean;
  run: (key: string, action: () => Promise<unknown>, done: string) => Promise<void>;
}) {
  const camera = role.replace("anomaly_", "");
  const cal = models.anomaly_calibration[role] ?? null;
  const [line, setLine] = useState<string>(cal ? String(cal.threshold) : "");
  const [live, setLive] = useState<LiveScore | null>(null);
  useEffect(() => setLine(cal ? String(cal.threshold) : ""), [cal]);

  // Live raw score of what the camera sees now: put a good then a defective brick in front of it to place the line.
  useEffect(() => {
    let stop = false;
    const tick = () =>
      api<Record<string, LiveScore>>("/api/anomaly/live")
        .then((all) => !stop && setLive(all[camera] ?? null))
        .catch(() => !stop && setLive(null))
        .finally(() => {
          if (!stop) window.setTimeout(tick, document.hidden ? 4000 : 1500);
        });
    tick();
    return () => {
      stop = true;
    };
  }, [camera]);

  const lineValue = Number(line);
  const lineValid = line.trim() !== "" && Number.isFinite(lineValue);
  return (
    <div className="model-active" style={{ marginTop: 8 }}>
      <div className="row between wrap" style={{ gap: 8 }}>
        <div>
          <div style={{ fontWeight: 500 }}>Defect line</div>
          {cal ? (
            <div className="kv">
              <span>
                {cal.manual ? "set by hand" : <>calibrated on <b>{cal.n ?? "?"}</b> good crops</>}
              </span>
              <span>
                {cal.terms ? (
                  <>
                    score <b>peak + mean</b> (peak weight {Math.round(cal.terms.peak_weight * 100)} %, peak ={" "}
                    {cal.terms.peak_from_raw ? "raw score" : `hottest ${Math.round(cal.terms.peak_fraction * 1000) / 10} %`})
                  </>
                ) : (
                  <>
                    score <b>raw</b> (recalibrate for peak + mean)
                  </>
                )}
              </span>
              <span>
                median <b>{cal.median?.toFixed(2) ?? "–"}</b>
              </span>
              <span>
                0.5 at {cal.terms ? "combined" : "raw"} <b>{cal.threshold.toFixed(2)}</b>
              </span>
              {cal.median != null && (
                <span>
                  good median <b>{cal.median.toFixed(2)}</b>
                  {cal.n != null && ` (${cal.n} crops)`}
                </span>
              )}
              <span>{cal.manual ? <b>set by hand</b> : <b>calibrated</b>}</span>
              <span>
                <b>{fmt.date(new Date(cal.created * 1000).toISOString())}</b>
              </span>
            </div>
          ) : (
            <div style={{ color: "var(--warn)", fontSize: 12 }}>Export's own scale: it can flag good bricks. Calibrate, or set the line by hand.</div>
          )}
        </div>
        <div className="row">
          {cal && (
            <button className="btn sm ghost" disabled={running || busy !== null} onClick={() => run(`reset-${camera}`, () => api(`/api/anomaly/${camera}/calibration/reset`, { body: {} }), "Back to the export's scale")}>
              Reset
            </button>
          )}
          <button
            className="btn sm primary"
            disabled={running || busy !== null}
            onClick={() => run(`calib-${camera}`, () => api(`/api/anomaly/${camera}/calibrate`, { body: { seconds: 8 } }), `${camera}: calibrated on good bricks`)}
          >
            {busy === `calib-${camera}` ? "Show good bricks… (8 s)" : "Calibrate on good bricks"}
          </button>
        </div>
      </div>
      <div className="exec-grid" style={{ marginTop: 6 }}>
        <label htmlFor={`line-${camera}`}>Defect at raw ≥</label>
        <div className="row">
          <input
            id={`line-${camera}`}
            className="num-input"
            type="number"
            step="0.5"
            value={line}
            placeholder="raw score"
            disabled={running}
            onChange={(event) => setLine(event.target.value)}
          />
          <button
            className="btn sm"
            disabled={running || busy !== null || !lineValid || (cal != null && cal.threshold === lineValue)}
            onClick={() => run(`line-${camera}`, () => api(`/api/anomaly/${camera}/defect-line`, { body: { threshold: lineValue } }), `${camera}: defect line at raw ${lineValue}`)}
          >
            Set line
          </button>
          {live?.raw != null && (
            <button className="btn sm ghost" disabled={running} onClick={() => setLine(String(live.raw))} title="Use the raw score of what the camera sees now">
              Use live
            </button>
          )}
        </div>
        <span>Now</span>
        <span className="mono num" style={{ fontSize: 12 }}>
          {live == null
            ? "—"
            : live.error
              ? live.error
              : `${live.found ? "brick" : "no brick"} · raw ${live.raw != null ? live.raw.toFixed(2) : "n/a"} → score ${live.score != null ? live.score.toFixed(2) : "—"}`}
        </span>
      </div>
      <div className="faint" style={{ fontSize: 11 }}>
        Raw = the model's own anomaly distance. At the line a crop scores 0.5; above it, the brick counts as defective (gray
        zone in Settings → Decision). Show a good and a defective brick and read "Now" to place it.
      </div>
    </div>
  );
}

function ExecutionForm({
  role,
  value,
  running,
  notify,
  onSaved,
}: {
  role: Role;
  value: Execution;
  running: boolean;
  notify: (m: string, e?: boolean) => void;
  onSaved: () => void;
}) {
  const [draft, setDraft] = useState<Execution>(value);
  useEffect(() => setDraft(value), [value]);
  const dirty = draft.mode !== value.mode || draft.hz !== value.hz || draft.request_threshold !== value.request_threshold;
  const save = () =>
    api("/api/config", { method: "PATCH", body: { execution: { [role]: draft } } })
      .then(() => {
        notify(`${fmt.role(role)}: ${draft.mode === "sync" ? "synchronous" : "asynchronous"}, ${draft.hz} Hz, next chunk at ${Math.round(draft.request_threshold * 100)} %`);
        onSaved();
      })
      .catch((error: Error) => notify(error.message, true));
  return (
    <div className="exec">
      <div className="row between">
        <span style={{ fontWeight: 500 }}>Execution</span>
        <button className="btn sm primary" disabled={!dirty || running} onClick={save}>
          Save
        </button>
      </div>
      <div className="exec-grid">
        <label htmlFor={`mode-${role}`}>Mode</label>
        <div className="segmented" role="group" id={`mode-${role}`}>
          {(["sync", "async"] as const).map((mode) => (
            <button key={mode} className="shadow" style={{ height: 26, fontSize: 12 }} aria-pressed={draft.mode === mode} disabled={running} onClick={() => setDraft({ ...draft, mode })}>
              {mode === "sync" ? "Synchronous" : "Asynchronous"}
            </button>
          ))}
        </div>
        <label htmlFor={`hz-${role}`}>Action rate</label>
        <div className="row">
          <input id={`hz-${role}`} type="range" min={5} max={100} step={5} value={draft.hz} disabled={running} onChange={(e) => setDraft({ ...draft, hz: Number(e.target.value) })} />
          <b className="mono num">{draft.hz} Hz</b>
        </div>
        <label htmlFor={`thr-${role}`}>Request the next chunk at</label>
        <div className="row">
          <input
            id={`thr-${role}`}
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={draft.request_threshold}
            disabled={running}
            onChange={(e) => setDraft({ ...draft, request_threshold: Number(e.target.value) })}
          />
          <b className="mono num">{Math.round(draft.request_threshold * 100)} %</b>
        </div>
      </div>
      <div className="faint" style={{ fontSize: 11 }}>
        Same rules as Studio's Run model: a new chunk is asked for when the queue holds less than this share of a chunk; synchronous waits for it, asynchronous keeps moving.
      </div>
    </div>
  );
}

function RoleCard({
  role,
  models,
  busy,
  running,
  run,
  children,
}: {
  role: Role;
  models: ModelsResponse;
  busy: string | null;
  running: boolean;
  run: (key: string, action: () => Promise<unknown>, done: string) => Promise<void>;
  children?: React.ReactNode;
}) {
  const active = models.active[role];
  const candidates = models.catalog[role] ?? [];
  const device = models.devices[role];
  const bench = models.benchmarks[role];
  const error = models.errors[role];
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? candidates : candidates.slice(0, 4);

  return (
    <Panel
      title={fmt.role(role)}
      actions={
        <div className="row">
          <label className="faint" style={{ fontSize: 11 }} htmlFor={`device-${role}`}>
            Device
          </label>
          <select
            id={`device-${role}`}
            className="btn sm"
            value={device}
            disabled={running}
            onChange={(event) =>
              run(`device-${role}`, () => api("/api/config", { method: "PATCH", body: { devices: { [role]: event.target.value as Device } } }), `${fmt.role(role)} now runs on ${event.target.value}`)
            }
          >
            {DEVICES.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
          <button
            className="btn sm"
            disabled={running || busy !== null}
            onClick={() => run(`bench-${role}`, () => api("/api/models/benchmark", { body: { role, devices: DEVICES, runs: 20 } }), `Benchmark of ${fmt.role(role)} done`)}
          >
            {busy === `bench-${role}` ? "Benchmarking…" : "Benchmark CPU/GPU/NPU"}
          </button>
        </div>
      }
    >
      <div className="faint" style={{ fontSize: 11.5, marginTop: -4, marginBottom: 8 }}>
        {ROLE_HINT[role]}
      </div>
      {active ? (
        <div className="model-active">
          <div className="row between">
            <span className="n">{active.name}</span>
            <span className="chip good">active</span>
          </div>
          <div className="kv">
            <span>
              dataset <b>{active.dataset}</b>
            </span>
            {active.steps != null && (
              <span>
                steps <b>{active.steps.toLocaleString()}</b>
              </span>
            )}
            {active.val_loss != null && (
              <span>
                val loss <b>{active.val_loss.toFixed(3)}</b>
              </span>
            )}
            {active.train_loss != null && (
              <span>
                train loss <b>{active.train_loss.toFixed(3)}</b>
              </span>
            )}
            <span>
              trained <b>{fmt.date(active.created_at)}</b>
            </span>
            <span>
              imported <b>{fmt.date(active.imported_at)}</b>
            </span>
            {active.uses_depth && <span className="chip warn">uses depth</span>}
            {active.simulated && <span className="chip">simulated</span>}
          </div>
        </div>
      ) : (
        <div className="model-active">
          <span className="muted">{role === "reinspect" ? "Not trained yet. Uncertain bricks go to the defect bin until then." : "Nothing imported. Pick a model below."}</span>
        </div>
      )}
      {error && <div style={{ color: "var(--bad)", fontSize: 12, marginTop: 6 }}>Load failed: {error}</div>}
      {children}
      {role.startsWith("anomaly_") && active && <Calibration role={role} models={models} busy={busy} running={running} run={run} />}

      {bench && (
        <table style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>Device</th>
              <th className="r">p50</th>
              <th className="r">p95</th>
              <th className="r">Compile</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {DEVICES.map((d) => {
              const result = bench.results[d];
              if (!result) return null;
              const best = Math.min(...DEVICES.map((x) => bench.results[x]?.p50 ?? Number.POSITIVE_INFINITY));
              return (
                <tr key={d}>
                  <td>
                    <DeviceTag device={d} />
                  </td>
                  {result.ok ? (
                    <>
                      <td className="r mono">{fmt.ms(result.p50)} ms</td>
                      <td className="r mono faint">{fmt.ms(result.p95)} ms</td>
                      <td className="r mono faint">{fmt.ms(result.compile_ms)} ms</td>
                      <td>{result.p50 === best && <span className="chip good">fastest</span>}</td>
                    </>
                  ) : (
                    <td colSpan={4} className="muted" style={{ fontSize: 11 }}>
                      {result.error}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {candidates.length > 0 && (
        <table className="cand" style={{ marginTop: 10 }}>
          <thead>
            <tr>
              <th>Studio model</th>
              <th className="r">Steps</th>
              <th className="r">Val loss</th>
              <th>Trained</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {shown.map((candidate: Candidate) => (
              <tr key={candidate.id}>
                <td title={candidate.source}>
                  {candidate.name}
                  {candidate.recommended && <span className="chip accent" style={{ marginLeft: 6 }}>latest</span>}
                  {candidate.checkpoint && <span className="faint"> · checkpoint</span>}
                </td>
                <td className="r mono">{candidate.steps?.toLocaleString() ?? "—"}</td>
                <td className="r mono">{candidate.val_loss?.toFixed(3) ?? "—"}</td>
                <td className="muted">{fmt.date(candidate.created_at)}</td>
                <td className="r">
                  {candidate.active ? (
                    <span className="faint">active</span>
                  ) : (
                    <button
                      className="btn sm"
                      disabled={running || busy !== null}
                      onClick={() => run(`import-${candidate.id}`, () => api("/api/models/import", { body: { role, model_id: candidate.id } }), `${candidate.name} imported for ${fmt.role(role)}`)}
                    >
                      {busy === `import-${candidate.id}` ? "Importing…" : candidate.imported ? "Use" : "Import"}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {candidates.length > 4 && (
        <button className="btn sm ghost" style={{ marginTop: 6 }} onClick={() => setShowAll(!showAll)}>
          {showAll ? "Show fewer" : `Show all ${candidates.length}`}
        </button>
      )}
    </Panel>
  );
}
