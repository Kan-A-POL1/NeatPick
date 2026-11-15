import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Config, type Metrics, type ModelsResponse, useCell, usePoll } from "./api";
import { verdictBridge } from "./Annotator";
import { Cockpit, StatusChips, sendVerdict } from "./Cockpit";
import { Data } from "./Data";
import { Models } from "./Models";
import { Settings } from "./Settings";

type Tab = "cell" | "data" | "models" | "settings";

export function App() {
  const { snapshot, online } = useCell();
  const metrics = usePoll<Metrics>("/api/metrics", 1000);
  const config = usePoll<Config>("/api/config", 5000);
  const models = usePoll<ModelsResponse>("/api/models", 6000);
  const [tab, setTab] = useState<Tab>("cell");
  const [toast, setToast] = useState<{ message: string; error: boolean } | null>(null);
  const toastTimer = useRef<number | undefined>(undefined);
  const [confirmLive, setConfirmLive] = useState(false);

  const notify = useCallback((message: string, error = false) => {
    setToast({ message, error });
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), Math.min(9000, 2500 + message.length * 40));
  }, []);

  const estop = useCallback(() => {
    api("/api/estop", { body: { reason: "E-STOP button" } }).catch((error: Error) => notify(error.message, true));
  }, [notify]);

  const awaiting = snapshot?.awaiting_verdict ?? false;
  const waitingBrick = snapshot?.inspection ?? null;

  // Space = e-stop, anywhere but in a form field; G / D / N answer the operator verdict prompt.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, select, textarea") || event.repeat) return;
      if (event.code === "Space") {
        event.preventDefault();
        estop();
        return;
      }
      const verdict = ({ KeyG: "good", KeyD: "bad", KeyN: "no_piece" } as const)[event.code as "KeyG" | "KeyD" | "KeyN"];
      // The open annotator (operator mode) sends the verdict WITH the piece box and defect shapes drawn.
      if (awaiting && verdict) (verdictBridge.submit ?? ((v: typeof verdict) => sendVerdict(v, waitingBrick, notify)))(verdict);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [estop, awaiting, waitingBrick?.run, waitingBrick?.cycle, notify]); // eslint-disable-line react-hooks/exhaustive-deps

  const setMode = (mode: "shadow" | "live") => {
    if (mode === "live" && !confirmLive) {
      setConfirmLive(true);
      return;
    }
    setConfirmLive(false);
    api("/api/mode", { body: { mode } }).catch((error: Error) => notify(error.message, true));
  };

  const mode = snapshot?.mode ?? "shadow";
  const latched = snapshot?.estop.latched ?? false;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            ◆
          </span>
          SortOps
          <span className="faint" style={{ fontWeight: 400 }}>
            SO-101 sorting cell
          </span>
        </div>
        <nav className="tabs" role="tablist" aria-label="Views">
          {(["cell", "data", "models", "settings"] as const).map((key) => (
            <button key={key} role="tab" className="tab" aria-selected={tab === key} onClick={() => setTab(key)}>
              {{ cell: "Cell", data: "Data", models: "Models", settings: "Settings" }[key]}
            </button>
          ))}
        </nav>
        <StatusChips snapshot={snapshot} online={online} />
        <div className="spacer" />
        <div className="segmented" role="group" aria-label="Motor mode">
          <button className="shadow" aria-pressed={mode === "shadow"} onClick={() => setMode("shadow")} title="Models run, motors never move">
            Shadow
          </button>
          <button className="live" aria-pressed={mode === "live"} disabled={latched} onClick={() => setMode("live")} title="Policy actions drive the motors">
            Live
          </button>
        </div>
        {latched ? (
          <button
            className="btn"
            onClick={() =>
              api("/api/estop/reset", { body: {} })
                .then(() => notify("E-stop reset. Back in shadow mode."))
                .catch((error: Error) => notify(error.message, true))
            }
          >
            Reset e-stop
          </button>
        ) : null}
        <button className="estop" onClick={estop} aria-label="Emergency stop (Space)">
          STOP <kbd>SPACE</kbd>
        </button>
      </header>

      <main className="page">
        {latched && (
          <div className="banner bad" role="alert" style={{ maxWidth: 1800, margin: "0 auto 12px" }}>
            <b>E-stop latched</b>
            <span className="muted">{snapshot?.estop.reason}. The arm holds its position; nothing moves until you reset.</span>
          </div>
        )}
        {confirmLive && (
          <div className="banner warn" role="alertdialog" style={{ maxWidth: 1800, margin: "0 auto 12px" }}>
            <b>Go live?</b>
            <span className="muted">Policy actions will drive the follower. Keep everyone out of reach and a hand on STOP.</span>
            <div className="spacer" />
            <button className="btn sm" onClick={() => setConfirmLive(false)}>
              Stay in shadow
            </button>
            <button className="btn sm primary" onClick={() => setMode("live")}>
              Go live
            </button>
          </div>
        )}
        {snapshot?.connect_error && (
          <div className="banner warn" style={{ maxWidth: 1800, margin: "0 auto 12px" }}>
            <b>Hardware</b>
            <span className="muted">
              {snapshot.released
                ? `${snapshot.connect_error}. SortOps does not reconnect while released: use the Studio session, then Cell → Take the arm back.`
                : `${snapshot.connect_error}. SortOps retries every 3 s: close the Studio session (even idle, torque off, it keeps the arm open) or power the arm.`}
            </span>
          </div>
        )}
        {tab === "cell" && <Cockpit snapshot={snapshot} metrics={metrics.data} config={config.data} refreshConfig={config.refresh} models={models.data} notify={notify} refreshMetrics={metrics.refresh} />}
        {tab === "data" && <Data notify={notify} />}
        {tab === "models" && <Models config={config.data} refreshConfig={config.refresh} models={models.data} refresh={models.refresh} notify={notify} running={snapshot?.running ?? false} />}
        {tab === "settings" && <Settings config={config.data} snapshot={snapshot} notify={notify} refresh={config.refresh} />}
      </main>

      {toast && (
        <div className={`toast ${toast.error ? "err" : ""}`} role="status">
          {toast.message}
        </div>
      )}
    </div>
  );
}
