import { useCallback, useEffect, useRef, useState } from "react";

export type Mode = "shadow" | "live";
export type Device = "CPU" | "GPU" | "NPU";
export const DEVICES: Device[] = ["CPU", "GPU", "NPU"];
export const POLICY_ROLES = ["pick_present", "place_good", "place_bad"] as const;
export type Role = "pick_present" | "place_good" | "place_bad" | "anomaly_front" | "anomaly_wrist" | "reinspect";
export const INSPECTION_CAMERAS = ["front", "wrist"] as const;
export type InspectionCamera = (typeof INSPECTION_CAMERAS)[number];

export interface Calibration {
  threshold: number;
  spread: number;
  median?: number;
  p95?: number;
  n?: number;
  created: number;
  manual?: boolean;
  /** Present when calibrated on raw anomaly maps: the peak + mean combination (sortops/pipeline/anomaly_score.py). */
  terms?: {
    peak_weight: number;
    peak_fraction: number;
    peak_from_raw: boolean;
    peak_median: number;
    peak_spread: number;
    mean_median: number;
    mean_spread: number;
  };
}

export interface LiveScore {
  found: boolean;
  raw?: number | null;
  exported?: number;
  score?: number;
  error?: string;
}

export interface RobotSnapshot {
  joints: number[];
  names: string[];
  ee: number[];
  temp: number[];
  load: number[];
  current: number[];
  voltage: number;
  speed: number;
}

export interface SystemSample {
  t: number;
  cpu_pct: number;
  ram_pct: number;
  package_w: number | null;
  package_temp_c: number | null;
  npu_pct: number | null;
  gpu_pct: number | null;
}

export interface Decision {
  cycle: number;
  verdict: "good" | "bad" | "uncertain" | "no_piece";
  score: number | null;
  reason: string;
  camera_scores: Record<string, number>;
  /** Each method's verdict on this brick (anomalib, algorithm, manual); null = no opinion. */
  opinions?: Record<string, { verdict: string; score: number | null } | null>;
}

export interface Snapshot {
  t: number;
  simulated: boolean;
  read_only: boolean;
  connected: boolean;
  connect_error: string | null;
  released: boolean;
  mode: Mode;
  estop: { latched: boolean; reason: string };
  torque: boolean;
  phase: string;
  phase_s: number;
  running: boolean;
  busy: string | null;
  cycle: {
    id: number;
    stages: Record<string, number>;
    scores: number[];
    camera_scores: Record<string, number[]>;
    frames_seen: number;
    frames_usable: number;
    decisions: DecisionNote[];
    pick_attempts: number;
  } | null;
  /** Live gates of step 1 (what the cycle waits for before inspecting); null outside step 1. */
  transition: Transition | null;
  robot: RobotSnapshot | null;
  ghost: number[] | null;
  execution: (Execution & { role: string; chunk: number; queue: number; requests: number }) | null;
  temp_median: number[] | null;
  temp_limit: number;
  clamped_ticks: number;
  heartbeat_age: number;
  system: SystemSample | null;
  last_decision: Decision | null;
  verdict_source: "model" | "cv" | "operator";
  awaiting_verdict: boolean;
  inspection_cameras: string[];
  /** Run and cycle of the brick being inspected (frames are stored under it). */
  inspection: { run: string; cycle: number } | null;
  piece: { index: number; defective: boolean } | null;
}

export interface LatencyStat {
  device: string;
  count: number;
  last: number;
  p50: number;
  p95: number;
  mean: number;
}

/** One transition the cycle took, and why. */
export interface DecisionNote {
  t: number;
  stage: string;
  next: string;
  text: string;
  ok: boolean;
}

export interface TransitionGate {
  key: string;
  label: string;
  ok: boolean;
  value: string;
  target: string;
  detail: string;
}

export interface Transition {
  stage: string;
  next: string;
  attempt: number;
  attempts: number;
  elapsed_s: number;
  gates: TransitionGate[];
  zone: { base_m: number; radius_m: number; growth: number; admissible: number; next_growth_at: number | null; distance_m: number | null };
  timers: { label: string; elapsed: number | null; limit: number; active: boolean }[];
}

export interface CycleRecord {
  id: number;
  run: string;
  started: number;
  ended: number | null;
  duration: number | null;
  outcome: string;
  decision_reason: string;
  score: number | null;
  scores: number[];
  frames_seen: number;
  frames_usable: number;
  recaptures: number;
  stages: Record<string, number>;
  energy_j: number | null;
  inferences: number;
  truth: "good" | "bad" | null;
  error: string | null;
  decisions?: DecisionNote[];
  pick_attempts?: number;
}

export interface Stats {
  run: string;
  cycles: number;
  sorted: number;
  good: number;
  bad: number;
  failed: number;
  recaptured: number;
  cycle_s_mean: number | null;
  cycle_s_p95: number | null;
  throughput_per_min: number | null;
  energy_j_mean: number | null;
  labelled: number;
  confusion: { tp: number; tn: number; fp: number; fn: number };
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
}

export interface Shot {
  id: number;
  cycle: number;
  phase: string;
  camera: string;
  t: number;
  score: number | null;
  has_heat: boolean;
  usability: {
    score: number;
    usable: boolean;
    sharpness: number;
    exposure: number;
    brick_fraction: number;
    pose_error_m: number;
    speed: number;
    reason: string;
  };
}

export interface LogEvent {
  t: number;
  level: "info" | "warn" | "error";
  msg: string;
}

export interface Metrics {
  latency: Record<string, LatencyStat>;
  latency_recent: Record<string, number[]>;
  system: SystemSample[];
  motor_temps: { t: number; raw: number[]; median: number[] }[];
  cycles: CycleRecord[];
  stats: Stats;
  events: LogEvent[];
  shots: Shot[];
}

export interface Candidate {
  id: string;
  name: string;
  role: Role;
  dataset: string;
  created_at: string;
  steps: number | null;
  train_loss: number | null;
  val_loss: number | null;
  source: string;
  checkpoint: boolean;
  uses_depth: boolean;
  imported: boolean;
  active: boolean;
  recommended: boolean;
}

export interface Provenance extends Omit<Candidate, "imported" | "active" | "recommended"> {
  imported_at: string;
  simulated: boolean;
}

export interface BenchmarkResult {
  ok: boolean;
  compile_ms?: number;
  p50?: number;
  p95?: number;
  runs?: number;
  error?: string;
}

export interface ModelsResponse {
  catalog: Partial<Record<Role, Candidate[]>>;
  active: Partial<Record<Role, Provenance | null>>;
  devices: Record<Role, Device>;
  errors: Record<string, string>;
  catalog_error: string | null;
  anomaly_calibration: Partial<Record<Role, Calibration | null>>;
  benchmarks: Partial<Record<Role, { t: number; results: Partial<Record<Device, BenchmarkResult>> }>>;
  roles: Role[];
}

export interface Execution {
  mode: "sync" | "async";
  hz: number;
  request_threshold: number;
}

export interface Config {
  hardware: { backend: string; cameras: string[] };
  execution: Partial<Record<Role, Execution>>;
  devices: Record<Role, Device>;
  inspection: {
    camera: string;
    cameras: string[];
    rois: Record<string, number[]>;
    model_input: Record<string, "crop" | "frame">;
    roi: number[];
    crop_mode: "auto" | "fixed";
    crop_size: number;
    crop_margin: number;
    crop_exclude_hues: number[][];
    crop_min_saturation: number;
    crop_min_value: number;
    crop_min_area: number;
    score_peak_weight: number;
    score_peak_fraction: number;
    pose_xyz: number[];
    pose_tolerance_m: number;
    period_s: number;
    window_s: number;
    min_usable_frames: number;
    best_frame_only: boolean;
    usability_threshold: number;
    min_sharpness: number;
    min_brick_fraction: number;
    max_speed_for_capture: number;
    aggregate: string;
  };
  decision: {
    verdict_source: "model" | "cv" | "operator";
    fusion: "max" | "mean" | "min" | "weighted";
    weights: Record<string, number>;
    good_below: number;
    bad_above: number;
    recapture_frames: number;
    recapture_period_s: number;
    reinspect_enabled: boolean;
    no_brick_defect_ratio: number;
  };
  release: {
    enabled: boolean;
    place_good: boolean;
    place_bad: boolean;
    trigger: number;
    open: number;
    open_tolerance: number;
    hold_s: number;
    max_open_s: number;
    min_s: number;
  };
  safety: {
    max_step: number;
    watchdog_s: number;
    watchdog_enabled: boolean;
    max_motor_temp_c: number;
    temp_window_s: number;
    soft_min: number[];
    soft_max: number[];
  };
  run: { loop: boolean; max_cycles: number; stop_after_no_brick: number; pick_attempts: number };
  segments: { pick_present: { min_s: number; still_s: number; timeout_s: number }; still_speed: number };
  pick: {
    open_below: number;
    min_closed_pos: number;
    min_load: number;
    hold_s: number;
    lost_s: number;
    require_grasp: boolean;
    zone_step_frames: number;
    zone_growth: number;
    zone_max_growth: number;
    min_admissible_in_zone: number;
    present_max_s: number;
    retry_wait_s: number;
  };
  home: { joints: number[]; speed: number };
  present: {
    enabled: boolean;
    dataset: string;
    episode: number;
    approach_speed: number;
    playback_speed: number;
    pause_epsilon: number;
    pick_open_above: number;
    pick_closed_below: number;
    pick_closed_s: number;
  };
}

export class ApiError extends Error {}

export async function api<T>(path: string, init?: { method?: string; body?: unknown }): Promise<T> {
  const response = await fetch(path, {
    method: init?.method ?? (init?.body === undefined ? "GET" : "POST"),
    headers: init?.body === undefined ? undefined : { "Content-Type": "application/json" },
    body: init?.body === undefined ? undefined : JSON.stringify(init.body),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload: unknown = await response.json();
      if (payload && typeof payload === "object" && "detail" in payload) {
        detail = String((payload as { detail: unknown }).detail);
      }
    } catch {
      // Non-JSON error body: keep the status line.
    }
    throw new ApiError(detail);
  }
  return (await response.json()) as T;
}

/** Live cell state over the WebSocket; sends the heartbeat the live-mode watchdog waits for. */
export function useCell(): { snapshot: Snapshot | null; online: boolean } {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [online, setOnline] = useState(false);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    let closed = false;
    // Heartbeats tick in a worker: browsers throttle timers of background tabs to >= 1 s, workers they leave alone.
    const ticker = new Worker(URL.createObjectURL(new Blob(["setInterval(() => postMessage(0), 250);"], { type: "text/javascript" })));
    ticker.onmessage = () => {
      if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "heartbeat" }));
    };

    const connect = () => {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${window.location.host}/api/ws`);
      socket.onopen = () => setOnline(true);
      socket.onmessage = (event: MessageEvent<string>) => {
        const message = JSON.parse(event.data) as { type: string } & Snapshot;
        if (message.type === "state") setSnapshot(message);
      };
      socket.onclose = () => {
        setOnline(false);
        if (!closed) retry = window.setTimeout(connect, 1000);
      };
    };
    connect();
    return () => {
      closed = true;
      ticker.terminate();
      window.clearTimeout(retry);
      socket?.close();
    };
  }, []);

  return { snapshot, online };
}

/** Poll ``path`` every ``ms`` (paused while the tab is hidden); ``refresh`` re-fetches now. */
export function usePoll<T>(path: string, ms: number): { data: T | null; error: string | null; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);

  const refresh = useCallback(() => {
    if (inFlight.current) return;
    inFlight.current = true;
    api<T>(path)
      .then((value) => {
        setData(value);
        setError(null);
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => {
        inFlight.current = false;
      });
  }, [path]);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(() => {
      if (!document.hidden) refresh();
    }, ms);
    return () => window.clearInterval(timer);
  }, [refresh, ms]);

  return { data, error, refresh };
}

export const fmt = {
  ms: (value: number | null | undefined) => (value == null ? "—" : value < 10 ? value.toFixed(1) : value.toFixed(0)),
  num: (value: number | null | undefined, digits = 1) => (value == null ? "—" : value.toFixed(digits)),
  pct: (value: number | null | undefined) => (value == null ? "—" : `${Math.round(value * 100)}%`),
  date: (iso: string) => {
    const date = new Date(iso);
    return Number.isNaN(date.getTime())
      ? iso
      : date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  },
  role: (role: string) =>
    ({
      pick_present: "Pick & present",
      pick: "Pick",
      present: "Present (replay)",
      place_good: "Place → good bin",
      place_bad: "Place → defect bin",
      anomaly_front: "Anomaly · front camera",
      anomaly_wrist: "Anomaly · wrist camera",
      reinspect: "Re-inspect (next)",
      inspect: "Inspect",
      await_operator: "Operator verdict",
      recapture: "Recapture",
      decide: "Decide",
      idle: "Idle",
      home: "Going home",
      end: "End of cycle",
      release: "Release",
    })[role] ?? role,
};

// Inspection data (sortops/pipeline/annotation.py) ------------------------------------------------------------

export type Box = [number, number, number, number];
export type DefectShape = { type: "rect"; box: Box } | { type: "stroke"; points: [number, number][]; width: number };
export type InspectionVerdict = "good" | "bad" | "no_piece";

export interface CameraAnnotation {
  shot: number | null;
  /** Piece box, normalized to the camera frame. */
  piece: Box | null;
  defects: DefectShape[];
}

export interface Annotation {
  verdict: InspectionVerdict;
  source: "operator" | "model";
  note: string;
  updated?: number;
  cameras: Record<string, CameraAnnotation>;
}

export interface HeatInfo {
  scale: "terms" | "raw" | "export";
  calibrated: boolean;
  max_units: number;
  peak: [number, number];
  regions: Box[];
}

export interface StoredShot {
  shot: number;
  camera: string;
  phase: string;
  t: number;
  frame: string;
  crop: string;
  heat?: string;
  heat_info?: HeatInfo;
  frame_size: [number, number];
  crop_box: Box;
  found: boolean;
  usability: { usable?: boolean; sharpness?: number; reason?: string; score?: number };
  score: number | null;
}

export interface InspectionCycle {
  run: string;
  cycle: number;
  shots: StoredShot[];
  best: Record<string, StoredShot>;
  annotation: Annotation | null;
}

export interface InspectionRun {
  run: string;
  cycles: number;
  good: number;
  bad: number;
  no_piece: number;
  unlabeled: number;
  frames: number;
  bytes: number;
  reviewed: number;
  /** Reviewed frames kept (not excluded), per camera: what Train can use. */
  kept_by_camera: Record<string, number>;
}

export interface InspectionCycleSummary {
  cycle: number;
  verdict: InspectionVerdict | null;
  source: "operator" | "model" | null;
  defects: number;
  frames: number;
  best: Record<string, { shot: number; crop: string; heat?: string | null; score: number | null; max_units: number | null }>;
}

export const inspectionFile = (run: string, cycle: number, name: string) =>
  `/api/inspection/runs/${encodeURIComponent(run)}/cycles/${cycle}/files/${encodeURIComponent(name)}`;

/** One stored frame in Data -> Review: its review, or a proposal to confirm. */
export interface ReviewItem {
  cycle: number;
  shot: number;
  camera: string;
  verdict: InspectionVerdict | null;
  frame: string;
  crop_box: Box;
  frame_size: [number, number] | null;
  usable: boolean;
  found: boolean;
  reviewed: boolean;
  piece: Box | null;
  defects: DefectShape[];
  excluded: boolean;
}

export interface ReviewQueue {
  run: string;
  items: ReviewItem[];
  total: number;
  reviewed: number;
  kept: number;
}

export interface TrainJob {
  id: string;
  runs: string[];
  camera: string;
  model: string;
  name: string;
  out: string;
  status: "preparing" | "training" | "done" | "failed" | "canceled";
  progress: number;
  message: string;
  started: number;
  ended: number | null;
  images: Record<string, number>;
  metrics: Record<string, number>;
  log_tail: string[];
}
