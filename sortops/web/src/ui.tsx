import { type ReactNode, useEffect, useRef, useState } from "react";

/**
 * A camera image refreshed by polling a JPEG (next request only after the previous one loaded).
 * Not MJPEG: browsers allow ~6 connections per host, and 4 never-ending streams starved the WebSocket and the API.
 */
export function LiveImage({ src, alt, intervalMs = 120 }: { src: string; alt: string; intervalMs?: number }) {
  const [url, setUrl] = useState(`${src}?t=0`);
  const timer = useRef<number | undefined>(undefined);
  const next = (delay: number) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setUrl(`${src}?t=${Date.now()}`), document.hidden ? 1000 : delay);
  };
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return <img src={url} alt={alt} onLoad={() => next(intervalMs)} onError={() => next(1000)} />;
}

export function Panel({ title, actions, children, className }: { title: string; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className ?? ""}`} aria-label={title}>
      <header className="panel-head">
        <h2 className="panel-title" style={{ margin: 0 }}>
          {title}
        </h2>
        <div className="spacer" />
        {actions}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

/** Minimal SVG sparkline; ``max`` pins the scale (e.g. 100 for percentages). */
export function Sparkline({
  values,
  color = "var(--accent)",
  max,
  min = 0,
  threshold,
}: {
  values: (number | null)[];
  color?: string;
  max?: number;
  min?: number;
  threshold?: number;
}) {
  const points = values.map((value, index) => [index, value] as const).filter((p): p is readonly [number, number] => p[1] !== null);
  if (points.length < 2) return <svg className="spark" aria-hidden="true" />;
  const top = max ?? Math.max(...points.map((p) => p[1])) * 1.15;
  const span = Math.max(1e-6, top - min);
  const width = 100;
  const height = 28;
  const n = Math.max(1, values.length - 1);
  const path = points
    .map(([i, v], k) => `${k === 0 ? "M" : "L"}${((i / n) * width).toFixed(2)},${(height - ((v - min) / span) * (height - 2) - 1).toFixed(2)}`)
    .join(" ");
  const first = points[0];
  const last = points[points.length - 1];
  const area = first && last ? `${path} L${((last[0] / n) * width).toFixed(2)},${height} L${((first[0] / n) * width).toFixed(2)},${height} Z` : "";
  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      <path d={area} fill={color} opacity={0.12} />
      {threshold !== undefined && threshold <= top && (
        <line
          x1={0}
          x2={width}
          y1={height - ((threshold - min) / span) * (height - 2) - 1}
          y2={height - ((threshold - min) / span) * (height - 2) - 1}
          stroke="var(--bad)"
          strokeDasharray="3 3"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
      )}
      <path d={path} fill="none" stroke={color} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function DeviceTag({ device }: { device: string }) {
  return <span className={`device-tag ${device}`}>{device}</span>;
}

export function Dot({ state }: { state: "ok" | "warn" | "err" | "off" }) {
  return <span className={`dot ${state === "off" ? "" : state}`} aria-hidden="true" />;
}
