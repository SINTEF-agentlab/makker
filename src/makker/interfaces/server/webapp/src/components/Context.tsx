// The context ring under the message box, and the context view it opens: how full
// the context is and what fills it. The figures come from the server
// (`context_panel.py`), which the terminal's /context renders too.

import { useEffect, useState } from "react";

import { useController, useSel } from "../context";
import { formatTokens } from "../format";
import { CloseIcon } from "../icons";
import type { ContextMeter as Meter, ContextReport } from "../protocol";

/** Share of the window, 0..1, or null before the first model call. */
function fraction(meter: Meter | null): number | null {
  if (!meter || meter.used == null || !meter.window) return null;
  return Math.min(1, meter.used / meter.window);
}

/** How near compaction the context is. */
function level(meter: Meter): "ok" | "warn" | "high" {
  const limit = meter.compact_at || meter.window;
  if (meter.used == null || !limit) return "ok";
  const share = meter.used / limit;
  return share >= 0.9 ? "high" : share >= 0.75 ? "warn" : "ok";
}

function pct(share: number): string {
  return share > 0 && share < 0.01 ? "<1%" : `${Math.round(share * 100)}%`;
}

/** "~52k of 200k tokens (26%)": the meter as words, for labels and the view. */
function describe(meter: Meter): string {
  const share = fraction(meter);
  if (meter.used == null || share === null) return "No model calls yet";
  const approx = meter.estimated ? "~" : "";
  return `${approx}${formatTokens(meter.used)} of ${formatTokens(meter.window ?? 0)} tokens (${pct(share)})`;
}

const RING_R = 6;
const RING_C = 2 * Math.PI * RING_R;

export function ContextMeter() {
  const meter = useSel((s) => s.context);
  const controller = useController();
  if (!meter?.window) return null;
  const share = fraction(meter) ?? 0;
  const label = `Context: ${describe(meter)}`;
  return (
    <button
      type="button"
      className={`ctx-ring ${level(meter)}`}
      title={label}
      aria-label={label}
      onClick={() => controller.openContext()}
    >
      <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
        <circle className="ctx-ring-track" cx="8" cy="8" r={RING_R} />
        <circle
          className="ctx-ring-fill"
          cx="8"
          cy="8"
          r={RING_R}
          strokeDasharray={`${share * RING_C} ${RING_C}`}
          transform="rotate(-90 8 8)"
        />
      </svg>
    </button>
  );
}

export function ContextView() {
  const open = useSel((s) => s.contextView.open);
  const report = useSel((s) => s.contextView.report);
  const meter = useSel((s) => s.context);
  const busy = useSel((s) => s.busy);
  const controller = useController();

  // Load the report on open, and again whenever the meter moves while it is open.
  useEffect(() => {
    if (open) void controller.refreshContext();
  }, [open, meter, controller]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") controller.closeContext();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, controller]);

  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={() => controller.closeContext()}>
      <div
        className="modal ctx-view"
        role="dialog"
        aria-modal="true"
        aria-label="Context"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h2>Context</h2>
          <button className="icon-btn" aria-label="Close" onClick={() => controller.closeContext()}>
            <CloseIcon />
          </button>
        </div>
        {report ? <ReportBody report={report} /> : null}
        <div className="modal-foot">
          <button
            className="btn"
            disabled={busy}
            title={busy ? "Available when the turn has finished" : undefined}
            onClick={() => {
              controller.closeContext();
              controller.compact();
            }}
          >
            Compact now
          </button>
          <button className="btn primary" onClick={() => controller.closeContext()}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

function ReportBody({ report }: { report: ContextReport }) {
  const [toolsOpen, setToolsOpen] = useState(false);
  const meter = report.meter;
  if (!meter) return null;
  const total = meter.window ?? 0;
  const compactAt = meter.compact_at ?? 0;
  const used = meter.used;
  const free = total && compactAt && used != null ? Math.max(0, compactAt - used) : null;
  const buffer = total && compactAt ? Math.max(0, total - compactAt) : null;
  const share = (tokens: number) => (total ? pct(tokens / total) : "");

  return (
    <>
      <p className="ctx-headline">{describe(meter)}</p>
      <p className="ctx-sub">
        <code>{meter.model}</code>
        {meter.window_source === "assumed" ? " · window size assumed" : ""}
      </p>
      {meter.estimated ? (
        // The split is the last call's, from before the compaction or switch.
        <p className="ctx-sub">The breakdown updates after the next model call.</p>
      ) : null}
      {total && used != null && !meter.estimated ? (
        <Bar report={report} total={total} compactAt={compactAt} />
      ) : null}
      {report.breakdown.length && !meter.estimated ? (
        <table className="ctx-table">
          <tbody>
            {report.breakdown.map((part) => {
              const expandable = part.key === "tool_results" && report.largest.length > 0;
              return [
                <tr key={part.key}>
                  <td>
                    <span className={`ctx-swatch part-${part.key}`} aria-hidden="true" />
                    {expandable ? (
                      <button
                        type="button"
                        className="ctx-expand"
                        aria-expanded={toolsOpen}
                        onClick={() => setToolsOpen((o) => !o)}
                      >
                        {part.label}
                        <span className="ctx-chevron" aria-hidden="true" />
                      </button>
                    ) : (
                      part.label
                    )}
                  </td>
                  <td>{formatTokens(part.tokens)}</td>
                  <td>{share(part.tokens)}</td>
                </tr>,
                expandable && toolsOpen
                  ? report.largest.map((item, i) => (
                      <tr key={`largest-${i}`} className="ctx-detail">
                        <td>{item.label}</td>
                        <td>{formatTokens(item.tokens)}</td>
                        <td />
                      </tr>
                    ))
                  : null,
              ];
            })}
            {free !== null ? (
              <tr className="ctx-free">
                <td>
                  <span className="ctx-swatch part-free" aria-hidden="true" />
                  Free space
                </td>
                <td>{formatTokens(free)}</td>
                <td>{share(free)}</td>
              </tr>
            ) : null}
            {buffer !== null ? (
              <tr className="ctx-free">
                <td>
                  <span className="ctx-swatch part-buffer" aria-hidden="true" />
                  Compaction buffer
                </td>
                <td>{formatTokens(buffer)}</td>
                <td>{share(buffer)}</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      ) : null}
      <Moved report={report} />
    </>
  );
}

/** One whole split into its parts: what fills the context, the free space up to
 *  compaction, and the compaction buffer. A mark shows where compaction starts. */
function Bar({ report, total, compactAt }: { report: ContextReport; total: number; compactAt: number }) {
  const used = report.breakdown.reduce((sum, p) => sum + p.tokens, 0);
  const parts = [
    ...report.breakdown,
    { key: "free", label: "Free space", tokens: compactAt ? Math.max(0, compactAt - used) : total - used },
    { key: "buffer", label: "Compaction buffer", tokens: compactAt ? total - Math.max(compactAt, used) : 0 },
  ].filter((p) => p.tokens > 0);
  return (
    <div className="ctx-bar-wrap">
      <div
        className="ctx-bar"
        role="img"
        aria-label={parts.map((p) => `${p.label} ${formatTokens(p.tokens)}`).join(", ")}
      >
        {parts.map((p) => (
          <span
            key={p.key}
            className={`ctx-seg part-${p.key}`}
            style={{ flexGrow: p.tokens }}
            title={`${p.label}: ${formatTokens(p.tokens)}`}
          />
        ))}
      </div>
      {compactAt ? (
        <span
          className="ctx-mark"
          style={{ left: `${(compactAt / total) * 100}%` }}
          title={`Compacts at ${formatTokens(compactAt)}`}
        />
      ) : null}
    </div>
  );
}

function Moved({ report }: { report: ContextReport }) {
  const { results, compactions } = report.moved;
  const bits = [
    results ? `${results} tool result${results === 1 ? "" : "s"} to files` : "",
    compactions ? `${compactions} compaction${compactions === 1 ? "" : "s"}` : "",
  ].filter(Boolean);
  return bits.length ? <p className="ctx-sub ctx-moved">Moved out: {bits.join(", ")}.</p> : null;
}
