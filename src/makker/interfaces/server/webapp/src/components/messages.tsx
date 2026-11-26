// Renderers for every kind of thread item. Presentational components driven by the
// store's render model; the streaming ones (assistant, steps) are memoized so only
// what changed re-renders during a turn.

import { Fragment, memo, useEffect, useLayoutEffect, useRef, useState } from "react";

import { terminalSegments } from "../ansi";
import { useController, useSel } from "../context";
import { fmtNum, formatElapsed } from "../format";
import { KindIcon, ChevronRight } from "../icons";
import { tokenizeJulia } from "../julia";
import { Markdown } from "../markdown";
import { SLASH_COMMANDS } from "../controller";
import {
  failedCount,
  groupDuration,
  groupSummary,
  liveActivity,
  snippet,
  type Step,
  type Todo,
  type ToolStep,
} from "../steps";
import { argPreview, asObject, summarizeArgs } from "../toolPolicy";
import { KIND_LABEL, type ViewKind } from "../store";

// --- shared building blocks -------------------------------------------------

function CopyablePre({ className, children }: { className: string; children: React.ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  return (
    <pre ref={ref} className={className} style={{ position: "relative" }}>
      {children}
      <button
        ref={btn}
        type="button"
        className="copy-btn"
        onClick={(e) => {
          e.stopPropagation();
          const code = ref.current?.querySelector("code") ?? ref.current;
          navigator.clipboard.writeText(code?.textContent ?? "").then(() => {
            if (btn.current) {
              btn.current.textContent = "Copied";
              setTimeout(() => btn.current && (btn.current.textContent = "Copy"), 1200);
            }
          });
        }}
      >
        Copy
      </button>
    </pre>
  );
}

function CodeBlock({ text, julia = false }: { text: string; julia?: boolean }) {
  return (
    <CopyablePre className="tool-code">
      <code>
        {julia
          ? tokenizeJulia(text).map((t, i) => <span key={i} className={t.cls}>{t.text}</span>)
          : text}
      </code>
    </CopyablePre>
  );
}

function TerminalOutput({ raw }: { raw: string }) {
  const ref = useRef<HTMLPreElement>(null);
  // Tail to the newest line like a terminal. `stick` stays true until the user
  // scrolls up to read, so a chunk larger than the box still follows the bottom
  // (inferring it from the current position alone breaks: the box opens scrolled to
  // the top, so the first overflow already reads as "scrolled up" and never tails).
  const stick = useRef(true);
  const onScroll = () => {
    const box = ref.current;
    if (box) stick.current = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
  };
  useLayoutEffect(() => {
    const box = ref.current;
    if (box && stick.current) box.scrollTop = box.scrollHeight;
  }, [raw]);
  return (
    <pre ref={ref} className="tool-output" onScroll={onScroll}>
      {terminalSegments(raw).map((seg, i) => (
        <span key={i} style={{ color: seg.color, fontWeight: seg.bold ? 600 : undefined }}>
          {seg.text}
        </span>
      ))}
    </pre>
  );
}

// --- simple items -----------------------------------------------------------

export function UserBubble({ text }: { text: string }) {
  return (
    <div className="msg user">
      <div className="bubble">{text}</div>
    </div>
  );
}

export const AssistantMessage = memo(function AssistantMessage({ text }: { text: string }) {
  return (
    <div className="msg assistant">
      <Markdown text={text} />
    </div>
  );
});

// --- what a step shows when opened -------------------------------------------

const TODO_MARK: Record<string, string> = { completed: "✓", in_progress: "▸", pending: "○" };

function Todos({ todos }: { todos: Todo[] }) {
  return (
    <ul className="todos">
      {todos.map((t, i) => {
        const status = t.status || "pending";
        return (
          <li key={i} className={`todo ${status}`}>
            <span className="todo-mark">{TODO_MARK[status] || "○"}</span>
            <span className="todo-text">{t.content || t.activeForm || ""}</span>
          </li>
        );
      })}
    </ul>
  );
}

function Diff({ oldStr, newStr }: { oldStr: string; newStr: string }) {
  return (
    <pre className="tool-diff">
      {oldStr && oldStr.split("\n").map((l, i) => <div key={`d${i}`} className="del">{"- " + l}</div>)}
      {newStr && newStr.split("\n").map((l, i) => <div key={`a${i}`} className="add">{"+ " + l}</div>)}
    </pre>
  );
}

function KV({ obj, label }: { obj: Record<string, unknown>; label?: string }) {
  return (
    <div className="tool-kv">
      {label && <div className="kv-label">{label}</div>}
      <div className="kv-grid">
        {Object.entries(obj).map(([k, v]) => (
          <Fragment key={k}>
            <span className="kv-key">{k}</span>
            <span className="kv-val">{Array.isArray(v) ? v.map(fmtNum).join(", ") : fmtNum(v)}</span>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function ToolBody({ name, args }: { name: string | null; args: Record<string, unknown> | null }) {
  if (!args) return null;
  if (name === "edit_file" && (args.old_string != null || args.new_string != null)) {
    return (
      <>
        {args.file_path ? <div className="tool-path">{String(args.file_path)}</div> : null}
        <Diff oldStr={String(args.old_string || "")} newStr={String(args.new_string || "")} />
      </>
    );
  }
  if (name === "write_file" && args.content != null) {
    return (
      <>
        {args.file_path ? <div className="tool-path">{String(args.file_path)}</div> : null}
        <CodeBlock text={String(args.content)} julia={/\.jl$/.test(String(args.file_path || ""))} />
      </>
    );
  }
  if (name === "record_attempt") {
    const metrics = asObject(args.metrics);
    const params = asObject(args.parameters_changed);
    return (
      <>
        {args.rationale ? <div className="attempt-rationale">{String(args.rationale)}</div> : null}
        {params ? <KV obj={params} label="changed" /> : null}
        {metrics ? <KV obj={metrics} label="metrics" /> : null}
        {args.notes ? <div className="attempt-notes">{String(args.notes)}</div> : null}
      </>
    );
  }
  if (args.code != null) return <CodeBlock text={String(args.code)} julia />;
  if (args.command != null) return <CodeBlock text={String(args.command)} />;
  if (Object.keys(args).length) return <div className="tool-args">{summarizeArgs(args)}</div>;
  return null;
}

// Tools whose arguments are the content: their reply only confirms it.
const CONFIRMS = new Set(["write_file", "edit_file", "record_attempt"]);

/** Milliseconds since `since`, re-rendered once a second while `active`.
 *
 * A step that takes minutes (the first plot of a session compiles most of the
 * plotting stack) otherwise shows a spinner and nothing else, which reads as a
 * hang. The clock is the difference between "stuck" and "still going".
 */
function useElapsed(since: number | null | undefined, active: boolean): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active || since == null) return;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [active, since]);
  if (since == null || !active) return null;
  return now - since;
}

// --- steps: the agent's tool calls and reasoning, one line each ---------------

// Under this a duration is not worth showing: a number that flashes up is noise.
const WORTH_TIMING = 2000;

const STATUS_WORD: Partial<Record<ToolStep["status"], string>> = {
  error: "Failed",
  stopped: "Stopped",
  rejected: "Declined",
};

function Line({
  open,
  onToggle,
  className,
  children,
}: {
  open: boolean;
  onToggle: () => void;
  className: string;
  children: React.ReactNode;
}) {
  return (
    <button type="button" className={`line ${className}`} aria-expanded={open} onClick={onToggle}>
      {children}
      <span className="line-chevron" aria-hidden="true">
        <ChevronRight />
      </span>
    </button>
  );
}

function StepDetail({ step }: { step: ToolStep }) {
  return (
    <div className="step-detail">
      <ToolBody name={step.name} args={step.args ?? null} />
      {step.output && (step.status === "error" || !CONFIRMS.has(step.name ?? "")) ? (
        <TerminalOutput raw={step.output} />
      ) : null}
    </div>
  );
}

const StepRow = memo(function StepRow({ step }: { step: Step }) {
  const [open, setOpen] = useState(false);
  const toggle = () => setOpen((o) => !o);
  if (step.kind === "reasoning") {
    return (
      <div className="step">
        <Line className="step-line" open={open} onToggle={toggle}>
          <span className="step-label">Thought</span>
          <span className="step-what prose">{snippet(step.text)}</span>
        </Line>
        {open && <div className="step-detail reasoning-text">{step.text}</div>}
      </div>
    );
  }
  const took = step.startedAt != null && step.endedAt != null ? step.endedAt - step.startedAt : 0;
  const word = STATUS_WORD[step.status];
  const path = step.args?.file_path ?? step.args?.path;
  return (
    <div className="step">
      <Line className="step-line" open={open} onToggle={toggle}>
        <span className="step-label">{step.label || step.name}</span>
        <span className="step-what" title={path ? String(path) : undefined}>
          {argPreview(step.args)}
        </span>
        {word && <span className={`step-status ${step.status}`}>{word}</span>}
        {step.note && <span className="step-note">{step.note}</span>}
        {took >= WORTH_TIMING && <span className="step-note elapsed">{formatElapsed(took)}</span>}
      </Line>
      {open && <StepDetail step={step} />}
    </div>
  );
});

const sameSteps = (a: Step[], b: Step[]) => a.length === b.length && a.every((s, i) => s === b[i]);

/** The agent's work between two things it says or shows: one line, with the
 *  steps behind it. A lone step is its own line. */
export const StepGroup = memo(
  function StepGroup({ steps }: { steps: Step[] }) {
    const [open, setOpen] = useState(false);
    const rows = steps.map((step) => <StepRow key={step.id} step={step} />);
    if (steps.length === 1) return <div className="step-group lone">{rows}</div>;
    const failed = failedCount(steps);
    const took = groupDuration(steps);
    return (
      <div className="step-group">
        <Line className="group-line" open={open} onToggle={() => setOpen((o) => !o)}>
          <span className="group-what">{groupSummary(steps)}</span>
          {failed > 0 && <span className="step-status error">{failed} failed</span>}
          {took != null && took >= WORTH_TIMING && <span className="step-note elapsed">{formatElapsed(took)}</span>}
        </Line>
        {open && <div className="steps">{rows}</div>}
      </div>
    );
  },
  (a, b) => sameSteps(a.steps, b.steps),
);

/** The running step on the status line. Opened, it shows what a finished step's
 *  row does; closed, the newest lines of what it prints once it has run a while,
 *  so a simulation's progress shows and a file read never gets far enough to. */
function RunningStep({ step, clock }: { step: ToolStep; clock: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const ran = useElapsed(step.startedAt, true);
  return (
    <>
      <Line className="status-line" open={open} onToggle={() => setOpen((o) => !o)}>
        <span className="spinner" />
        <span className="status-label">{step.label || step.name}</span>
        <span className="step-what">{argPreview(step.args)}</span>
        {clock}
      </Line>
      {open ? (
        <StepDetail step={step} />
      ) : step.output && ran != null && ran >= WORTH_TIMING ? (
        <div className="live-tail">
          <TerminalOutput raw={step.output} />
        </div>
      ) : null}
    </>
  );
}

/** The status line: what the agent is doing now and how long the turn has run.
 *  It is the last thing in the thread for the whole turn, so a quiet stretch
 *  never reads as a hang. `steps` are the turn's newest group, if it ends in one. */
export const Status = memo(
  function Status({ steps }: { steps: Step[] }) {
    const since = useSel((s) => s.turnStartedAt);
    const elapsed = useElapsed(since, true);
    const { label, what, step } = liveActivity(steps);
    const clock =
      elapsed != null ? <span className="step-note elapsed">{formatElapsed(elapsed)}</span> : null;
    return (
      <div className="status">
        {step ? (
          <RunningStep key={step.id} step={step} clock={clock} />
        ) : (
          <div className="line status-line">
            <span className="spinner" />
            <span className="status-label">{label}</span>
            {what && <span className="step-what prose">{what}</span>}
            {clock}
          </div>
        )}
      </div>
    );
  },
  (a, b) => sameSteps(a.steps, b.steps),
);

/** The agent's plan, shown once while it works and updated in place. */
export function Plan({ todos }: { todos: Todo[] }) {
  return (
    <div className="plan">
      <Todos todos={todos} />
    </div>
  );
}

// --- canvas references ------------------------------------------------------

export function VizChip({ viewId, title, viewKind }: { viewId: string; title: string; viewKind: ViewKind }) {
  const openView = useSel((s) => s.openView);
  const active = useSel((s) => s.canvasOpen && s.activeView === viewId);
  return (
    <button className={`viz-chip${active ? " active" : ""}`} onClick={() => openView(viewId)}>
      <span className={`ico ${viewKind}`}>
        <KindIcon kind={viewKind} />
      </span>
      <div className="info">
        <div className="t">{title}</div>
        <div className="s">{KIND_LABEL[viewKind] ?? "View"}</div>
      </div>
      <span className="go">
        Open <ChevronRight />
      </span>
    </button>
  );
}

export function ArtifactImageCard({ url, title }: { url: string; title: string }) {
  const openImage = useSel((s) => s.openImage);
  return (
    <div className="art-card">
      <div className="head">
        <span className="grow">{title}</span>
        <button className="ghost" onClick={() => openImage(url, title)}>Open</button>
      </div>
      <img src={url} alt={title} onClick={() => openImage(url, title)} />
    </div>
  );
}

export function ArtifactFileCard({ url, caption }: { url: string; caption: string }) {
  return (
    <div className="art-card">
      <div className="head">{caption}</div>
      <a className="file" href={url} target="_blank" rel="noopener noreferrer">
        {url}
      </a>
    </div>
  );
}

export function SysNote({ text, level }: { text: string; level?: "warn" }) {
  return <div className={`sys-note${level ? " " + level : ""}`}>{text}</div>;
}

export function UiNote({ action, payload }: { action: string; payload: Record<string, unknown> }) {
  const json = payload && Object.keys(payload).length ? JSON.stringify(payload) : "";
  return (
    <div className="ui-note">
      <span className="ui-gear">⚙</span>
      <span className="ui-action">{action || "ui"}</span>
      {json && <span className="ui-payload">{json}</span>}
    </div>
  );
}

export function ErrorCard({ message, retry }: { message: string; retry: string | null }) {
  const controller = useController();
  return (
    <div className="error-card">
      <div className="err-msg">{message}</div>
      {retry ? (
        <button className="btn" onClick={() => controller.retry(retry)}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function HelpCard() {
  return (
    <div className="help-card">
      <div className="help-title">Commands</div>
      {SLASH_COMMANDS.map((c) => (
        <div key={c.name} className="help-row">
          <span className="help-name">{c.name + (c.hint ? " " + c.hint : "")}</span>
          <span className="help-desc">{c.desc}</span>
        </div>
      ))}
    </div>
  );
}
