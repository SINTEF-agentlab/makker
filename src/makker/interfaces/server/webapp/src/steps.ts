// How the thread condenses the agent's work. Everything the agent does between two
// things it says or shows (its tool calls and its reasoning) is one group: a single
// line once it is done, with the steps one click away. Pure functions, unit-tested.

import type { ThreadItem } from "./store";
import { argPreview } from "./toolPolicy";

export type ToolStep = Extract<ThreadItem, { kind: "tool" }>;
export type Step = ToolStep | Extract<ThreadItem, { kind: "reasoning" }>;

export type Block =
  | { kind: "group"; id: string; steps: Step[] }
  | { kind: "item"; item: Exclude<ThreadItem, Step> };

const isStep = (item: ThreadItem): item is Step => item.kind === "tool" || item.kind === "reasoning";
const isPlan = (item: ThreadItem): boolean => item.kind === "tool" && item.name === "write_todos";

/** The thread as blocks: each run of consecutive steps is one group. An update
 *  to the plan is not a step: the plan shows it (see `currentPlan`). */
export function groupSteps(items: ThreadItem[]): Block[] {
  const blocks: Block[] = [];
  for (const item of items) {
    if (isPlan(item)) continue;
    const last = blocks[blocks.length - 1];
    if (!isStep(item)) blocks.push({ kind: "item", item });
    else if (last?.kind === "group") last.steps.push(item);
    else blocks.push({ kind: "group", id: item.id, steps: [item] });
  }
  return blocks;
}

// What a tool did, for one call and for several. A tool not listed here reads
// "Used <label>", so a tool an extension adds needs no entry.
const DID: Record<string, [one: string, many: string]> = {
  read_file: ["Read a file", "Read {n} files"],
  grep: ["Searched", "Searched {n} times"],
  glob: ["Looked for files", "Looked for files {n} times"],
  ls: ["Listed a folder", "Listed {n} folders"],
  run_julia: ["Ran Julia", "Ran Julia {n} times"],
  plot_julia: ["Made a plot", "Made {n} plots"],
  recapture_plot: ["Captured a plot", "Captured {n} plots"],
  close_plots: ["Closed plots", "Closed plots {n} times"],
  execute: ["Ran a command", "Ran {n} commands"],
  write_file: ["Wrote a file", "Wrote {n} files"],
  edit_file: ["Edited a file", "Made {n} edits"],
  delete: ["Deleted a file", "Deleted {n} files"],
  task: ["Delegated a task", "Delegated {n} tasks"],
  record_attempt: ["Recorded an attempt", "Recorded {n} attempts"],
  write_report: ["Wrote a report", "Wrote {n} reports"],
  reset_julia: ["Restarted Julia", "Restarted Julia {n} times"],
  remember: ["Saved a note", "Saved {n} notes"],
};

/** A group's one-line summary: "Read 4 files, searched 2 times, ran Julia". It
 *  counts the calls that ran: one the user declined or stopped did not happen. */
export function groupSummary(steps: Step[]): string {
  const counts = new Map<string, { n: number; label: string }>();
  for (const step of steps) {
    if (step.kind !== "tool" || step.status === "rejected" || step.status === "stopped") continue;
    const key = step.name ?? "tool";
    const seen = counts.get(key);
    if (seen) seen.n += 1;
    else counts.set(key, { n: 1, label: step.label || key });
  }
  const phrases = [...counts].map(([name, { n, label }]) => {
    const [one, many] = DID[name] ?? [`Used ${label}`, `Used ${label} {n} times`];
    return n === 1 ? one : many.replace("{n}", String(n));
  });
  if (!phrases.length) return steps.some((s) => s.kind === "tool") ? "Nothing ran" : "Thought";
  return phrases.map((p, i) => (i ? p[0].toLowerCase() + p.slice(1) : p)).join(", ");
}

export const failedCount = (steps: Step[]): number =>
  steps.filter((s) => s.kind === "tool" && s.status === "error").length;

/** How long a group took, first step's start to last step's end, when known. */
export function groupDuration(steps: Step[]): number | null {
  const tools = steps.filter((s): s is ToolStep => s.kind === "tool");
  const start = tools.find((s) => s.startedAt != null)?.startedAt;
  const end = Math.max(...tools.map((s) => s.endedAt ?? 0));
  return start != null && end > start ? end - start : null;
}

/** The first line of some text, without markdown marks, cut at a word. An
 *  underscore stays: it is far more often part of a name than emphasis. */
export function snippet(text: string, max = 90): string {
  const first = text.split("\n").map((l) => l.trim()).find((l) => l) || "";
  const clean = first.replace(/[*#`>]/g, "").trim();
  return clean.length > max ? clean.slice(0, max).replace(/\s+\S*$/, "") + "…" : clean;
}

/** A step still under way: a call that has not ended, a thought still growing. */
export const isRunning = (step: Step): boolean =>
  step.kind === "tool" ? step.status === "running" : step.live;

/** What the agent is doing now, for the status line: the step under way, else
 *  just that it is working (the model does not say when it is writing a call). */
export function liveActivity(steps: Step[]): { label: string; what: string; step?: ToolStep } {
  const running = [...steps].reverse().find(isRunning);
  if (running?.kind === "tool") {
    return {
      label: running.label || running.name || "Tool",
      what: argPreview(running.args),
      step: running,
    };
  }
  if (running) {
    // The thought as it grows: its newest line says more than its first.
    const lines = running.text.split("\n").filter((l) => l.trim());
    return { label: "Thinking", what: snippet(lines[lines.length - 1] ?? "") };
  }
  return { label: "Working", what: "" };
}

export interface Todo {
  content?: string;
  activeForm?: string;
  status?: string;
}

/** The agent's plan as it stands: the newest plan update since the user last
 *  spoke. Null when this turn has not planned, or has nothing left on its plan. */
export function currentPlan(items: ThreadItem[]): Todo[] | null {
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (item.kind === "user") return null;
    if (isPlan(item) && item.kind === "tool" && Array.isArray(item.args?.todos)) {
      const todos = (item.args.todos as unknown[]).filter((t): t is Todo => !!t && typeof t === "object");
      return todos.some((t) => t.status !== "completed") ? todos : null;
    }
  }
  return null;
}
