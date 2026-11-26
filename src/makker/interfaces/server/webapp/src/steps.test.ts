import { describe, expect, it } from "vitest";

import type { ThreadItem } from "./store";
import {
  currentPlan,
  failedCount,
  groupDuration,
  groupSteps,
  groupSummary,
  liveActivity,
  snippet,
  type Step,
} from "./steps";

let n = 0;
const tool = (name: string, patch: Partial<Extract<ThreadItem, { kind: "tool" }>> = {}): Step => ({
  kind: "tool",
  id: `t${++n}`,
  toolCallId: `c${n}`,
  name,
  status: "done",
  output: "",
  ...patch,
});
const thought = (text: string, live = false): Step => ({ kind: "reasoning", id: `r${++n}`, text, live });
const said = (text: string): ThreadItem => ({ kind: "assistant", id: `a${++n}`, text });
const user = (text: string): ThreadItem => ({ kind: "user", id: `u${++n}`, text });
const plan = (...statuses: string[]): ThreadItem =>
  tool("write_todos", { args: { todos: statuses.map((status, i) => ({ content: `step ${i}`, status })) } });

describe("groupSteps", () => {
  it("makes one group of the steps between two things the agent says or shows", () => {
    const items = [
      user("go"),
      thought("hm"),
      tool("read_file"),
      tool("grep"),
      said("I looked."),
      tool("run_julia"),
      { kind: "viz-chip", id: "v", viewId: "x", title: "Plot", viewKind: "plot" } as ThreadItem,
      tool("run_julia"),
    ];
    const blocks = groupSteps(items);
    expect(blocks.map((b) => (b.kind === "group" ? b.steps.length : b.item.kind))).toEqual([
      "user", 3, "assistant", 1, "viz-chip", 1,
    ]);
  });

  it("leaves plan updates out: the plan shows them", () => {
    const blocks = groupSteps([plan("in_progress"), tool("read_file"), plan("completed")]);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].kind === "group" && blocks[0].steps.map((s) => s.kind === "tool" && s.name)).toEqual(["read_file"]);
  });
});

describe("groupSummary", () => {
  it("counts what was done, in the order it first happened", () => {
    const steps = [tool("read_file"), tool("grep"), tool("read_file"), thought("x"), tool("run_julia")];
    expect(groupSummary(steps)).toBe("Read 2 files, searched, ran Julia");
  });

  it("names a tool it does not know by its label", () => {
    expect(groupSummary([tool("pick_horizon", { label: "Pick horizon" }), tool("pick_horizon", { label: "Pick horizon" })])).toBe(
      "Used Pick horizon 2 times",
    );
  });

  it("is a thought when nothing else happened", () => {
    expect(groupSummary([thought("a"), thought("b")])).toBe("Thought");
  });

  it("leaves out a call that was declined or stopped: it did not happen", () => {
    const steps = [tool("read_file"), tool("edit_file", { status: "rejected" }), tool("run_julia", { status: "stopped" })];
    expect(groupSummary(steps)).toBe("Read a file");
    expect(groupSummary(steps.slice(1))).toBe("Nothing ran");
  });
});

describe("failures and time", () => {
  it("counts failed steps and times the group from first start to last end", () => {
    const steps = [
      tool("run_julia", { startedAt: 1000, endedAt: 3000 }),
      tool("run_julia", { status: "error", startedAt: 4000, endedAt: 9000 }),
    ];
    expect(failedCount(steps)).toBe(1);
    expect(groupDuration(steps)).toBe(8000);
    expect(groupDuration([thought("x")])).toBeNull();
  });
});

describe("snippet", () => {
  it("is the first line without markdown marks, names intact, cut at a word", () => {
    expect(snippet("\n**Plan** the `run_julia` call\nmore")).toBe("Plan the run_julia call");
    expect(snippet("word ".repeat(40))).toMatch(/^(word ){17}word…$/);
  });
});

describe("liveActivity", () => {
  it("is the step under way, by its label and what it acts on", () => {
    const running = tool("run_julia", { label: "Julia", status: "running", args: { code: "x", description: "Run the base case" } });
    expect(liveActivity([tool("read_file"), running])).toMatchObject({ label: "Julia", what: "Run the base case", step: running });
  });

  it("is the newest line of a thought still growing", () => {
    expect(liveActivity([thought("first line\nsecond line", true)])).toEqual({ label: "Thinking", what: "second line" });
  });

  it("otherwise just says the agent is working", () => {
    expect(liveActivity([tool("read_file")]).label).toBe("Working");
    expect(liveActivity([]).label).toBe("Working");
  });
});

describe("currentPlan", () => {
  it("is the newest plan of this turn while something is left on it", () => {
    expect(currentPlan([user("go"), plan("completed", "in_progress"), tool("read_file")])).toHaveLength(2);
    expect(currentPlan([user("go"), plan("in_progress"), plan("completed")])).toBeNull();
    expect(currentPlan([plan("in_progress"), user("next")])).toBeNull();
  });

  it("drops entries that are not todos", () => {
    const bad = tool("write_todos", { args: { todos: [null, { content: "real", status: "pending" }] } });
    expect(currentPlan([bad])).toEqual([{ content: "real", status: "pending" }]);
  });
});
