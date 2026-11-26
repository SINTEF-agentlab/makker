import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { framePool } from "./canvas/framePool";
import { createSessionStore } from "./store";
import type { SessionStore, ThreadItem } from "./store";
import type { StoreApi } from "zustand/vanilla";

let store: StoreApi<SessionStore>;
const state = () => store.getState();
const items = () => state().items;
const byKind = <K extends ThreadItem["kind"]>(kind: K) =>
  items().filter((it): it is Extract<ThreadItem, { kind: K }> => it.kind === kind);

beforeEach(() => {
  store = createSessionStore();
});

describe("assistant text streaming", () => {
  it("coalesces deltas into one assistant item and clears the thinking indicator", () => {
    state().beginWorking();
    state().handle({ type: "text", text: "Hel" });
    state().handle({ type: "text", text: "lo" });
    const msgs = byKind("assistant");
    expect(msgs).toHaveLength(1);
    expect(msgs[0].text).toBe("Hello");
    expect(state().working).toBe(false);
  });
});

describe("compaction", () => {
  const compaction = {
    type: "compaction" as const,
    text: "Context compacted automatically: summarized 16 older messages.",
    summarized: 16,
    kept: 3,
    offloaded: true,
    manual: false,
  };

  it("adds a note mid-turn without ending the turn, and later text starts a new bubble", () => {
    state().beginWorking();
    state().handle({ type: "text", text: "before" });
    state().handle(compaction);
    expect(state().busy).toBe(true);
    state().handle({ type: "text", text: "after" });
    expect(items().map((it) => it.kind)).toEqual(["assistant", "sys-note", "assistant"]);
    expect(byKind("sys-note")[0].text).toBe(compaction.text);
  });

  it("replays as the same note", () => {
    state().replay([{ type: "user", text: "q" }, compaction]);
    expect(byKind("sys-note").map((it) => it.text)).toEqual([compaction.text]);
  });
});

describe("reasoning", () => {
  it("streams into a live block and collapses it when assistant prose starts", () => {
    state().handle({ type: "reasoning", text: "thinking " });
    state().handle({ type: "reasoning", text: "hard" });
    expect(byKind("reasoning")[0]).toMatchObject({ text: "thinking hard", live: true });
    state().handle({ type: "text", text: "Answer" });
    expect(byKind("reasoning")[0].live).toBe(false);
    expect(byKind("assistant")[0].text).toBe("Answer");
  });

  it("starts a fresh assistant bubble after interleaved reasoning, in order", () => {
    // text -> reasoning -> text must render the second text AFTER the reasoning,
    // not appended back into the first bubble.
    state().handle({ type: "text", text: "first" });
    state().handle({ type: "reasoning", text: "rethink" });
    state().handle({ type: "text", text: "second" });
    const assistants = byKind("assistant");
    expect(assistants.map((a) => a.text)).toEqual(["first", "second"]);
    // The reasoning sits between the two assistant items in the thread.
    const kinds = items().map((it) => it.kind);
    expect(kinds).toEqual(["assistant", "reasoning", "assistant"]);
  });
});

describe("tool lifecycle", () => {
  it("creates one card and ends with the final terminal output", () => {
    state().handle({ type: "tool", event: "requested", name: "run_julia", tool_call_id: "1", args: { code: "1+1" } });
    state().handle({ type: "tool", event: "delta", name: "run_julia", tool_call_id: "1", content: "running…" });
    state().handle({ type: "tool", event: "finished", name: "run_julia", tool_call_id: "1", content: "2" });
    const tools = byKind("tool");
    expect(tools).toHaveLength(1);
    expect(tools[0]).toMatchObject({ status: "done", output: "2", args: { code: "1+1" } });
  });

  it("honours per-tool policy: read_file shows a line note, not raw output", () => {
    state().handle({ type: "tool", event: "requested", name: "read_file", tool_call_id: "r", args: { file_path: "a.jl" } });
    state().handle({ type: "tool", event: "finished", name: "read_file", tool_call_id: "r", content: "l1\nl2\nl3" });
    expect(byKind("tool")[0]).toMatchObject({ status: "done", output: "", note: "3 lines" });
  });

  it("always surfaces error output", () => {
    state().handle({ type: "tool", event: "error", name: "ls", tool_call_id: "e", content: "boom" });
    expect(byKind("tool")[0]).toMatchObject({ status: "error", output: "boom" });
  });
});

describe("canvas views", () => {
  it("opens a plot view and adds an inline chip; same slot refreshes in place", () => {
    state().handle({ type: "viz", url: "/a", kind: "plot", slot: "fig", title: "First" });
    expect(state().canvasOpen).toBe(true);
    expect(state().activeView).toBe("slot:fig");
    expect(state().viewOrder).toEqual(["slot:fig"]);
    const nonce0 = state().views["slot:fig"].nonce;

    state().handle({ type: "viz", url: "/b", kind: "plot", slot: "fig", title: "Updated" });
    expect(state().viewOrder).toEqual(["slot:fig"]); // not stacked
    expect(state().views["slot:fig"].url).toBe("/b");
    expect(state().views["slot:fig"].nonce).toBe(nonce0 + 1); // forces a reload
    expect(byKind("viz-chip")).toHaveLength(2); // a fresh reference each time
  });

  it("removing the active view closes the canvas", () => {
    state().handle({ type: "viz", url: "/a", kind: "plot", slot: "fig" });
    state().removeView("slot:fig");
    expect(state().canvasOpen).toBe(false);
    expect(state().activeView).toBeNull();
  });
});

describe("artifacts", () => {
  it("an image artifact is an inline image item and claims no canvas tab", () => {
    // Its card shows the image in full, so a tab up front is one the user never
    // asked for, sitting next to the live views the canvas exists for.
    state().handle({ type: "artifact", url: "/p.png", mime: "image/png", caption: "Fig" });
    expect(byKind("artifact-image")[0]).toMatchObject({ url: "/p.png", title: "Fig" });
    expect(Object.keys(state().views)).toHaveLength(0);
    expect(state().canvasOpen).toBe(false);
  });
  it("opening an image registers its view on demand and shows it", () => {
    state().handle({ type: "artifact", url: "/p.png", mime: "image/png", caption: "Fig" });
    state().openImage("/p.png", "Fig");
    expect(state().views["url:/p.png"].kind).toBe("image");
    expect(state().activeView).toBe("url:/p.png");
    expect(state().canvasOpen).toBe(true);
    // Opening the same image again must not stack a duplicate tab.
    state().openImage("/p.png", "Fig");
    expect(state().viewOrder).toEqual(["url:/p.png"]);
  });
  it("a non-image artifact is just a file link, no view", () => {
    state().handle({ type: "artifact", url: "/f.csv", mime: "text/csv", caption: "Data" });
    expect(byKind("artifact-file")[0]).toMatchObject({ url: "/f.csv", caption: "Data" });
    expect(Object.keys(state().views)).toHaveLength(0);
  });
});

describe("interrupts", () => {
  it("records the pending approval and frees the composer", () => {
    state().beginWorking();
    state().handle({
      type: "interrupt",
      interrupt_id: "x",
      actions: [{ name: "execute" }],
      allowed_decisions: ["approve", "reject"],
      allowlist: [],
    });
    expect(state().busy).toBe(false);
    expect(state().pending?.actions[0].name).toBe("execute");
    state().clearInterrupt();
    expect(state().pending).toBeNull();
  });

  it("drops the card and shows the turn running when the server resolves it itself", () => {
    // /approval-mode auto with a card on screen: the server resumes the turn.
    state().handle({
      type: "interrupt",
      interrupt_id: "x",
      actions: [{ name: "execute" }],
      allowed_decisions: ["approve", "reject"],
      allowlist: [],
    });
    state().handle({ type: "interrupt_resolved" });
    expect(state().pending).toBeNull();
    expect(state().busy).toBe(true);
    expect(state().working).toBe(true);
  });

  it("shows every pending interrupt on one card, offering what all of them allow", () => {
    // The server applies one decision to all of them.
    const card = (id: string, name: string, allowed: string[], allowlist: string[]) => ({
      type: "interrupt" as const,
      interrupt_id: id,
      actions: [{ name }],
      allowed_decisions: allowed,
      allowlist,
    });
    state().handle(card("a", "write_file", ["approve", "reject"], ["file_edits"]));
    state().handle(card("b", "execute", ["approve", "reject", "respond"], []));
    state().handle(card("a", "write_file", ["approve", "reject"], ["file_edits"])); // re-sent
    expect(state().pending).toMatchObject({ ids: ["a", "b"], allowed: ["approve", "reject"], allowlist: [] });
    expect(state().pending?.actions.map((a) => a.name)).toEqual(["write_file", "execute"]);
  });
});

describe("context meter", () => {
  const meter = { model: "openai:m", window: 1000, compact_at: 850, offload_at: 595, used: 260, estimated: false };
  it("takes the meter from each context message", () => {
    state().handle({ type: "context", ...meter });
    expect(state().context).toEqual(meter);
    state().handle({ type: "context", ...meter, used: 400 });
    expect(state().context?.used).toBe(400);
  });
  it("takes the meter from the snapshot, and keeps none when it has none", () => {
    state().handle({ type: "snapshot", messages: [], turn: [], busy: false, model: null, context: meter });
    expect(state().context).toEqual(meter);
    state().handle({ type: "snapshot", messages: [], turn: [], busy: false, model: null });
    expect(state().context).toBeNull();
  });
});

describe("side outputs do not clear the thinking indicator", () => {
  it("keeps working through context ticks and viz, clears on text", () => {
    state().beginWorking();
    state().handle({ type: "context", window: 1000, used: 10 });
    expect(state().working).toBe(true);
    state().handle({ type: "viz", url: "/a", kind: "plot" });
    expect(state().working).toBe(true);
    state().handle({ type: "text", text: "done" });
    expect(state().working).toBe(false);
  });
});

describe("turn end / error / notice", () => {
  it("turn_end clears busy and collapses live reasoning", () => {
    state().handle({ type: "reasoning", text: "mid" });
    state().beginWorking();
    state().handle({ type: "turn_end", text: "" });
    expect(state().busy).toBe(false);
    expect(state().working).toBe(false);
    expect(byKind("reasoning")[0].live).toBe(false);
  });

  it("a stopped turn stops the spinner of the tool it interrupted", () => {
    state().handle({ type: "tool", event: "requested", name: "run_julia", tool_call_id: "c1" });
    state().handle({ type: "turn_end", text: "", cancelled: true });
    expect(byKind("tool")[0].status).toBe("stopped");
    // A turn that ends normally settles a card that never heard its end as done,
    // which is how the record replays it.
    state().handle({ type: "tool", event: "requested", name: "write_todos", tool_call_id: "c2" });
    state().handle({ type: "turn_end", text: "" });
    expect(byKind("tool").map((t) => t.status)).toEqual(["stopped", "done"]);
  });

  it("an error offers to retry the prompt that failed, once one was sent", () => {
    state().handle({ type: "error", message: "first" });
    expect(byKind("error")[0].retry).toBeNull();
    state().startTurn("hello");
    state().handle({ type: "error", message: "second" });
    state().startTurn("again");
    expect(byKind("error").map((e) => e.retry)).toEqual([null, "hello"]);
  });

  it("a declined call's card shows it was rejected instead of spinning", () => {
    const args = { file_path: "a.txt", content: "x" };
    state().handle({ type: "tool", event: "requested", name: "write_file", tool_call_id: "w", args });
    state().handle({ type: "tool", event: "requested", name: "read_file", tool_call_id: "r", args: {} });
    state().markRejected([{ name: "write_file", args }]);
    expect(byKind("tool").map((t) => t.status)).toEqual(["rejected", "running"]);
  });

  it("notice clears busy and adds a system note", () => {
    state().beginWorking();
    state().handle({ type: "notice", text: "Compacted." });
    expect(state().busy).toBe(false);
    expect(byKind("sys-note")[0].text).toBe("Compacted.");
  });
});

describe("ui history_changed is internal (no thread item)", () => {
  it("does not surface as a ui-note", () => {
    state().handle({ type: "ui", action: "history_changed", payload: { title: "x" } });
    expect(byKind("ui-note")).toHaveLength(0);
    state().handle({ type: "ui", action: "well_placed", payload: { id: 3 } });
    expect(byKind("ui-note")[0].action).toBe("well_placed");
  });
});

describe("replay", () => {
  it("reconstructs a recorded conversation and opens the last view", () => {
    state().replay([
      { type: "user", text: "hi" },
      { type: "assistant", text: "hello" },
      { type: "tool", event: "requested", name: "ls", tool_call_id: "1" },
      { type: "tool", event: "finished", name: "ls", tool_call_id: "1", content: "['a/']" },
      { type: "viz", url: "/p.png", kind: "plot", slot: "fig" },
    ]);
    expect(byKind("user")[0].text).toBe("hi");
    expect(byKind("assistant")[0].text).toBe("hello");
    expect(byKind("tool")[0].status).toBe("done");
    expect(state().canvasOpen).toBe(true);
    expect(state().liveAssistantId).toBeNull();
  });
});

describe("snapshot", () => {
  it("replaces the page and carries on with a running turn", () => {
    state().setSession("s1", "model · s1");
    state().addSysNote("from before the reconnect");
    state().handle({
      type: "snapshot",
      messages: [{ type: "user", text: "simulate it" }],
      busy: true,
      model: null,
      turn: [
        { type: "tool", event: "requested", name: "run_julia", tool_call_id: "c1" },
        {
          type: "tool", event: "delta", name: "run_julia", tool_call_id: "c1",
          content: "step 1", replace: true,
        },
        { type: "text", text: "Hal" },
      ],
    });
    state().handle({ type: "text", text: "fway" }); // the turn goes on, live

    expect(items().map((it) => it.kind)).toEqual(["user", "tool", "assistant"]);
    expect(byKind("tool")[0]).toMatchObject({ status: "running", output: "step 1" });
    expect(byKind("assistant")[0].text).toBe("Halfway");
    expect(state().busy).toBe(true);
    expect(state().sessionId).toBe("s1");
    expect(state().meta).toBe("model · s1");
  });

  it("leaves the composer free when no turn is running", () => {
    state().handle({
      type: "snapshot", messages: [{ type: "user", text: "hi" }], busy: false, turn: [], model: null,
    });
    expect(byKind("user")[0].text).toBe("hi");
    expect(state().busy).toBe(false);
  });
});

describe("reset", () => {
  it("clears the thread and canvas but keeps config", () => {
    state().setConfig({ sim: "battmo", model: "anthropic:claude" });
    state().startTurn("hi");
    state().handle({ type: "viz", url: "/a", kind: "plot" });
    state().reset();
    expect(state().items).toHaveLength(0);
    expect(state().canvasOpen).toBe(false);
    expect(Object.keys(state().views)).toHaveLength(0);
    expect(state().sim).toBe("battmo");
    expect(state().model).toBe("anthropic:claude");
  });

  it("keeps the provider key status across a reset (it is account-wide)", () => {
    state().setCredentials([
      {
        provider: "openai",
        label: "OpenAI",
        env_var: "OPENAI_API_KEY",
        is_set: true,
        masked: "sk-***z",
        source: "file",
        shadowed: false,
      },
    ]);
    state().reset();
    expect(state().credentials).toHaveLength(1);
  });
});

describe("api-keys modal", () => {
  it("opens with a required provider and closes back to empty", () => {
    const required = { provider: "openai", label: "OpenAI", env_var: "OPENAI_API_KEY" };
    state().openApiKeys(required);
    expect(state().apiKeys).toEqual({ open: true, required });
    state().closeApiKeys();
    expect(state().apiKeys).toEqual({ open: false, required: null });
  });
});

describe("view sizing mode", () => {
  it("a viz carries its live/record/size fields into the view", () => {
    state().handle({
      type: "viz",
      url: "/live/s/viz/res",
      kind: "plot",
      slot: "res",
      live: true,
      record: "artifacts/res.png",
      width: 1600,
      height: 900,
    });
    const v = state().views["slot:res"];
    expect(v.live).toBe(true);
    expect(v.record).toBe("artifacts/res.png");
    expect([v.width, v.height]).toEqual([1600, 900]);
    expect(v.mode).toBeUndefined(); // unset: the stage defaults to "scale"
  });

  it("setViewMode flips the mode and a same-slot refresh keeps the choice", () => {
    state().handle({ type: "viz", url: "/a", kind: "plot", slot: "res", width: 800, height: 600 });
    state().setViewMode("slot:res", "fill");
    expect(state().views["slot:res"].mode).toBe("fill");
    state().handle({ type: "viz", url: "/b", kind: "plot", slot: "res", width: 800, height: 600 });
    expect(state().views["slot:res"].mode).toBe("fill"); // the user's choice survives
    expect(state().views["slot:res"].url).toBe("/b");
  });
});

describe("replayed live views and the frame pool", () => {
  afterEach(() => framePool.clear());

  const liveViz = {
    type: "viz" as const,
    url: "/live/s1/viz/res",
    kind: "plot",
    slot: "res",
    live: true,
    poster: "/sessions/s1/artifacts/res.png",
    record: "artifacts/res.png",
    width: 1600,
    height: 900,
  };

  it("a replayed live view adopts its frame when it survives in the pool", () => {
    state().setSession("s1", "");
    framePool.register({
      sessionId: "s1",
      viewId: "slot:res",
      url: liveViz.url,
      title: "res",
      nonce: 2,
    });
    state().replay([liveViz]);
    const v = state().views["slot:res"];
    expect(v.live).toBe(true);
    expect(v.url).toBe(liveViz.url);
    expect(v.nonce).toBe(2); // aligned with the mounted frame's reload token
  });

  it("a replayed live view without its frame stays live, on a fresh frame", () => {
    // After a reload: the server still serves the figure, and a new view takes
    // it over, so the plot comes back interactive rather than as its poster.
    state().setSession("s1", "");
    state().replay([liveViz]);
    const v = state().views["slot:res"];
    expect(v.live).toBe(true);
    expect(v.url).toBe(liveViz.url);
    expect(v.nonce).toBe(0);
  });

  it("a figure the server no longer serves replays as its poster", () => {
    // The server sends the poster, not live, and the stale frame is let go.
    state().setSession("s1", "");
    framePool.register({
      sessionId: "s1",
      viewId: "slot:res",
      url: liveViz.url,
      title: "res",
      nonce: 0,
    });
    state().replay([{ ...liveViz, live: false, url: liveViz.poster }]);
    const v = state().views["slot:res"];
    expect(v.live).toBe(false);
    expect(v.url).toBe(liveViz.poster);
    expect(v.record).toBe("artifacts/res.png"); // regenerate stays available
    expect(framePool.has("s1", "slot:res", liveViz.url)).toBe(false);
  });

  it("downgradeView releases the frame and falls back to the poster", () => {
    state().setSession("s1", "");
    state().handle(liveViz);
    framePool.register({
      sessionId: "s1",
      viewId: "slot:res",
      url: liveViz.url,
      title: "res",
      nonce: 0,
    });
    state().downgradeView("slot:res");
    expect(framePool.has("s1", "slot:res", liveViz.url)).toBe(false);
    const v = state().views["slot:res"];
    expect(v.live).toBe(false);
    expect(v.url).toBe(liveViz.poster);
  });

  it("removing a view releases its pooled frame", () => {
    state().setSession("s1", "");
    state().handle(liveViz);
    framePool.register({
      sessionId: "s1",
      viewId: "slot:res",
      url: liveViz.url,
      title: "res",
      nonce: 0,
    });
    state().removeView("slot:res");
    expect(framePool.has("s1", "slot:res", liveViz.url)).toBe(false);
  });
});

describe("closed views and the thread chip", () => {
  const liveViz = {
    type: "viz" as const,
    url: "/live/s1/viz/res",
    kind: "plot",
    slot: "res",
    live: true,
    poster: "/sessions/s1/artifacts/res.png",
    record: "artifacts/res.png",
    width: 1600,
    height: 900,
  };

  it("a closed view keeps its record and the chip reopens it downgraded", () => {
    state().setSession("s1", "");
    state().handle(liveViz);
    state().removeView("slot:res");
    expect(state().canvasOpen).toBe(false);
    const kept = state().views["slot:res"];
    expect(kept.closed).toBe(true);
    expect(kept.live).toBe(false); // its frame is gone; a fresh one would corrupt
    expect(kept.url).toBe(liveViz.poster);
    expect(kept.record).toBe("artifacts/res.png"); // regenerate stays available

    // The chip in the thread calls openView: the tab re-pins and shows again.
    state().openView("slot:res");
    expect(state().canvasOpen).toBe(true);
    expect(state().activeView).toBe("slot:res");
    expect(state().viewOrder).toEqual(["slot:res"]);
    expect(state().views["slot:res"].closed).toBe(false);
  });

  it("a fresh viz on a closed slot re-pins its tab", () => {
    state().setSession("s1", "");
    state().handle(liveViz);
    state().removeView("slot:res");
    expect(state().viewOrder).toEqual([]);
    state().handle({ ...liveViz, url: "/live/s1/viz/res" });
    expect(state().viewOrder).toEqual(["slot:res"]);
    expect(state().views["slot:res"].closed).toBe(false);
    expect(state().views["slot:res"].live).toBe(true);
  });
});

describe("failed tool replies", () => {
  const finished = (name: string, content: string) => {
    state().handle({ type: "tool", event: "requested", name, tool_call_id: "c1", args: {} });
    state().handle({ type: "tool", event: "finished", name, tool_call_id: "c1", content });
  };

  it("a plot call replying ERROR shows as failed with the reply surfaced", () => {
    finished("plot_julia", "ERROR: the code did not produce a Makie figure.");
    const [card] = byKind("tool");
    expect(card.status).toBe("error");
    // plot_julia hides successful output (the figure is the result); a failure
    // must surface anyway or the card reads done while no plot appeared.
    expect(card.output).toContain("did not produce a Makie figure");
  });

  it("a successful plot call still reads done with its output hidden", () => {
    finished("plot_julia", "served a live interactive plot (artifacts/res.png)");
    const [card] = byKind("tool");
    expect(card.status).toBe("done");
    expect(card.output).toBe("");
  });

  it("tools without the convention are not re-labeled by ERROR-looking text", () => {
    finished("grep", "ERROR_CODES = {1: 'bad'}");
    const [card] = byKind("tool");
    expect(card.status).toBe("done");
  });
});
