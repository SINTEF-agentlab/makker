import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { api, type HistoryEntry } from "./api";
import { registerPanel } from "./canvas/registry";
import { Canvas } from "./components/Canvas";
import { Composer } from "./components/Composer";
import { MoreModels } from "./components/Models";
import { Sidebar } from "./components/Sidebar";
import { Thread } from "./components/Thread";
import type { ServerMessage } from "./protocol";
import { renderWithStore } from "./test/util";

function drive(store: { getState: () => { handle: (m: ServerMessage) => void } }, ...msgs: ServerMessage[]) {
  act(() => {
    for (const m of msgs) store.getState().handle(m);
  });
}

describe("model control", () => {
  const luna = { id: "openai:gpt-6-luna", label: "GPT-6 Luna", provider: "openai", window: 1_050_000 };
  const sol = { id: "openai:gpt-6.2-sol", label: "GPT-6.2 Sol", provider: "openai", window: 1_050_000 };
  const qwen = { id: "ollama:qwen3.8:27b", label: "qwen3.8:27b", provider: "ollama", local: true };
  const menu = {
    sections: [
      { title: "Recent", models: [luna, qwen] },
      { title: "OpenAI", models: [sol] },
    ],
    pinned: [],
  };

  it("shows the current model and switches from the short list", async () => {
    vi.spyOn(api, "modelMenu").mockResolvedValue(menu);
    const { store, controller } = renderWithStore(<Composer />);
    const setModel = vi.spyOn(controller, "setModel");
    act(() => store.setState({ models: [luna, sol, qwen], model: luna.id }));

    fireEvent.click(screen.getByRole("button", { name: "GPT-6 Luna" }));
    const current = await screen.findByRole("menuitemradio", { name: /GPT-6 Luna/ });
    expect(current).toHaveAttribute("aria-checked", "true");
    expect(current).toHaveTextContent(/^GPT-6 Luna$/); // the name alone, no window


    fireEvent.click(screen.getByRole("menuitemradio", { name: /GPT-6.2 Sol/ }));
    expect(setModel).toHaveBeenCalledWith(sol.id);
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("opens the API keys from the menu", async () => {
    vi.spyOn(api, "modelMenu").mockResolvedValue(menu);
    const { store } = renderWithStore(<Composer />);
    act(() => store.setState({ models: [luna], model: luna.id }));

    fireEvent.click(screen.getByRole("button", { name: "GPT-6 Luna" }));
    fireEvent.click(await screen.findByRole("menuitem", { name: "API keys…" }));
    expect(store.getState().apiKeys).toEqual({ open: true, required: null });
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("shows the effort and changes it from a row at the foot of the menu", async () => {
    const settings = {
      settings: { effort: null, output: null, context: null },
      effort: { levels: ["low", "medium", "xhigh"], default: "medium", current: "medium" },
      output: { choices: [8_192, 128_000], default: 128_000 },
      context: null,
    };
    vi.spyOn(api, "modelMenu").mockResolvedValue(menu);
    vi.spyOn(api, "modelSettings").mockResolvedValue(settings);
    const save = vi.spyOn(api, "saveModelSettings").mockResolvedValue({
      ...settings,
      settings: { ...settings.settings, effort: "xhigh" },
      effort: { ...settings.effort, current: "xhigh" },
    });
    const { store, controller } = renderWithStore(<Composer />);
    const send = vi.spyOn(controller.transport, "send").mockReturnValue(true);
    vi.spyOn(controller.transport, "isOpen").mockReturnValue(true);
    act(() => store.setState({ models: [luna], model: luna.id }));

    fireEvent.click(await screen.findByRole("button", { name: "GPT-6 Luna Medium" }));
    const effortRow = await screen.findByRole("menuitem", { name: /Effort\s*Medium/ });
    // The output length is in the config file only.
    expect(screen.queryByRole("menuitem", { name: /Output length/ })).toBeNull();
    // Passing over a row opens nothing; a click does.
    fireEvent.mouseEnter(effortRow);
    expect(screen.queryByRole("menu", { name: "Effort" })).toBeNull();
    fireEvent.click(effortRow);
    fireEvent.click(within(screen.getByRole("menu", { name: "Effort" })).getByRole("menuitemradio", { name: "Extra high" }));
    // Only the list closes; the menu stays, its row showing the new level.
    expect(screen.queryByRole("menu", { name: "Effort" })).toBeNull();
    expect(screen.getByRole("menu", { name: "Model" })).toBeInTheDocument();
    expect(await screen.findByRole("menuitem", { name: /Effort\s*Extra high/ })).toHaveFocus();
    await waitFor(() => expect(screen.getByRole("button", { name: "GPT-6 Luna Extra high" })).toBeInTheDocument());
    expect(save).toHaveBeenCalledWith(luna.id, { effort: "xhigh", output: null, context: null });
    // The running session rebuilds on the same model to pick it up.
    expect(send).toHaveBeenCalledWith({ type: "command", command: "set_model", arg: luna.id });
  });

  it("lists a local model's context sizes with their memory, and takes a typed size", async () => {
    const unmeasured = {
      settings: { effort: null, output: null, context: null },
      effort: null,
      output: null,
      context: {
        local: true,
        choices: [{ tokens: 32_768, memory: null, on_gpu: null }],
        auto: 32_768,
        min: 16_384,
        max: 262_144,
        measured: false,
      },
    };
    vi.spyOn(api, "modelMenu").mockResolvedValue(menu);
    vi.spyOn(api, "modelSettings").mockResolvedValue(unmeasured);
    const measure = vi.spyOn(api, "measureModel").mockResolvedValue({
      ...unmeasured,
      context: {
        ...unmeasured.context,
        choices: [
          { tokens: 32_768, memory: 16.4 * 2 ** 30, on_gpu: true },
          { tokens: 131_072, memory: 31 * 2 ** 30, on_gpu: false },
          { tokens: 262_144, memory: null, on_gpu: null, too_large: true },
        ],
        measured: true,
      },
    });
    const save = vi.spyOn(api, "saveModelSettings").mockResolvedValue(unmeasured);
    const { store } = renderWithStore(<Composer />);
    act(() => store.setState({ models: [qwen], model: qwen.id }));
    await waitFor(() => expect(store.getState().modelSettings?.model).toBe(qwen.id));

    fireEvent.click(screen.getByRole("button", { name: "qwen3.8:27b" }));
    fireEvent.click(await screen.findByRole("menuitem", { name: /Context size\s*Auto/ }));
    expect(await screen.findByText("16.4 GB")).toBeInTheDocument();
    // The note goes once the figures are in.
    await waitFor(() => expect(screen.queryByText("Measuring memory…")).toBeNull());
    expect(screen.getByText("31.0 GB · partly on CPU")).toBeInTheDocument();
    // A size Ollama refused can't be chosen.
    expect(screen.getByRole("menuitemradio", { name: /^256k/ })).toBeDisabled();
    expect(screen.getByText("too large")).toBeInTheDocument();
    expect(screen.getByRole("menuitemradio", { name: /^Auto/ })).toHaveAttribute("aria-checked", "true");
    expect(measure).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("menuitemradio", { name: "Custom…" }));
    const input = screen.getByRole("textbox", { name: "Context size in tokens" });
    expect(screen.getByRole("menu", { name: "Model" })).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "8k" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(screen.getByText("Between 16k and 256k")).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "100k" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(save).toHaveBeenCalledWith(qwen.id, { effort: null, output: null, context: 100_000 });
  });

  it("opens the menu from /model", async () => {
    vi.spyOn(api, "modelMenu").mockResolvedValue(menu);
    const { store } = renderWithStore(<Composer />);
    act(() => store.setState({ models: [luna], model: luna.id }));

    const composer = screen.getByRole("textbox", { name: "Message" });
    fireEvent.change(composer, { target: { value: "/model" } });
    fireEvent.keyDown(composer, { key: "Enter" });
    expect(await screen.findByRole("menu", { name: "Model" })).toBeInTheDocument();
  });

  it("searches the full list, pins, and takes a typed provider:model", async () => {
    vi.spyOn(api, "liveModels").mockResolvedValue({ models: [luna, sol, qwen] });
    const pin = vi.spyOn(api, "pinModel").mockResolvedValue({ pinned: [sol.id] });
    const { store, controller } = renderWithStore(<MoreModels />);
    const setModel = vi.spyOn(controller, "setModel");
    act(() => {
      store.setState({ models: [luna, sol, qwen], model: luna.id });
      controller.openMoreModels();
    });

    const search = screen.getByRole("searchbox", { name: "Search models" });
    fireEvent.change(search, { target: { value: "sol" } });
    expect(screen.queryByText("GPT-6 Luna")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Pin GPT-6.2 Sol" }));
    await waitFor(() => expect(pin).toHaveBeenCalledWith(sol.id, true));
    expect(screen.getByRole("button", { name: "Pin GPT-6.2 Sol" })).toHaveAttribute("aria-pressed", "true");

    fireEvent.change(search, { target: { value: "openrouter:some/model" } });
    fireEvent.click(screen.getByRole("button", { name: "Use openrouter:some/model" }));
    expect(setModel).toHaveBeenCalledWith("openrouter:some/model");
  });
});

describe("composer", () => {
  it("keeps the draft when Enter is pressed while a turn is running", () => {
    const { store } = renderWithStore(<Composer />);
    act(() => store.setState({ busy: true }));

    const composer = screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement;
    fireEvent.change(composer, { target: { value: "and then plot the pressure" } });
    fireEvent.keyDown(composer, { key: "Enter" });

    expect(composer.value).toBe("and then plot the pressure");
  });

  it("shows the send key only with text, and a stop button while a turn runs", () => {
    const { store, controller } = renderWithStore(<Composer />);
    const send = vi.spyOn(controller, "send").mockReturnValue(true);
    const stop = vi.spyOn(controller, "stop").mockImplementation(() => {});
    const composer = screen.getByRole("textbox", { name: "Message" });
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull();

    fireEvent.change(composer, { target: { value: "plot the pressure" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(send).toHaveBeenCalledWith("plot the pressure");

    act(() => store.setState({ busy: true }));
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(stop).toHaveBeenCalled();
  });
});

describe("context ring and view", () => {
  const meter = {
    model: "openai:gpt-6",
    window: 200_000,
    window_source: "provider" as const,
    compact_at: 170_000,
    offload_at: 119_000,
    used: 52_000,
    estimated: false,
  };

  it("opens the view from the ring, with the tool results expandable", async () => {
    const report = vi.spyOn(api, "context").mockResolvedValue({
      meter,
      breakdown: [
        { key: "system", label: "System prompt", tokens: 12_000 },
        { key: "tool_results", label: "Tool results", tokens: 40_000 },
      ],
      largest: [{ label: "Read · src/model.jl", tokens: 9_000 }],
      moved: { results: 4, compactions: 0 },
    });
    const { store } = renderWithStore(<Composer />);
    act(() => store.setState({ sessionId: "s1", context: meter }));

    const ring = screen.getByRole("button", { name: "Context: 52k of 200k tokens (26%)" });
    fireEvent.click(ring);
    await waitFor(() => expect(screen.getByText("System prompt")).toBeInTheDocument());
    expect(screen.getByText("Free space")).toBeInTheDocument();
    expect(screen.getByText("Moved out: 4 tool results to files.")).toBeInTheDocument();
    expect(report).toHaveBeenCalledWith("s1");

    // The largest results show only when asked for.
    expect(screen.queryByText("Read · src/model.jl")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Tool results" }));
    expect(screen.getByText("Read · src/model.jl")).toBeInTheDocument();

    // The ring, or a click anywhere else, closes it.
    fireEvent.click(ring);
    expect(store.getState().contextView.open).toBe(false);
    fireEvent.click(ring);
    expect(screen.getByRole("dialog", { name: "Context" })).toBeInTheDocument();
    fireEvent.click(document.body);
    expect(screen.queryByRole("dialog", { name: "Context" })).toBeNull();
    report.mockRestore();
  });

  it("closes on Escape in the message box without stopping the turn", () => {
    const { store, controller } = renderWithStore(<Composer />);
    const stop = vi.spyOn(controller, "stop").mockImplementation(() => {});
    act(() => store.setState({ sessionId: "s1", context: meter, busy: true }));
    act(() => store.getState().setContextView({ open: true }));

    const box = screen.getByRole("textbox", { name: "Message" });
    fireEvent.keyDown(box, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Context" })).toBeNull();
    expect(stop).not.toHaveBeenCalled();

    // With the view closed, Escape stops the turn as before.
    fireEvent.keyDown(box, { key: "Escape" });
    expect(stop).toHaveBeenCalledTimes(1);
  });

  it("stays open when the click that opened it came from a button now gone", () => {
    const { store } = renderWithStore(<Composer />);
    act(() => store.setState({ sessionId: "s1", context: meter }));
    act(() => store.getState().setContextView({ open: true }));

    // As with /context sent from the Send button: the button leaves the page
    // before its click reaches the document.
    const gone = document.createElement("button");
    document.body.append(gone);
    gone.addEventListener("click", () => gone.remove());
    fireEvent.click(gone);
    expect(screen.getByRole("dialog", { name: "Context" })).toBeInTheDocument();
  });

  it("changes where auto-compaction runs from the foot of the view", async () => {
    vi.spyOn(api, "context").mockResolvedValue({
      meter,
      breakdown: [{ key: "system", label: "System prompt", tokens: 12_000 }],
      largest: [],
      moved: { results: 0, compactions: 0 },
      compact_percent: 85,
    });
    const save = vi.spyOn(api, "saveContextSettings").mockResolvedValue({ compact_percent: 70 });
    const { store, controller } = renderWithStore(<Composer />);
    const send = vi.spyOn(controller.transport, "send").mockReturnValue(true);
    vi.spyOn(controller.transport, "isOpen").mockReturnValue(true);
    act(() => store.setState({ sessionId: "s1", context: meter, model: "openai:gpt-6" }));

    fireEvent.click(screen.getByRole("button", { name: /^Context:/ }));
    const setting = await screen.findByRole("button", { name: "85%" });
    // The choices show only when asked for.
    expect(screen.queryByRole("radiogroup", { name: "Auto-compact at" })).toBeNull();
    fireEvent.click(setting);
    const choice = within(screen.getByRole("radiogroup", { name: "Auto-compact at" })).getByRole("radio", { name: "70%" });
    // In a browser React removes the list before the click reaches the document;
    // the view must not take that for a click outside it.
    document.body.addEventListener("click", () => choice.remove(), { once: true });
    fireEvent.click(choice);

    expect(save).toHaveBeenCalledWith({ compact_percent: 70 });
    expect(await screen.findByRole("button", { name: "70%" })).toBeInTheDocument();
    // The view stays open, and the session rebuilds on the same model to pick it up.
    expect(screen.getByRole("dialog", { name: "Context" })).toBeInTheDocument();
    expect(send).toHaveBeenCalledWith({ type: "command", command: "set_model", arg: "openai:gpt-6" });
  });

  it("shows an empty ring before the first model call", () => {
    const { store } = renderWithStore(<Composer />);
    act(() => store.setState({ context: { ...meter, used: null } }));
    expect(screen.getByRole("button", { name: "Context: No model calls yet" })).toBeInTheDocument();
  });
});

describe("Thread rendering", () => {
  it("shows the welcome screen when empty", () => {
    renderWithStore(<Thread />);
    expect(screen.getByText(/What would you like to explore/)).toBeInTheDocument();
  });

  it("lets an installed capability name the welcome screen and set its prompts", () => {
    const { store } = renderWithStore(<Thread />);
    act(() =>
      store.getState().setConfig({
        sim: "jutuldarcy",
        simDetails: { jutuldarcy: { display_name: "JutulDarcy", examples: ["Sim prompt."] } },
        branding: { display_name: "MyApp", tagline: "Build a model.", examples: ["Build the model."] },
      }),
    );
    expect(screen.getByText(/explore with MyApp/)).toBeInTheDocument();
    expect(screen.getByText("Build a model.")).toBeInTheDocument();
    expect(screen.getByText("Build the model.")).toBeInTheDocument();
    expect(screen.queryByText("Sim prompt.")).toBeNull();
  });

  it("falls back to the simulator when nothing is branded", () => {
    const { store } = renderWithStore(<Thread />);
    act(() =>
      store.getState().setConfig({
        sim: "jutuldarcy",
        simDetails: { jutuldarcy: { display_name: "JutulDarcy", examples: ["Sim prompt."] } },
      }),
    );
    expect(screen.getByText(/explore with JutulDarcy/)).toBeInTheDocument();
    expect(screen.getByText("Sim prompt.")).toBeInTheDocument();
  });

  it("renders a streamed exchange: user, assistant, tool", () => {
    const { store } = renderWithStore(<Thread />);
    act(() => store.getState().addUser("run a sim"));
    drive(
      store,
      { type: "text", text: "On it." },
      { type: "tool", event: "requested", name: "run_julia", label: "run_julia", tool_call_id: "1", args: { code: "1+1" } },
      { type: "tool", event: "finished", name: "run_julia", tool_call_id: "1", content: "2" },
    );
    expect(screen.getByText("run a sim")).toBeInTheDocument();
    expect(screen.getByText("On it.")).toBeInTheDocument();
    expect(screen.getByText("run_julia")).toBeInTheDocument();
    expect(screen.getByText("done")).toBeInTheDocument();
  });

  it("counts up on a tool that is still running, and stops when it finishes", () => {
    vi.useFakeTimers();
    try {
      const { store, container } = renderWithStore(<Thread />);
      drive(store, {
        type: "tool",
        event: "requested",
        name: "setup_mesh",
        label: "setup_mesh",
        tool_call_id: "1",
        args: {},
      });
      // Nothing yet: a number that flashes up and vanishes is noise.
      expect(container.querySelector(".elapsed")).toBeNull();
      act(() => void vi.advanceTimersByTime(65_000));
      expect(container.querySelector(".elapsed")?.textContent).toBe("1:05");
      drive(store, { type: "tool", event: "finished", name: "setup_mesh", tool_call_id: "1", content: "ok" });
      expect(container.querySelector(".elapsed")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows the thinking indicator only while working", () => {
    const { store, container } = renderWithStore(<Thread />);
    act(() => store.getState().beginWorking());
    expect(container.querySelector(".working")).toBeInTheDocument();
    drive(store, { type: "text", text: "hi" });
    expect(container.querySelector(".working")).toBeNull();
  });
});

describe("canvas interaction", () => {
  it("mounts the panel an extension registered for a view's kind", () => {
    registerPanel("test-map", ({ view }) => <div>map of {view.title}</div>);
    const { store } = renderWithStore(
      <>
        <Thread />
        <Canvas />
      </>,
    );
    drive(store, { type: "viz", url: "/wells.json", kind: "test-map", slot: "m", title: "Wells" });
    expect(screen.getByText("map of Wells")).toBeInTheDocument();
    expect(screen.getByText("View", { selector: ".viz-chip .s" })).toBeInTheDocument();
  });

  it("a view's tab has its own remove button", () => {
    const { store } = renderWithStore(<Canvas />);
    drive(store, { type: "viz", url: "/plot.html", kind: "plot", slot: "fig", title: "My figure" });
    fireEvent.click(screen.getByRole("button", { name: "Remove My figure" }));
    expect(store.getState().viewOrder).toEqual([]);
  });

  it("a viz chip opens the canvas to that view", () => {
    const { store } = renderWithStore(
      <>
        <Thread />
        <Canvas />
      </>,
    );
    drive(store, { type: "viz", url: "/plot.html", kind: "plot", slot: "fig", title: "My figure" });
    // Opening happens in the store on viz; closing then re-opening via the chip.
    act(() => store.getState().closeCanvas());
    expect(store.getState().canvasOpen).toBe(false);
    fireEvent.click(screen.getByText("My figure", { selector: ".viz-chip .t" }));
    expect(store.getState().canvasOpen).toBe(true);
    expect(store.getState().activeView).toBe("slot:fig");
  });

  it("a resumed session's poster clears the loader without looping", () => {
    // A session resumed from disk has no plot server, so its figure replays as
    // its PNG poster, and a poster the browser has already decoded reports its
    // load from the panel's mount check. Since that report re-renders the canvas,
    // the whole path has to settle rather than feed itself another report.
    const complete = vi.spyOn(HTMLImageElement.prototype, "complete", "get").mockReturnValue(true);
    const { store, container } = renderWithStore(<Canvas />);
    act(() => store.getState().setSession("s1", ""));
    act(() =>
      store.getState().replay([
        {
          type: "viz",
          url: "/sessions/s1/artifacts/fig.png",
          kind: "plot",
          slot: "fig",
          title: "My figure",
          live: false,
          poster: "/sessions/s1/artifacts/fig.png",
        },
      ]),
    );
    expect(container.querySelector("img")?.getAttribute("src")).toBe("/sessions/s1/artifacts/fig.png");
    expect(container.querySelector(".canvas-loading")).toBeNull();
    complete.mockRestore();
  });
});

describe("a part that fails to render", () => {
  it("shows a note in its place and leaves the rest of the thread", () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const { store } = renderWithStore(<Thread />);
    drive(
      store,
      { type: "text", text: "Before the bad card." },
      // A model's malformed todo list: the card's preview reads each entry.
      { type: "tool", event: "requested", name: "write_todos", tool_call_id: "t", args: { todos: [null] } },
      { type: "turn_end", text: "" },
    );
    expect(screen.getByText("Could not show this item.")).toBeInTheDocument();
    expect(screen.getByText("Before the bad card.")).toBeInTheDocument();
    error.mockRestore();
  });
});

describe("approval", () => {
  it("renders the request and clears it on approve", () => {
    const { store } = renderWithStore(<Thread />);
    drive(store, {
      type: "interrupt",
      interrupt_id: "x",
      actions: [{ name: "execute", label: "execute", args: { command: "ls" } }],
      allowed_decisions: ["approve", "reject"],
      allowlist: [],
    });
    expect(screen.getByRole("alert")).toHaveTextContent("Approve execute?");
    fireEvent.click(screen.getByRole("button", { name: "approve" }));
    expect(store.getState().pending).toBeNull();
    expect(screen.queryByText(/Approve execute\?/)).toBeNull();
  });
});

describe("session list live markers", () => {
  const entry = (id: string, extra: Partial<HistoryEntry> = {}): HistoryEntry => ({
    id,
    title: `Session ${id}`,
    started: "2026-01-01T00:00:00Z",
    last_active: "2026-01-01T00:00:00Z",
    sim: "jutuldarcy",
    ...extra,
  });

  function renderList(history: HistoryEntry[]) {
    const rendered = renderWithStore(<Sidebar collapsed={false} />);
    act(() => rendered.store.getState().setHistory(history));
    return rendered;
  }

  it("marks only the sessions the server calls live", () => {
    const { container } = renderList([
      entry("a", { live: true }),
      entry("b"),
      entry("c", { live: false }),
    ]);
    const items = container.querySelectorAll(".history-item");
    expect(items.length).toBe(3);
    expect(items[0].querySelector(".h-live")).not.toBeNull();
    expect(items[1].querySelector(".h-live")).toBeNull();
    expect(items[2].querySelector(".h-live")).toBeNull();
  });

  it("labels the marker for screen readers and on hover", () => {
    const { container } = renderList([entry("a", { live: true })]);
    const dot = container.querySelector(".h-live");
    expect(dot?.getAttribute("aria-label")).toMatch(/Julia session intact/);
    expect(dot?.getAttribute("title")).toBe(dot?.getAttribute("aria-label"));
  });

  it("keeps the title truncatable next to the marker", () => {
    // The dot turned .h-title into a flex row, which cannot ellipsis a bare text
    // node, so the title has to keep its own element.
    const { container } = renderList([entry("a", { live: true })]);
    expect(container.querySelector(".h-title-text")?.textContent).toBe("Session a");
  });
});

describe("session list freshness", () => {
  const listed = (id: string): HistoryEntry => ({
    id,
    title: `Session ${id}`,
    started: "2026-01-01T00:00:00Z",
    last_active: "2026-01-01T00:00:00Z",
    sim: "jutuldarcy",
    live: true,
  });

  it("re-reads the list when a new chat replaces the current session", async () => {
    // newChat used to leave the sidebar showing whatever it had until the next
    // turn ended, so the session just left kept a stale place and marker.
    const { store, controller } = renderWithStore(<Sidebar collapsed={false} />);
    const history = vi.spyOn(api, "history").mockResolvedValue([listed("a")]);
    vi.spyOn(controller, "startSession").mockResolvedValue(undefined);
    await act(() => controller.newChat());
    expect(history).toHaveBeenCalled();
    expect(store.getState().history.map((h) => h.id)).toEqual(["a"]);
    vi.restoreAllMocks();
  });

  it("re-reads the list when the tab comes back to the front", async () => {
    // A host can be evicted while the tab is in the background, which clears a
    // live marker this client never hears about.
    renderWithStore(<Sidebar collapsed={false} />);
    const history = vi.spyOn(api, "history").mockResolvedValue([listed("a")]);
    document.dispatchEvent(new Event("visibilitychange"));
    await act(async () => {});
    expect(history).toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});

describe("switching sessions", () => {
  it("marks the resumed session live without waiting for the replay", async () => {
    // The refresh used to run after the conversation had been re-fetched and laid
    // down, so on a long chat the dot appeared well after the click. Liveness is
    // already settled once the resume answers, before the socket brings the replay.
    const { controller } = renderWithStore(<Sidebar collapsed={false} />);
    vi.spyOn(api, "resumeSession").mockResolvedValue({
      session_id: "a",
      kernel_restarted: true,
    });
    const history = vi.spyOn(api, "history").mockResolvedValue([]);

    void controller.resume("a");
    await act(async () => {});

    expect(history).toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});

describe("version line", () => {
  it("names Makker and the simulator quietly, and an available update", () => {
    const { store, container } = renderWithStore(<Sidebar collapsed={false} />);
    expect(container.querySelector(".sidebar-foot")).toBeNull();
    act(() =>
      store.setState({
        versions: {
          makker: "0.2.1",
          update: null,
          simulator: { package: "JutulDarcy", version: "0.4.0" },
        },
      }),
    );
    expect(container.querySelector(".sidebar-versions")?.textContent).toBe(
      "Makker 0.2.1 · JutulDarcy 0.4.0",
    );
    expect(container.querySelector(".sidebar-update")).toBeNull();
    act(() => store.setState({ versions: { makker: "0.2.1", update: "0.2.2", simulator: null } }));
    const update = container.querySelector(".sidebar-update");
    expect(update?.textContent).toBe("Makker 0.2.2 available");
    expect(update?.tagName).toBe("BUTTON"); // opens the update dialog
  });

  it("offers the environment update when only that is due", () => {
    const { store, container } = renderWithStore(<Sidebar collapsed={false} />);
    act(() =>
      store.setState({
        versions: { makker: "0.2.1", update: null, simulator: null, env_update: true },
      }),
    );
    expect(container.querySelector(".sidebar-update")?.textContent).toBe("Update environment");
  });
});
