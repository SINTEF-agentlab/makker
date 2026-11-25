import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./api";
import { Controller, preparePopoutDocument } from "./controller";
import { createSessionStore } from "./store";

describe("plot popout placeholder", () => {
  it("treats a model-provided title as text, never markup", () => {
    const doc = document.implementation.createHTMLDocument();
    preparePopoutDocument(doc, '<script data-test="payload">boom()</script>');

    expect(doc.title).toBe('<script data-test="payload">boom()</script>');
    expect(doc.querySelector("script")).toBeNull();
    expect(doc.body.textContent).toBe("Preparing an independent view…");
  });
});

describe("model catalog refresh", () => {
  it("adds API-listed models to the picker after startup", async () => {
    const staticModel = { id: "openai:gpt-5.4-mini", label: "gpt-5.4-mini", provider: "openai" };
    const liveModel = {
      id: "openai:gpt-6-luna",
      label: "gpt-6-luna",
      provider: "openai",
      note: "API-listed · tool support unverified",
    };
    const spies = [
      vi.spyOn(api, "simulators").mockResolvedValue({
        simulators: ["jutuldarcy"], default: "jutuldarcy", details: {},
      }),
      vi.spyOn(api, "models").mockResolvedValue({
        default: staticModel.id, providers: ["openai"], models: [staticModel],
      }),
      vi.spyOn(api, "credentials").mockResolvedValue([]),
      vi.spyOn(api, "liveModels").mockResolvedValue({ models: [staticModel, liveModel] }),
    ];
    const store = createSessionStore();
    const controller = new Controller(store);
    const start = vi.spyOn(controller, "startSession").mockResolvedValue();
    const history = vi.spyOn(controller, "refreshHistory").mockResolvedValue();
    const context = vi.spyOn(controller, "refreshContextWindow").mockResolvedValue();

    try {
      await controller.init();
      await vi.waitFor(() => expect(store.getState().models).toContainEqual(liveModel));
    } finally {
      for (const spy of [...spies, start, history, context]) spy.mockRestore();
    }
  });
});

describe("session startup failures", () => {
  it("shows the server refusal and leaves the composer idle", async () => {
    const create = vi
      .spyOn(api, "createSession")
      .mockRejectedValue(new ApiError(409, "Rebuild it with: makker sysimage build"));
    const store = createSessionStore();
    store.setState({ sim: "jutuldarcy", busy: true, working: true });
    const controller = new Controller(store);

    await controller.startSession();

    const state = store.getState();
    expect(state.busy).toBe(false);
    expect(state.working).toBe(false);
    expect(state.items.at(-1)).toMatchObject({
      kind: "sys-note",
      text: "Rebuild it with: makker sysimage build",
      level: "warn",
    });
    create.mockRestore();
  });

  it("retries session creation for a prompt sent after startup failed", async () => {
    const create = vi
      .spyOn(api, "createSession")
      .mockRejectedValue(new ApiError(409, "The system image is stale."));
    const store = createSessionStore();
    store.setState({ sim: "jutuldarcy" });
    const controller = new Controller(store);
    vi.spyOn(controller, "sendCanvasHint").mockImplementation(() => undefined);

    await controller.startSession();
    controller.send("Run the model");
    await vi.waitFor(() => expect(create).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(store.getState().busy).toBe(false));

    expect(store.getState().working).toBe(false);
    expect(store.getState().items).toContainEqual(
      expect.objectContaining({ kind: "user", text: "Run the model" }),
    );
    create.mockRestore();
  });
});

/** A WebSocket the test opens, feeds and reads by hand. */
class FakeSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 3;
  static all: FakeSocket[] = [];
  readyState = FakeSocket.CONNECTING;
  sent: Record<string, unknown>[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(public url: string) {
    FakeSocket.all.push(this);
  }

  send(data: string): void {
    this.sent.push(JSON.parse(data));
  }

  close(): void {
    this.readyState = FakeSocket.CLOSED;
  }

  connect(): void {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }

  receive(msg: unknown): void {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }
}

const IDLE = { type: "snapshot", messages: [], turn: [], busy: false, model: null };

describe("session lifecycle", () => {
  beforeEach(() => {
    FakeSocket.all = [];
    vi.stubGlobal("WebSocket", FakeSocket);
    vi.spyOn(api, "history").mockResolvedValue([]);
    vi.spyOn(api, "version").mockResolvedValue({ makker: null, update: null, simulator: null });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function setup() {
    const store = createSessionStore();
    store.setState({ sim: "jutuldarcy" });
    const controller = new Controller(store);
    vi.spyOn(controller, "sendCanvasHint").mockImplementation(() => undefined);
    return { store, controller };
  }

  it("sends a prompt typed while connecting after the snapshot, and keeps it shown", async () => {
    vi.spyOn(api, "createSession").mockResolvedValue({ session_id: "s1" });
    const { store, controller } = setup();

    controller.send("Run the model");
    await vi.waitFor(() => expect(FakeSocket.all).toHaveLength(1));
    const socket = FakeSocket.all[0];
    socket.connect();
    expect(socket.sent).toEqual([]); // held until the page shows the session

    socket.receive(IDLE);
    expect(socket.sent).toEqual([{ type: "prompt", text: "Run the model" }]);
    expect(store.getState().items).toContainEqual(
      expect.objectContaining({ kind: "user", text: "Run the model" }),
    );
    expect(store.getState().busy).toBe(true);
  });

  it("stays on its session, socket and all, when a switch is refused", async () => {
    vi.spyOn(api, "createSession").mockResolvedValue({ session_id: "s1" });
    vi.spyOn(api, "resumeSession").mockRejectedValue(
      new ApiError(409, "session is already open in another connection"),
    );
    const { store, controller } = setup();
    await controller.startSession();
    const socket = FakeSocket.all[0];
    socket.connect();
    socket.receive(IDLE);

    await controller.resume("s2");

    expect(socket.readyState).toBe(FakeSocket.OPEN);
    expect(FakeSocket.all).toHaveLength(1);
    expect(store.getState().sessionId).toBe("s1");
    expect(controller.transport.isOpen()).toBe(true);
    expect(store.getState().items.at(-1)).toMatchObject({ kind: "sys-note", level: "warn" });
  });

  it("shows a model switch only once the server confirms it", async () => {
    vi.spyOn(api, "createSession").mockResolvedValue({ session_id: "s1" });
    vi.spyOn(api, "modelWindow").mockResolvedValue({ model: "", window: null });
    const { store, controller } = setup();
    store.setState({ model: "openai:a" });
    await controller.startSession();
    const socket = FakeSocket.all[0];
    socket.connect();
    socket.receive({ ...IDLE, model: "openai:a" });

    controller.setModel("openai:b");
    expect(socket.sent.at(-1)).toEqual({ type: "command", command: "set_model", arg: "openai:b" });
    expect(store.getState().model).toBe("openai:a");
    socket.receive({ type: "error", message: "could not apply set_model: no such model" });
    expect(store.getState().model).toBe("openai:a"); // refused: nothing to roll back

    controller.setModel("openai:c");
    socket.receive({ type: "model", model: "openai:c" });
    expect(store.getState().model).toBe("openai:c");
    expect(store.getState().meta).toContain("openai:c");
  });

  it("shows the model a reattached session runs", async () => {
    vi.spyOn(api, "createSession").mockResolvedValue({ session_id: "s1" });
    vi.spyOn(api, "modelWindow").mockResolvedValue({ model: "", window: null });
    const { store, controller } = setup();
    store.setState({ model: "openai:a" });
    await controller.startSession();
    const socket = FakeSocket.all[0];
    socket.connect();
    socket.receive({ ...IDLE, model: "anthropic:z" });
    expect(store.getState().model).toBe("anthropic:z");
  });

  it("lets a past chat clicked while a new one starts win", async () => {
    let created!: (value: { session_id: string }) => void;
    vi.spyOn(api, "createSession").mockReturnValue(new Promise((r) => (created = r)));
    vi.spyOn(api, "resumeSession").mockResolvedValue({ session_id: "old", kernel_restarted: false });
    const { store, controller } = setup();

    const starting = controller.startSession();
    await controller.resume("old");
    created({ session_id: "new" });
    await starting;

    expect(store.getState().sessionId).toBe("old");
    expect(FakeSocket.all.map((s) => s.url)).toEqual([
      expect.stringContaining("/sessions/old/stream"),
    ]);
  });
});
