// The REST surface of the server (everything except the per-turn WebSocket),
// typed. Mirrors the FastAPI routes in interfaces/server/routes/.

import type { HostContext } from "./hostContext";
import type { ContextReport } from "./protocol";
import { withBase } from "./basePath";

export interface ModelInfo {
  id: string;
  label: string;
  provider: string;
  note?: string | null;
  /** Input context in tokens, when the provider data reports it. */
  window?: number | null;
  /** Runs on this machine. */
  local?: boolean;
}

/** The model menu's short list, from `GET /models/menu`. */
export interface ModelMenu {
  sections: { title: string; models: ModelInfo[] }[];
  pinned: string[];
}

/** What a model can be set to, and how it is set, from `GET /models/settings`. */
export interface ModelSettings {
  settings: { effort: string | null; output: number | null; context: number | null };
  /** Absent when the model offers no effort levels. */
  effort: { levels: string[]; default: string | null; current: string | null } | null;
  output: { choices: number[]; default: number } | null;
  /** Each size, with the memory it takes here for a local model (once measured).
   *  For a hosted model a size only caps the window Makker compacts against. */
  context: {
    local: boolean;
    /** `memory` is what the model took when loaded at that size (all of it, the
     *  part on the CPU included); `too_large` when Ollama refused the load. */
    choices: { tokens: number; memory: number | null; on_gpu: boolean | null; too_large?: boolean }[];
    auto: number | null;
    min: number;
    max: number;
    measured: boolean;
  } | null;
}

export type ModelSettingsPatch = Partial<ModelSettings["settings"]>;

export interface SimDetails {
  display_name?: string;
  examples?: string[];
}

/** What an installed capability calls this UI, when it takes the simulator's place
 *  on the welcome screen. Each field falls back to the simulator's own on its own. */
export interface Branding {
  display_name?: string;
  tagline?: string;
  examples?: string[];
}

export interface HistoryEntry {
  id: string;
  title: string;
  started: string;
  last_active: string;
  sim: string;
  /** The session's host is still in memory, so a resume keeps its Julia REPL state.
   * Optional: an older server does not send it, and absent must not read as live. */
  live?: boolean;
}

export interface SimulatorsResponse {
  simulators: string[];
  default: string | null;
  details: Record<string, SimDetails>;
  branding?: Branding | null;
}

/** What the server runs, from `GET /version`. Each part is `null` when unknown. */
export interface VersionInfo {
  makker: string | null;
  /** A newer Makker release, when the update check knows of one. */
  update: string | null;
  simulator: { package: string; version: string | null } | null;
  /** This folder's Julia environment is older than this version supports. */
  env_update?: boolean;
  /** Identifies the server process; it changes when the server restarts. */
  boot?: string | null;
}

/** A simulator package in the folder's Julia environment that needs updating. */
export interface OutdatedPackage {
  package: string;
  version: string;
  required: string;
}

/** An update started from the browser, as it runs. */
export interface UpdateJob {
  state: "running" | "restarting" | "done" | "failed";
  install: boolean;
  error: string | null;
  /** Output lines after the `since` the request asked for. */
  lines: string[];
  /** Pass as `since` next time to get only newer lines. */
  next: number;
}

/** What an update would do here, from `GET /update`. */
export interface UpdateStatus {
  current: string | null;
  latest: string | null;
  /** Served on this computer, so the browser may update it. */
  local: boolean;
  /** Makker can upgrade itself from here; otherwise `install_note` says how. */
  can_install: boolean;
  install_note: string | null;
  /** Upgrading Makker continues in a separate window, which it then runs in. */
  separate_window: boolean;
  env: { outdated: OutdatedPackage[]; sysimage: boolean } | null;
  job: UpdateJob | null;
}

export type SysimageChoice = "rebuild" | "off";

export interface ModelsResponse {
  default: string | null;
  providers: string[];
  models: ModelInfo[];
}

/** One provider's API-key state, from `GET /credentials`. */
export interface CredentialInfo {
  provider: string;
  label: string;
  env_var: string;
  is_set: boolean;
  masked: string | null;
  source: "file" | "environment" | "none";
  shadowed: boolean;
}

/** An HTTP error that carries the server's parsed `detail`, so callers can act on a
 *  structured error (e.g. a `credential_required` create-session refusal). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

// Endpoints use unprefixed paths; apply the deployment prefix only at this boundary.
function request(path: string, init?: RequestInit): Promise<Response> {
  return fetch(withBase(path), init);
}

async function getJSON<T>(url: string, fallback: T): Promise<T> {
  try {
    const resp = await request(url);
    if (!resp.ok) return fallback;
    return (await resp.json()) as T;
  } catch {
    return fallback;
  }
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const resp = await request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    // FastAPI wraps an error in {detail: ...}; surface the detail so a structured
    // error (a dict) stays usable instead of collapsing to a string.
    const text = await resp.text().catch(() => resp.statusText);
    let detail: unknown = text;
    try {
      const parsed = JSON.parse(text);
      detail = "detail" in parsed ? parsed.detail : parsed;
    } catch {
      /* not JSON; keep the text */
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

export const api = {
  simulators: () =>
    getJSON<SimulatorsResponse>("/simulators", {
      simulators: [],
      default: null,
      details: {},
    }),

  version: () =>
    getJSON<VersionInfo>("/version", { makker: null, update: null, simulator: null }),

  /** `null` when the server can't be reached, as while it restarts. */
  updateStatus: async (since = 0): Promise<UpdateStatus | null> => {
    try {
      const resp = await request(`/update?since=${since}`);
      return resp.ok ? ((await resp.json()) as UpdateStatus) : null;
    } catch {
      return null;
    }
  },

  startUpdate: (body: { install: boolean; sysimage: SysimageChoice | null }) =>
    postJSON<UpdateStatus>("/update", body),

  models: () =>
    getJSON<ModelsResponse>("/models", { default: null, providers: [], models: [] }),

  liveModels: () => getJSON<{ models: ModelInfo[] }>("/models/live", { models: [] }),

  modelMenu: (current: string | null) =>
    getJSON<ModelMenu>(
      current ? `/models/menu?current=${encodeURIComponent(current)}` : "/models/menu",
      { sections: [], pinned: [] },
    ),

  pinModel: (model: string, pinned: boolean) =>
    postJSON<{ pinned: string[] }>("/models/pinned", { model, pinned }),

  modelSettings: (model: string) =>
    getJSON<ModelSettings | null>(`/models/settings?model=${encodeURIComponent(model)}`, null),

  saveModelSettings: (model: string, settings: ModelSettings["settings"]) =>
    postJSON<ModelSettings>("/models/settings", { model, ...settings }),

  measureModel: (model: string) => postJSON<ModelSettings>("/models/measure", { model }),

  saveContextSettings: (settings: { compact_percent: number }) =>
    postJSON<{ compact_percent: number }>("/context/settings", settings),

  credentials: async (): Promise<CredentialInfo[]> => {
    const data = await getJSON<{ path: string; providers: CredentialInfo[] }>(
      "/credentials",
      {
        path: "",
        providers: [],
      },
    );
    return data.providers;
  },

  setCredential: (provider: string, value: string) =>
    postJSON<{ provider: string; env_var: string; path: string }>("/credentials", {
      provider,
      value,
    }),

  history: async (limit = 40): Promise<HistoryEntry[]> => {
    const data = await getJSON<{ sessions: HistoryEntry[] }>(
      `/sessions/history?limit=${limit}`,
      { sessions: [] },
    );
    return data.sessions;
  },

  // `host_context` is the embedding application's current selection and
  // `host_api` is where its own API listens. Both are sent on resume as well as
  // create because they describe the application now, not the stored session:
  // omitting the selection (a UI opened outside the host) leaves the session's
  // stored one alone, while omitting the URL drops the host's tools, which is
  // right, because the port it was on is gone.
  createSession: (body: {
    sim?: string;
    model?: string;
    host_context?: HostContext;
    host_api?: string;
  }) => postJSON<{ session_id: string }>("/sessions", body),

  resumeSession: (
    id: string,
    body: { sim?: string; model?: string; host_context?: HostContext; host_api?: string },
  ) =>
    postJSON<{ session_id: string; kernel_restarted: boolean }>(
      `/sessions/${id}/resume`,
      body,
    ),

  deleteSession: (id: string) =>
    request(`/sessions/${id}`, { method: "DELETE" }).catch(() => undefined),

  context: (id: string) => getJSON<ContextReport | null>(`/sessions/${id}/context`, null),

  uploadFile: async (id: string, file: File): Promise<{ path: string }> => {
    const fd = new FormData();
    fd.append("file", file);
    const resp = await request(`/sessions/${id}/upload`, { method: "POST", body: fd });
    if (!resp.ok) throw new Error(await resp.text().catch(() => resp.statusText));
    return (await resp.json()) as { path: string };
  },

  transcriptUrl: (id: string, fmt: "html" | "md") =>
    withBase(`/sessions/${id}/transcript?format=${fmt}`),

  memoryUrl: (id: string) => withBase(`/sessions/${id}/memory`),
};
