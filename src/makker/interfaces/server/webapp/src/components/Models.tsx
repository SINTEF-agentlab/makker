// The model control under the message box: the current model and its effort, a short
// menu of the models most likely wanted (built by the server, the same list the
// terminal shows), the current model's settings, the full catalog behind "More
// models…", and the API keys.

import { useEffect, useMemo, useRef, useState } from "react";

import type { ModelInfo, ModelMenu, ModelSettings } from "../api";
import { useController, useSel } from "../context";
import { formatTokens } from "../format";
import { CheckIcon, ChevronLeft, CloseIcon, StarIcon } from "../icons";

const EFFORT_LABELS: Record<string, string> = {
  none: "None",
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "Extra high",
  max: "Max",
};

function effortLabel(level: string): string {
  return EFFORT_LABELS[level] ?? level;
}

/** "16.6 GB": memory as the system reports it (in GiB). */
function formatMemory(bytes: number): string {
  return `${(bytes / 2 ** 30).toFixed(1)} GB`;
}

const BUSY_TITLE = "Available when the turn has finished";

/** "gpt-6-luna" for "openai:gpt-6-luna": the name without its provider. */
function bareName(id: string): string {
  return id.includes(":") ? id.slice(id.indexOf(":") + 1) : id;
}

function useLabel(id: string | null): string {
  const models = useSel((s) => s.models);
  if (!id) return "No model";
  return models.find((m) => m.id === id)?.label ?? bareName(id);
}

export function ModelControl() {
  const model = useSel((s) => s.model);
  const state = useSel((s) => s.modelMenu);
  const settings = useSel((s) => s.modelSettings);
  const controller = useController();
  const label = useLabel(model);

  useEffect(() => {
    void controller.loadModelSettings(model);
  }, [model, controller]);

  const effort = settings?.model === model ? settings.data.effort?.current : null;
  const toggle = (page: "menu" | "settings") =>
    state === page
      ? controller.closeModelMenu()
      : page === "menu"
        ? controller.openModelMenu()
        : controller.openModelSettings();
  return (
    <div className="model-control">
      <button
        type="button"
        className="model-btn"
        aria-haspopup="menu"
        aria-expanded={state === "menu"}
        title={model ?? undefined}
        onClick={() => toggle("menu")}
      >
        {label}
      </button>
      {effort ? (
        <button
          type="button"
          className="model-btn effort-btn"
          aria-haspopup="dialog"
          aria-expanded={state === "settings"}
          aria-label={`Effort: ${effortLabel(effort)}`}
          onClick={() => toggle("settings")}
        >
          {effortLabel(effort)}
        </button>
      ) : null}
      {state === "menu" ? <ModelMenuPopover /> : null}
      {state === "settings" ? <ModelSettingsPage /> : null}
    </div>
  );
}

/** Close the open menu or page on a click anywhere outside the control. */
function useCloseOnOutsideClick() {
  const controller = useController();
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      // A row that swapped the menu for another page is gone by the time the click
      // gets here; it was inside.
      if (!target?.isConnected || target.closest(".model-control")) return;
      controller.closeModelMenu();
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [controller]);
}

/** Escape closes, and gives focus back to the model button. */
function closeOnEscape(e: React.KeyboardEvent, from: HTMLElement | null, close: () => void): boolean {
  if (e.key !== "Escape") return false;
  e.preventDefault();
  const button = from?.parentElement?.querySelector<HTMLElement>(".model-btn");
  close();
  button?.focus();
  return true;
}

function ModelMenuPopover() {
  const controller = useController();
  const current = useSel((s) => s.model);
  const [menu, setMenu] = useState<ModelMenu | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    void controller.modelMenu().then((m) => live && setMenu(m));
    return () => {
      live = false;
    };
  }, [controller]);

  // Focus the current model once the list is in, so the arrow keys start there.
  useEffect(() => {
    if (!menu) return;
    const items = ref.current?.querySelectorAll<HTMLElement>("[role^=menuitem]");
    const checked = ref.current?.querySelector<HTMLElement>("[aria-checked=true]");
    (checked ?? items?.[0])?.focus();
  }, [menu]);

  useCloseOnOutsideClick();

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (closeOnEscape(e, ref.current, () => controller.closeModelMenu())) return;
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = Array.from(ref.current?.querySelectorAll<HTMLElement>("[role^=menuitem]") ?? []);
    const at = items.indexOf(document.activeElement as HTMLElement);
    const next = e.key === "ArrowDown" ? at + 1 : at - 1;
    items[(next + items.length) % items.length]?.focus();
  };

  return (
    <div className="model-menu" role="menu" aria-label="Model" ref={ref} onKeyDown={onKeyDown}>
      {menu?.sections.map((section) => (
        <div key={section.title} role="group" aria-label={section.title}>
          <div className="model-menu-title">{section.title}</div>
          {section.models.map((m) => (
            <button
              key={m.id}
              type="button"
              role="menuitemradio"
              aria-checked={m.id === current}
              className="model-row"
              title={m.id}
              onClick={() => controller.setModel(m.id)}
            >
              <span className="model-check">{m.id === current ? <CheckIcon /> : null}</span>
              <span className="model-name">{m.label}</span>
              {m.note ? <span className="model-detail">{m.note}</span> : null}
            </button>
          ))}
        </div>
      ))}
      {menu ? (
        <div className="model-menu-foot">
          <button
            type="button"
            role="menuitem"
            className="model-row"
            onClick={() => controller.openModelSettings()}
          >
            <span className="model-check" />
            <span className="model-name">Model settings…</span>
          </button>
          <button
            type="button"
            role="menuitem"
            className="model-row"
            onClick={() => controller.openMoreModels()}
          >
            <span className="model-check" />
            <span className="model-name">More models…</span>
          </button>
          <button type="button" role="menuitem" className="model-row" onClick={() => controller.openKeys()}>
            <span className="model-check" />
            <span className="model-name">API keys…</span>
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** The current model's settings, in the menu's place: effort, a local model's
 *  context size with the memory each takes, and the output limit. Only what the
 *  model supports is shown, and a change applies to the running session. */
function ModelSettingsPage() {
  const controller = useController();
  const model = useSel((s) => s.model);
  const current = useSel((s) => s.modelSettings);
  const busy = useSel((s) => s.busy);
  const label = useLabel(model);
  const ref = useRef<HTMLDivElement>(null);
  const [measuring, setMeasuring] = useState(false);
  const measured = useRef<string | null>(null);
  const data: ModelSettings | null = current?.model === model ? current.data : null;

  useCloseOnOutsideClick();

  // A local model's memory per context size is measured on first look (it loads
  // the model a few times), once per model.
  const unmeasured = !!data?.context && !data.context.measured;
  useEffect(() => {
    if (!unmeasured || busy || !model || measured.current === model) return;
    measured.current = model;
    setMeasuring(true);
    void controller.measureModel().finally(() => setMeasuring(false));
  }, [unmeasured, busy, model, controller]);

  const save = (patch: Partial<ModelSettings["settings"]>) => void controller.saveModelSettings(patch);
  const title = busy ? BUSY_TITLE : undefined;

  return (
    <div
      className="model-menu model-settings"
      role="dialog"
      aria-label={`${label} settings`}
      ref={ref}
      onKeyDown={(e) => closeOnEscape(e, ref.current, () => controller.closeModelMenu())}
    >
      <button type="button" className="model-back" onClick={() => controller.openModelMenu()}>
        <ChevronLeft />
        {label}
      </button>
      {data?.effort ? (
        <>
          <div className="model-menu-title">Effort</div>
          <div className="effort-levels" role="radiogroup" aria-label="Effort">
            {data.effort.levels.map((level) => (
              <button
                key={level}
                type="button"
                role="radio"
                aria-checked={level === data.effort?.current}
                disabled={busy}
                title={title}
                onClick={() => save({ effort: level === data.effort?.default ? null : level })}
              >
                {effortLabel(level)}
              </button>
            ))}
          </div>
        </>
      ) : null}
      {data?.context ? (
        <div role="radiogroup" aria-label="Context size">
          <div className="model-menu-title">Context size</div>
          {[{ tokens: null, memory: null, on_gpu: null }, ...data.context.choices].map((choice) => {
            const chosen = (data.settings.context ?? null) === choice.tokens;
            const name =
              choice.tokens === null
                ? `Auto${data.context?.auto ? ` · ${formatTokens(data.context.auto)}` : ""}`
                : formatTokens(choice.tokens);
            const detail = [
              choice.memory ? formatMemory(choice.memory) : "",
              choice.on_gpu === false ? "partly on CPU" : "",
            ].filter(Boolean);
            return (
              <button
                key={choice.tokens ?? "auto"}
                type="button"
                role="radio"
                aria-checked={chosen}
                className="model-row"
                disabled={busy}
                title={title}
                onClick={() => save({ context: choice.tokens })}
              >
                <span className="model-check">{chosen ? <CheckIcon /> : null}</span>
                <span className="model-name">{name}</span>
                {detail.length ? <span className="model-detail">{detail.join(" · ")}</span> : null}
              </button>
            );
          })}
          {measuring ? <p className="model-note">Measuring memory…</p> : null}
        </div>
      ) : null}
      {data?.output ? (
        <label className="model-setting">
          <span className="model-menu-title">Output limit</span>
          <select
            value={data.settings.output ?? ""}
            disabled={busy}
            title={title}
            onChange={(e) => save({ output: e.target.value ? Number(e.target.value) : null })}
          >
            <option value="">Default · {formatTokens(data.output.default)}</option>
            {data.output.choices.map((tokens) => (
              <option key={tokens} value={tokens}>
                {formatTokens(tokens)}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {data && !data.effort && !data.context && !data.output ? (
        <p className="model-note">This model has no settings to change.</p>
      ) : null}
    </div>
  );
}

/** The whole catalog, searchable, newest first per provider, with pins. */
export function MoreModels() {
  const open = useSel((s) => s.modelMenu === "more");
  const models = useSel((s) => s.models);
  const pinned = useSel((s) => s.pinned);
  const current = useSel((s) => s.model);
  const credentials = useSel((s) => s.credentials);
  const controller = useController();
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (open) setQuery("");
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") controller.closeModelMenu();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, controller]);

  const typed = query.trim();
  const groups = useMemo(() => {
    const needle = typed.toLowerCase();
    const byProvider = new Map<string, ModelInfo[]>();
    for (const m of models) {
      if (needle && !`${m.id} ${m.label}`.toLowerCase().includes(needle)) continue;
      byProvider.set(m.provider, [...(byProvider.get(m.provider) ?? []), m]);
    }
    return [...byProvider];
  }, [models, typed]);

  if (!open) return null;
  const providerLabel = (p: string) =>
    credentials.find((c) => c.provider === p)?.label ?? (p === "ollama" ? "Ollama" : p);
  // Any provider:model the provider accepts can be typed in.
  const freeText = typed.includes(":") && !models.some((m) => m.id === typed) ? typed : null;

  return (
    <div className="modal-backdrop" onClick={() => controller.closeModelMenu()}>
      <div
        className="modal more-models"
        role="dialog"
        aria-modal="true"
        aria-label="Models"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h2>Models</h2>
          <button className="icon-btn" aria-label="Close" onClick={() => controller.closeModelMenu()}>
            <CloseIcon />
          </button>
        </div>
        <input
          className="more-search"
          type="search"
          placeholder="Search, or type provider:model"
          aria-label="Search models"
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && freeText) controller.setModel(freeText);
          }}
        />
        <div className="more-list">
          {freeText ? (
            <div className="more-row">
              <button type="button" className="more-pick" onClick={() => controller.setModel(freeText)}>
                <span className="model-name">Use {freeText}</span>
              </button>
            </div>
          ) : null}
          {groups.map(([provider, rows]) => (
            <div key={provider} role="group" aria-label={providerLabel(provider)}>
              <div className="model-menu-title">{providerLabel(provider)}</div>
              {rows.map((m) => {
                const isPinned = pinned.includes(m.id);
                return (
                  <div key={m.id} className={`more-row${m.id === current ? " current" : ""}`}>
                    <button
                      type="button"
                      className="more-pick"
                      title={m.id}
                      onClick={() => controller.setModel(m.id)}
                    >
                      <span className="model-name">{m.label}</span>
                      {m.note ? <span className="model-detail">{m.note}</span> : null}
                    </button>
                    <button
                      type="button"
                      className={`more-pin${isPinned ? " on" : ""}`}
                      aria-pressed={isPinned}
                      aria-label={`Pin ${m.label}`}
                      title={isPinned ? "Unpin" : "Pin to the model menu"}
                      onClick={() => void controller.pinModel(m.id, !isPinned)}
                    >
                      <StarIcon filled={isPinned} />
                    </button>
                  </div>
                );
              })}
            </div>
          ))}
          {!groups.length && !freeText ? <p className="more-empty">No models match.</p> : null}
        </div>
      </div>
    </div>
  );
}
