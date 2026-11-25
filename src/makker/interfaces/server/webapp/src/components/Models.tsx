// The model control under the message box: the current model and its effort, and a
// menu with the models most likely wanted (built by the server, the same list the
// terminal shows), the current model's effort and context size as rows that open a
// list beside the menu, the full catalog behind "More models…", and the API keys.
// The output length is set in the config file only.

import { useEffect, useMemo, useRef, useState } from "react";

import type { ModelInfo, ModelMenu, ModelSettings } from "../api";
import { useController, useSel } from "../context";
import { formatTokens } from "../format";
import { CheckIcon, ChevronRight, CloseIcon, StarIcon } from "../icons";

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
  const open = useSel((s) => s.modelMenu === "menu");
  const settings = useSel((s) => s.modelSettings);
  const controller = useController();
  const label = useLabel(model);

  useEffect(() => {
    void controller.loadModelSettings(model);
  }, [model, controller]);

  const effort = settings?.model === model ? settings.data.effort?.current : null;
  return (
    <div className="model-control">
      <button
        type="button"
        className="model-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        title={model ?? undefined}
        onClick={() => (open ? controller.closeModelMenu() : controller.openModelMenu())}
      >
        {label}
        {effort ? " " : null}
        {effort ? <span className="model-effort">{effortLabel(effort)}</span> : null}
      </button>
      {open ? <ModelMenuPopover /> : null}
    </div>
  );
}

type Setting = "effort" | "context";

/** The items a key press moves between: those of the open submenu when focus is
 *  in it, else the menu's own (leaving any submenu's out). */
function menuItems(menu: HTMLElement | null): HTMLElement[] {
  const sub = (document.activeElement as HTMLElement | null)?.closest<HTMLElement>(".model-sub");
  const scope = sub ?? menu;
  const items = Array.from(scope?.querySelectorAll<HTMLElement>("[role^=menuitem]") ?? []);
  return sub ? items : items.filter((el) => !el.closest(".model-sub"));
}

function ModelMenuPopover() {
  const controller = useController();
  const current = useSel((s) => s.model);
  const stored = useSel((s) => s.modelSettings);
  const [menu, setMenu] = useState<ModelMenu | null>(null);
  const [open, setOpen] = useState<Setting | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const data = stored?.model === current ? stored.data : null;

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
    const checked = ref.current?.querySelector<HTMLElement>(".model-menu-list [aria-checked=true]");
    (checked ?? menuItems(ref.current)[0])?.focus();
  }, [menu]);

  // Close on a click anywhere outside the control. A row the click replaced (such
  // as "Custom…" turning into a field) is gone by the time it gets here; it was inside.
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target?.isConnected || target.closest(".model-control")) return;
      controller.closeModelMenu();
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [controller]);

  const openSetting = (setting: Setting, focus: boolean) => {
    setOpen(setting);
    if (!focus) return;
    requestAnimationFrame(() => {
      const sub = ref.current?.querySelector<HTMLElement>(".model-sub");
      (sub?.querySelector<HTMLElement>("[aria-checked=true]") ?? menuItems(sub ?? null)[0])?.focus();
    });
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    // Keys typed into the custom size are for the field, except Escape.
    if ((e.target as HTMLElement).tagName === "INPUT" && e.key !== "Escape") return;
    const active = document.activeElement as HTMLElement | null;
    const inSub = !!active?.closest(".model-sub");
    if (e.key === "Escape" || (e.key === "ArrowLeft" && inSub)) {
      e.preventDefault();
      if (inSub) {
        // Back to the row the submenu belongs to.
        ref.current?.querySelector<HTMLElement>(`[data-setting=${open}]`)?.focus();
        setOpen(null);
        return;
      }
      const button = ref.current?.parentElement?.querySelector<HTMLElement>(".model-btn");
      controller.closeModelMenu();
      button?.focus();
      return;
    }
    const setting = active?.dataset.setting as Setting | undefined;
    if ((e.key === "ArrowRight" || e.key === "Enter") && setting && !inSub) {
      e.preventDefault();
      openSetting(setting, true);
      return;
    }
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = menuItems(ref.current);
    const at = items.indexOf(active as HTMLElement);
    const next = e.key === "ArrowDown" ? at + 1 : at - 1;
    items[(next + items.length) % items.length]?.focus();
  };

  // After a choice only the list closes: the menu stays, its row showing the new
  // value, with focus back on that row.
  const closeSetting = (setting: Setting) => {
    setOpen(null);
    ref.current?.querySelector<HTMLElement>(`[data-setting=${setting}]`)?.focus();
  };

  // A setting's list opens on a click (or Enter, or the right arrow) and stays open
  // until another is opened or the menu closes, so the pointer can cross other rows
  // on its way to it.
  const row = (setting: Setting, label: string, value: string) => (
    <div className="model-setting-row">
      <button
        type="button"
        role="menuitem"
        aria-haspopup="menu"
        aria-expanded={open === setting}
        data-setting={setting}
        className="model-row"
        onClick={() => (open === setting ? setOpen(null) : openSetting(setting, false))}
      >
        <span className="model-check" />
        <span className="model-name">{label}</span>
        <span className="model-detail">{value}</span>
        <ChevronRight />
      </button>
      {open === setting && data ? (
        <SettingMenu setting={setting} data={data} onChosen={() => closeSetting(setting)} />
      ) : null}
    </div>
  );

  const contextValue = (context: NonNullable<ModelSettings["context"]>, chosen: number | null) =>
    chosen ? formatTokens(chosen) : context.local ? "Auto" : "Full";

  return (
    <div className="model-menu" role="menu" aria-label="Model" ref={ref} onKeyDown={onKeyDown}>
      <div className="model-menu-list">
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
      </div>
      {menu ? (
        <div className="model-menu-foot">
          {data?.effort?.current ? row("effort", "Effort", effortLabel(data.effort.current)) : null}
          {data?.context ? row("context", "Context size", contextValue(data.context, data.settings.context)) : null}
          <div className="model-menu-tail">
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
        </div>
      ) : null}
    </div>
  );
}

/** "100k" -> 100000, "1.5M" -> 1500000, "65536" -> 65536; NaN if not a size. */
function parseTokens(text: string): number {
  const match = /^\s*([\d.]+)\s*([kKmM]?)\s*$/.exec(text);
  if (!match) return NaN;
  const scale = { k: 1e3, m: 1e6, "": 1 }[match[2].toLowerCase() as "k" | "m" | ""];
  return Math.round(Number(match[1]) * scale);
}

/** The list beside a setting's row: the choices, the current one checked. A choice
 *  is saved for the model and applies to the running session from its next turn. */
function SettingMenu({
  setting,
  data,
  onChosen,
}: {
  setting: Setting;
  data: ModelSettings;
  onChosen: () => void;
}) {
  const controller = useController();
  const busy = useSel((s) => s.busy);
  const [measuring, setMeasuring] = useState(false);
  const [typing, setTyping] = useState(false);
  const [typed, setTyped] = useState("");
  const [problem, setProblem] = useState<string | null>(null);

  // A local model's memory per size is measured the first time its sizes are
  // shown: it loads the model a few times, once per model version.
  const unmeasured = setting === "context" && !!data.context?.local && !data.context.measured;
  useEffect(() => {
    if (!unmeasured || busy) return;
    setMeasuring(true);
    void controller.measureModel().finally(() => setMeasuring(false));
  }, [unmeasured, busy, controller]);

  const choose = (patch: Partial<ModelSettings["settings"]>) => {
    onChosen();
    void controller.saveModelSettings(patch);
  };

  const item = (
    key: string,
    name: string,
    checked: boolean,
    onClick: () => void,
    detail = "",
    unavailable = false,
  ) => (
    <button
      key={key}
      type="button"
      role="menuitemradio"
      aria-checked={checked}
      className="model-row"
      disabled={busy || unavailable}
      title={busy ? BUSY_TITLE : undefined}
      onClick={onClick}
    >
      <span className="model-check">{checked ? <CheckIcon /> : null}</span>
      <span className="model-name">{name}</span>
      {detail ? <span className="model-detail">{detail}</span> : null}
    </button>
  );

  if (setting === "effort" && data.effort) {
    const { levels, default: fallback, current } = data.effort;
    return (
      <div className="model-menu model-sub" role="menu" aria-label="Effort">
        <p className="model-note">More effort is more thorough, and slower.</p>
        {levels.map((level) =>
          item(level, effortLabel(level), level === current, () =>
            choose({ effort: level === fallback ? null : level }),
          ),
        )}
      </div>
    );
  }

  const context = data.context;
  if (setting !== "context" || !context) return null;
  const chosen = data.settings.context;
  const custom = chosen && !context.choices.some((c) => c.tokens === chosen) ? chosen : null;
  const submit = () => {
    const tokens = parseTokens(typed);
    if (!(tokens >= context.min && tokens <= context.max)) {
      setProblem(`Between ${formatTokens(context.min)} and ${formatTokens(context.max)}`);
      return;
    }
    choose({ context: tokens });
  };
  return (
    <div className="model-menu model-sub" role="menu" aria-label="Context size">
      {item(
        "auto",
        context.local ? "Auto" : "Full",
        !chosen,
        () => choose({ context: null }),
        context.auto ? formatTokens(context.auto) : "",
      )}
      {context.choices
        .filter((c) => c.tokens !== context.auto || context.local)
        .map((c) =>
          item(
            String(c.tokens),
            formatTokens(c.tokens),
            chosen === c.tokens,
            () => choose({ context: c.tokens }),
            c.too_large
              ? "too large"
              : [c.memory ? formatMemory(c.memory) : "", c.on_gpu === false ? "partly on CPU" : ""]
                  .filter(Boolean)
                  .join(" · "),
            c.too_large,
          ),
        )}
      {/* Gone as soon as the figures are in, whatever else is still settling. */}
      {measuring && unmeasured ? <p className="model-note">Measuring memory…</p> : null}
      {typing ? (
        <div className="model-custom">
          <input
            autoFocus
            aria-label="Context size in tokens"
            placeholder="e.g. 100k"
            value={typed}
            onChange={(e) => {
              setTyped(e.target.value);
              setProblem(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                e.stopPropagation();
                submit();
              }
            }}
          />
          {problem ? <p className="model-note">{problem}</p> : null}
        </div>
      ) : (
        item("custom", custom ? `Custom · ${formatTokens(custom)}` : "Custom…", !!custom, () => setTyping(true))
      )}
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
