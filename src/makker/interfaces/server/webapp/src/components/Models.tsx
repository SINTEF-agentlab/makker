// The model control under the message box: the current model, a short menu of the
// models most likely wanted (built by the server, the same list the terminal shows),
// and the full catalog behind "More models…".

import { useEffect, useMemo, useRef, useState } from "react";

import type { ModelInfo, ModelMenu } from "../api";
import { useController, useSel } from "../context";
import { formatTokens } from "../format";
import { CheckIcon, ChevronDown, CloseIcon, StarIcon } from "../icons";

/** "gpt-6-luna" for "openai:gpt-6-luna": the name without its provider. */
function bareName(id: string): string {
  return id.includes(":") ? id.slice(id.indexOf(":") + 1) : id;
}

function useLabel(id: string | null): string {
  const models = useSel((s) => s.models);
  if (!id) return "No model";
  return models.find((m) => m.id === id)?.label ?? bareName(id);
}

/** The right-hand detail of a row: the window, and whether it runs here. */
function details(model: ModelInfo): string {
  const bits = [model.window ? formatTokens(model.window) : "", model.local ? "local" : "", model.note ?? ""];
  return bits.filter(Boolean).join(" · ");
}

export function ModelControl() {
  const model = useSel((s) => s.model);
  const state = useSel((s) => s.modelMenu);
  const controller = useController();
  const label = useLabel(model);
  const open = state === "menu";
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
        <ChevronDown />
      </button>
      {open ? <ModelMenuPopover /> : null}
    </div>
  );
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

  // Close on a click anywhere outside the control.
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (!(e.target as HTMLElement)?.closest(".model-control")) controller.closeModelMenu();
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [controller]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      const button = ref.current?.parentElement?.querySelector<HTMLElement>(".model-btn");
      controller.closeModelMenu();
      button?.focus();
      return;
    }
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
              <span className="model-detail">{details(m)}</span>
            </button>
          ))}
        </div>
      ))}
      {menu?.add_key.length ? (
        <div role="group" aria-label="Other providers">
          <div className="model-menu-title">Other providers</div>
          {menu.add_key.map((p) => (
            <button
              key={p.provider}
              type="button"
              role="menuitem"
              className="model-row"
              onClick={() => controller.addKey(p.provider)}
            >
              <span className="model-check" />
              <span className="model-name">{p.label}</span>
              <span className="model-detail">Add key</span>
            </button>
          ))}
        </div>
      ) : null}
      {menu ? (
        <button
          type="button"
          role="menuitem"
          className="model-row model-more"
          onClick={() => controller.openMoreModels()}
        >
          <span className="model-check" />
          <span className="model-name">More models…</span>
        </button>
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
                      <span className="model-detail">{details(m)}</span>
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
