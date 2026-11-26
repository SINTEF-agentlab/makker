// The composer: the message box plus its keyboard-driven slash-command menu, prompt
// history (↑/↓), and file attach, with the model and context controls below it. Enter
// sends; a return key shows once there is something to send, and a stop button while
// a turn runs.

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { SLASH_COMMANDS, type SlashCommand } from "../controller";
import { useController, useSel } from "../context";
import { ContextMeter } from "./Context";
import { ModelControl } from "./Models";
import { AttachIcon, ReturnIcon, StopIcon } from "../icons";
import { replyDecision } from "../store";

export function Composer() {
  const controller = useController();
  const busy = useSel((s) => s.busy);
  const contextOpen = useSel((s) => s.contextView.open);
  const pending = useSel((s) => s.pending);

  const [value, setValue] = useState("");
  const [slashIndex, setSlashIndex] = useState(0);

  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const slashItems = useMemo<SlashCommand[]>(
    () =>
      value.startsWith("/") && !/\s/.test(value)
        ? SLASH_COMMANDS.filter((c) => c.name.startsWith(value))
        : [],
    [value],
  );
  const slashOpen = slashItems.length > 0;

  // Only a hint that says something: an empty box needs no label.
  const placeholder = pending && replyDecision(pending.allowed)
    ? "Tell the agent what to do instead…"
    : undefined;

  // Auto-grow the textarea up to a cap.
  useLayoutEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [value]);

  // An uploaded file's path (from the attach button or a drag-drop) is appended to
  // the message box through this sink, so the user can reference it in their prompt.
  useEffect(() => {
    controller.setComposerSink((t) => {
      setValue((v) => v + (v && !/\s$/.test(v) ? " " : "") + t + " ");
      textarea.current?.focus();
    });
    return () => controller.setComposerSink(null);
  }, [controller]);

  const focus = () => textarea.current?.focus();

  const submit = () => {
    const text = value.trim();
    if (!text) return;
    if (controller.send(text)) setValue("");
  };

  const completeSlash = (cmd: SlashCommand) => {
    setValue(cmd.name + (cmd.hint ? " " : ""));
    focus();
  };

  const recall = (dir: -1 | 1) => {
    const next = controller.recall(dir, value);
    if (next == null) return false;
    setValue(next);
    requestAnimationFrame(() => {
      const el = textarea.current;
      if (el) el.setSelectionRange(el.value.length, el.value.length);
    });
    return true;
  };

  const atStart = () =>
    textarea.current?.selectionStart === 0 && textarea.current?.selectionEnd === 0;

  const atEnd = () => {
    const el = textarea.current;
    return !!el && el.selectionStart === el.value.length && el.selectionEnd === el.value.length;
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      // With the context view open, Escape closes that and the turn runs on.
      if (busy && !contextOpen) return controller.stop();
      return;
    }
    if (slashOpen) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        return setSlashIndex((i) => (i + 1) % slashItems.length);
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        return setSlashIndex((i) => (i - 1 + slashItems.length) % slashItems.length);
      }
      const highlighted = slashItems[Math.min(slashIndex, slashItems.length - 1)];
      if (e.key === "Tab") {
        e.preventDefault();
        return completeSlash(highlighted);
      }
      if (e.key === "Enter") {
        e.preventDefault();
        // A fully-typed command runs; a partial one completes to the highlighted item.
        if (SLASH_COMMANDS.some((c) => c.name === value.trim())) return submit();
        return completeSlash(highlighted);
      }
    } else if (e.key === "ArrowUp" && atStart()) {
      if (recall(-1)) e.preventDefault();
      return;
    } else if (e.key === "ArrowDown" && atEnd()) {
      // Only recall forward from the end of the draft, mirroring ArrowUp-at-start,
      // so pressing Down to move the cursor within a draft never clobbers it.
      if (recall(1)) e.preventDefault();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const onAttach = (files: FileList | null) => {
    if (!files) return;
    for (const f of Array.from(files)) void controller.upload(f); // sink appends the path
  };

  return (
    <footer className="composer-wrap">
      {slashOpen ? (
        <div className="slash-menu">
          {slashItems.map((c, i) => (
            <div
              key={c.name}
              className={`slash-item${i === Math.min(slashIndex, slashItems.length - 1) ? " active" : ""}`}
              onClick={() => completeSlash(c)}
            >
              <span className="slash-name">{c.name + (c.hint ? " " + c.hint : "")}</span>
              <span className="slash-desc">{c.desc}</span>
            </div>
          ))}
        </div>
      ) : null}
      <div className="composer">
        <button
          className="attach"
          title="Attach a file"
          aria-label="Attach a file"
          onClick={() => fileInput.current?.click()}
        >
          <AttachIcon />
        </button>
        <input
          ref={fileInput}
          type="file"
          hidden
          multiple
          onChange={(e) => {
            onAttach(e.target.files);
            e.target.value = "";
          }}
        />
        <textarea
          ref={textarea}
          rows={1}
          aria-label="Message"
          placeholder={placeholder}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            setSlashIndex(0);
          }}
          onKeyDown={onKeyDown}
        />
        {busy ? (
          <button className="send" aria-label="Stop" onClick={() => controller.stop()}>
            <StopIcon />
          </button>
        ) : value.trim() ? (
          <button className="send" aria-label="Send" onClick={submit}>
            <ReturnIcon />
          </button>
        ) : null}
      </div>
      <div className="composer-foot">
        <ModelControl />
        <ContextMeter />
      </div>
    </footer>
  );
}
