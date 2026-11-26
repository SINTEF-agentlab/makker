// The conversation pane: the welcome screen when empty, otherwise the thread (the
// agent's steps condensed into groups), the plan and status line while a turn runs,
// and any pending approval. Keeps the view pinned to the bottom while streaming
// unless the user has scrolled up to read.

import { memo, useLayoutEffect, useMemo, useRef } from "react";

import { useController, useSel } from "../context";
import { currentPlan, groupSteps, isRunning, type Block, type Step } from "../steps";
import { Approval } from "./Approval";
import { ErrorBoundary } from "./ErrorBoundary";
import {
  ArtifactFileCard,
  ArtifactImageCard,
  AssistantMessage,
  ErrorCard,
  HelpCard,
  Plan,
  Status,
  StepGroup,
  SysNote,
  UiNote,
  UserBubble,
  VizChip,
} from "./messages";

const NO_STEPS: Step[] = [];

const DEFAULT_EXAMPLES = [
  "Set up a small simulation and show me the interactive result.",
  "Plot the results from the run.",
  "Give me a quick tour of what this simulator can do.",
];

const DEFAULT_TAGLINE =
  "Ask a question or describe a task. The agent runs the simulator, writes and runs Julia, and shows results here.";

function Welcome() {
  const sim = useSel((s) => s.sim);
  const details = useSel((s) => s.simDetails);
  const branding = useSel((s) => s.branding);
  const busy = useSel((s) => s.busy);
  const controller = useController();
  const d = sim ? details[sim] : undefined;
  // An installed capability speaks for the whole session, so it names the screen
  // and sets its starter prompts ahead of the simulator underneath it.
  const display = branding?.display_name || d?.display_name || sim;
  const tagline = branding?.tagline || DEFAULT_TAGLINE;
  const prompts =
    (branding?.examples?.length && branding.examples) ||
    (d?.examples?.length && d.examples) ||
    DEFAULT_EXAMPLES;
  return (
    <div className="welcome">
      <h1>{display ? `What would you like to explore with ${display}?` : "What would you like to explore?"}</h1>
      <p>{tagline}</p>
      <div className="examples">
        {prompts.map((t) => (
          <button key={t} className="example" disabled={busy} onClick={() => controller.send(t)}>
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}

type Item = Extract<Block, { kind: "item" }>["item"];

const ThreadItemView = memo(function ThreadItemView({ item }: { item: Item }) {
  switch (item.kind) {
    case "user":
      return <UserBubble text={item.text} />;
    case "assistant":
      return <AssistantMessage text={item.text} />;
    case "viz-chip":
      return <VizChip viewId={item.viewId} title={item.title} viewKind={item.viewKind} />;
    case "artifact-image":
      return <ArtifactImageCard url={item.url} title={item.title} />;
    case "artifact-file":
      return <ArtifactFileCard url={item.url} caption={item.caption} />;
    case "sys-note":
      return <SysNote text={item.text} level={item.level} />;
    case "ui-note":
      return <UiNote action={item.action} payload={item.payload} />;
    case "error":
      return <ErrorCard message={item.message} retry={item.retry} />;
    case "help":
      return <HelpCard />;
  }
});

export function Thread() {
  const items = useSel((s) => s.items);
  const busy = useSel((s) => s.busy);
  const pending = useSel((s) => s.pending);
  const bottomPin = useSel((s) => s.bottomPin);
  const blocks = useMemo(() => groupSteps(items), [items]);
  // While the turn runs, the steps still under way belong to the status line at
  // the bottom; a group shows them once they are done.
  const last = blocks[blocks.length - 1];
  const live = busy && last?.kind === "group" ? last.steps : NO_STEPS;
  const plan = useMemo(() => (busy ? currentPlan(items) : null), [busy, items]);

  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const onScroll = () => {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };
  // After any content change, keep the view pinned to the bottom if it already was.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [items, busy, pending]);
  // Force the view to the bottom when the user sends a message, even if they had
  // scrolled up, so their prompt and the incoming reply are visible.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (!el) return;
    stick.current = true;
    el.scrollTop = el.scrollHeight;
  }, [bottomPin]);

  return (
    <main className="conversation" ref={scroller} onScroll={onScroll}>
      <div className="thread">
        {items.length === 0 ? (
          <Welcome />
        ) : (
          blocks.map((block) => {
            if (block.kind === "item") {
              return (
                <ErrorBoundary key={block.item.id} what="item">
                  <ThreadItemView item={block.item} />
                </ErrorBoundary>
              );
            }
            // The turn's newest group is still being added to: its steps under
            // way are on the status line (or on the approval card, while one
            // waits there), and a thought alone is not a line yet (the steps
            // that follow it would replace it with their summary).
            const open = (busy || pending !== null) && block === last;
            const steps = open ? block.steps.filter((step) => !isRunning(step)) : block.steps;
            return steps.some((step) => !open || step.kind === "tool") ? (
              <ErrorBoundary key={block.id} what="item">
                <StepGroup steps={steps} />
              </ErrorBoundary>
            ) : null;
          })
        )}
        {busy ? (
          <ErrorBoundary what="item">
            {plan ? <Plan todos={plan} /> : null}
            <Status steps={live} />
          </ErrorBoundary>
        ) : null}
        {pending ? <Approval /> : null}
      </div>
    </main>
  );
}
