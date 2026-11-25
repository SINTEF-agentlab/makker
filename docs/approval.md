# Approval and safety

The agent can edit files and run shell commands. Approval is the
human-in-the-loop checkpoint in front of those actions. This page explains
how it works and, just as importantly, what it does and does not protect
against.

## What is gated

Four tools require approval (`agent/approval.py`):

| Tool | Action |
|---|---|
| `execute` | run a shell command |
| `write_file` | write a file |
| `edit_file` | edit a file |
| `delete` | delete a file or directory (directories recursively) |

The gate is implemented as a langgraph interrupt: when the agent calls a
gated tool, the session's approval policy decides whether this call needs a
person. If it does, the graph pauses, the pending call (tool name and
arguments) surfaces as an approval request, and the turn resumes with the
decision. Calls the policy allows run without a pause, so an approval card
only ever holds what needs a decision. In the TUI this is the approval card.
Typing a reply instead of choosing declines the request and gives the agent
your text as the reason. When several requests are pending at once, the card
shows them all and one decision answers every one. Every request and decision
is recorded in the trace (`hitl_request`, `hitl_response`).

A card that holds only edits to files inside the workspace offers "always
allow file edits" for the rest of the session. Nothing else can be always
allowed: each shell command and each delete gets its own prompt, and so does
an edit to a file outside the workspace.

## Modes

| Mode | Behavior |
|---|---|
| `ask` (default) | prompt for all four tools |
| `workspace` ("accept edits") | allow `write_file`/`edit_file` inside the workspace; prompt for the rest |
| `auto` | allow all four |

Set it at launch (`--approval-mode`), with `/approval-mode` in the TUI and
web UI, or cycle with `Shift+Tab` in the TUI. A folder never sets it, so a
folder from someone else can't turn approval off.
A change applies from the agent's next tool call, mid-turn included, and
`/approval-mode` resumes a pending approval the new mode allows at once.
`Shift+Tab` leaves the card on screen for you to answer, since cycling from
"accept edits" back to "default" passes through `auto`. Headless runs cannot
prompt, so a gated call under `ask` ends the turn with an error; scripts and
the bench run with `auto`.

## What approval is not

Approval is a review checkpoint for visible side effects, not a security
boundary. Two honest limits:

- `run_julia` and `plot_julia` are not gated. Running arbitrary Julia is
  the agent's core job (a simulation can read and write files and use the
  network like any program), so gating it would mean approving every
  working step. The design trusts Julia evaluation the way it trusts the
  user's own REPL.
- Approval inspects intent, not effect. You approve the command the agent
  *says* it will run. A reviewer skimming `execute` arguments is the
  protection, and that protection is only as good as the attention given.

The consequence: approval is the right tool for supervising an interactive
session you are watching. For unattended runs, the protection that matters
is *where the process runs*, not which tools are gated.

## Isolation for unattended runs

`--approval-mode auto` hands the agent its full toolset without review, so
match the blast radius to the trust level:

- Bench and CI runs of the bundled task suites are our own prompts against
  major-provider models, where the practical risk is accidental damage, not
  malice. CI runners are ephemeral VMs, which is already strong isolation,
  so scheduled bench runs belong there.
- On a workstation, prefer a dedicated workspace directory and
  `--ephemeral-memory`, and remember the shell is not confined to the
  workspace.
- For adversarial evaluation tiers, third-party task packs, or any prompt
  you did not write, run the whole invocation in a container or VM. That
  is the boundary that actually contains a destructive command.

See [evaluation](evaluation.md) for how the bench applies this.
