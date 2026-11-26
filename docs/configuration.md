# Configuration

Three layers, each optional: workspace config, user config, environment.
For the model, precedence is `--model` flag, then workspace, then user
config, then `$MAKKER_MODEL`, then the built-in default.

## Workspace config

`.makker/config.toml` in the workspace, written by `init` and by the
TUI when you save a choice:

```toml
model = "<provider:model>"         # optional, overrides the user default

[workspace]
simulator = "jutuldarcy"           # active simulator for this workspace

[simulator.jutuldarcy]
source_path = "/path/to/JutulDarcy.jl"  # dev-link, set by init --source-path

[julia]
sysimage = true                    # start sessions from this folder's system image
```

The `[julia] sysimage` key is written by `makker sysimage build` and
`clear` (and by `init --sysimage`), so it is not normally edited by hand. When
it is on, a launch that finds the image missing or out of date refuses to
start rather than falling back to the slow path; see
[the system image](sysimage.md).

This file is safe to commit: API keys never go here, and the approval mode
is never the folder's to set.

Opening a folder runs its Julia environment, the way `julia --project` does:
its packages, and a system image if it has one, run as you. So open a folder
from someone else only if you would run its code.

## User config

`<state home>/config.toml` (default `~/.local/share/makker/config.toml`)
holds user-wide defaults: the model, the models pinned to the top of the
model menu, where compaction runs, and each model's
[settings](models.md#settings):

```toml
model = "<provider:model>"
pinned_models = ["<provider:model>"]
compact_at = 85                    # percent of the window, 50 to 95

[model_settings."<provider:model>"]
effort = "high"
output = 32768
context = 131072
```

Set the model from the TUI model selector with `Ctrl+A` (save for all
workspaces), the settings from the model's settings in the web UI or
`/effort` in the TUI, and `compact_at` at the foot of the web UI's context view
(see [context](context.md#growth-and-its-limits)).

## API keys

Keys are read from the process environment, a `.env` in the workspace
folder, and the user-global `<state home>/.env`, in that order of
precedence. The global file is what the interactive key prompts write to,
with owner-only permissions. Recognized:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`.
Only these are read from the workspace `.env`; its other variables are left
alone.
Ollama needs none. The agent's shell commands and Julia session run without
these variables, or any other named like a credential (ending in `_API_KEY`,
`_ADMIN_KEY`, `_TOKEN` or `_SECRET`, such as `GITHUB_TOKEN`), so a command
that prints its environment cannot copy one into the session's trace or
transcript.

You never have to find or edit that file by hand. Set or change a key in
whichever interface you use:

- **Command line:** `makker key` lists which provider keys are set (and
  where each comes from); `makker key <provider>` (e.g. `makker key
  openai`) prompts for a key and saves it. This is the way to replace a wrong
  or expired key, and it needs no checkout, workspace, or Julia.
- **Terminal UI:** `/key` shows the same status; `/key <provider>` opens a
  prompt to set or replace one.
- **Web UI:** if the selected model has no key, a dialog asks for it before the
  session starts; **API keys…** in the model menu opens the same dialog any time.

Because the process environment and a project `.env` take precedence over the
saved file, a key set there shadows the saved one. The status views flag this
so a stale shell variable can't silently override a key you just changed.

## Environment variables

| Variable | Meaning |
|---|---|
| `MAKKER_MODEL` | Default model when no flag or config sets one (useful in CI) |
| `OPENROUTER_API_KEY` | API key for `openrouter:<vendor>/<model>` models; also saved by `makker key openrouter` |
| `MAKKER_JULIA_THREADS` | Compute threads for the Julia kernel: an integer, or `auto` for all logical cores. Defaults to physical cores minus one; the `--threads` flag overrides it. Jutul's assembly and preconditioner are threaded, so this speeds up larger solves. The kernel adds one interactive thread on top, and pins OpenBLAS to one thread (unless `OPENBLAS_NUM_THREADS` is set) to avoid oversubscription. The eval/benchmark harness uses the same policy (production parity); pin it per run with this variable. |
| `MAKKER_HYPRE_THREADS` | OpenMP threads for HYPRE's BoomerAMG (JutulDarcy's CPR pressure preconditioner). Defaults to physical cores minus one, capped at 8 (HYPRE's solver performance degrades with many threads). Independent of the Julia compute-thread count. |
| `MAKKER_NO_XVFB` | Opt out of starting a virtual display on headless Linux (plotting then errors at use) |
| `MAKKER_SYSIMAGE` | Start from the folder's [system image](sysimage.md) when the workspace config says nothing (`1`/`true` to enable). The `--sysimage` / `--no-sysimage` flags override it |
| `MAKKER_NO_OPEN` | Never open the web UI in a browser or artifacts in the OS default application (CI, tests) |
| `MAKKER_NO_UPDATE_CHECK` | Skip the once-a-day release check at launch |
| `MAKKER_REVIEW` | Developer tool: set to `1` to auto-review every finished turn (see [session review](review.md)) |
| `MAKKER_REVIEW_MODEL` | Model the reviewer uses (default `openai:gpt-5.4`) |
| `XDG_DATA_HOME` | Relocates the state home (`<value>/makker`) |
| `MAKKER_SERVER_TOKEN` | The token the web server asks for, in place of the one in the state home (for a launcher that builds the link itself) |

## What lives where

In the workspace:

```
<workspace>/
  .makker/
    config.toml          # workspace config (committable)
    julia-env/           # the simulator's Julia environment
    sysimage/            # the system image, if this folder has one (several GB)
  makker-output/
    sessions/<date>-<id>/
      artifacts/         # plots and reports the agent saved
```

In the state home (default `~/.local/share/makker/`, created owner-only):

```
config.toml              # user config
.env                     # saved API keys (owner-only)
server-token             # the web server's token (owner-only)
ollama-context.json      # context sizes measured for local Ollama models
workspaces/<hash>/
  memory/                # the workspace's agent-maintained memory
  sessions/<id>/
    trace.sqlite         # the session event log
    checkpoints.sqlite   # conversation state (resume, model switching)
    julia-startup.log    # kernel boot log, referenced by error messages
eval-logs/               # Inspect logs from makker eval
eval-envs/<sim>/         # golden envs the bench copies per sample
```

The state home is keyed by a hash of the workspace path, so the same folder
always maps to the same memory and session history.
