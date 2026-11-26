# Use Makker from another agent

`makker mcp` lets an agent you already use work in a Makker session, through
the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). That agent
brings its own model and sign-in, so Makker needs no API key. Makker brings the
persistent Julia session for your simulator, its guides, the figures, and the
record of what was run.

## Set it up

You need Makker installed (`uv tool install makker`), Julia, and a folder for
the work. Set the folder's simulator once:

```sh
makker init --sim jutuldarcy
```

Then add Makker to your agent as a local (stdio) MCP server with the command
`makker mcp`. Each agent has its own way to do that; for example:

```sh
claude mcp add --scope user makker -- makker mcp    # Claude Code
codex mcp add makker -- makker mcp                  # Codex
copilot mcp add makker -- makker mcp                # GitHub Copilot CLI
```

Some agents also read a `.mcp.json` in the folder, which you can commit so
everyone working there gets the server:

```json
{
  "mcpServers": {
    "makker": { "type": "stdio", "command": "makker", "args": ["mcp"] }
  }
}
```

A session started after you add the server has its tools: ask the agent to use
Makker. A settings dialog that asks for a URL is for servers on the internet and
cannot reach Makker.

If the agent cannot find `makker`, give the command's full path (`uv tool dir
--bin` prints its folder). An agent with no folder open needs `--workspace
<folder>` in the arguments. An organisation can switch local MCP servers off or
allow them by exact command: the command to allow is `makker mcp`.

## What the agent gets

| Tool | What it does |
|---|---|
| `run_julia` | Runs code in the session's Julia. Variables and loaded packages stay between calls. |
| `plot_julia` | Draws a figure, saves it as a PNG, and links to the live figure. |
| `recapture_plot`, `close_plots` | Looks at a live figure as you left it; closes figures. |
| `reset_julia` | Starts Julia afresh. |
| `read_guide` | Reads Makker's guides for the folder's simulator, the same ones Makker's own agent uses. |
| `julia_wait`, `julia_interrupt` | Collects or stops a call that is still running (see below). |
| `record_attempt`, `write_report` | Keeps a log of approaches tried and writes an HTML report. |

The agent keeps its own tools for files, the shell and the web. Makker's own
memory, approval prompts and model settings do not apply: those are the other
agent's.

Julia starts when the agent first uses Makker, not when it connects, so the
server costs nothing in a session that never uses it. In a folder used for the
first time Makker installs the Julia environment first, which takes minutes.

## Long calls

Agents stop waiting for a tool after a while, some after a minute. A call that
needs more than 45 seconds answers "still running" with what the code has
printed so far, and keeps going. The agent calls `julia_wait` to collect the
result, or `julia_interrupt` to stop the code. `--max-wait <seconds>` changes
the 45 seconds.

If the agent gives up on a call, or you press stop in it, the Julia code keeps
running: a time limit and a stop look the same to the server, and a simulation
should not be lost to a time limit. Ask the agent to interrupt Julia when you
want the code stopped.

## Figures

Each plot is saved as a PNG in the session's output folder, and the result links
to the live figure: the same interactive view as in Makker's web UI, which you
can rotate, zoom and step. Open the link in a browser, or in your agent's
browser pane if it has one. When the agent draws a figure again under the same
name, the open page loads the new one by itself.

The figures are served on this computer only (127.0.0.1), on a port opened when
the first figure is drawn. Each link carries a random key, and nothing is served
without it. A live figure stops responding while Julia is busy, as in the web UI.

## What is recorded

An MCP session is a Makker session. What the other agent ran, what it got back,
and the figures and reports are in the trace, so `makker sessions` and
`makker transcript` work as usual. The other agent's own messages and reasoning
are not recorded: Makker only sees the calls made to it.

## Options

```sh
makker mcp [--sim <name>] [--max-wait <seconds>] [--workspace <dir>]
           [--julia-project <dir>] [--threads <n>] [--sysimage | --no-sysimage]
```

`--sim` names the simulator; by default it is the folder's saved choice, else
what its packages say. The rest mean what they do for the other commands; see
the [CLI reference](cli.md).
