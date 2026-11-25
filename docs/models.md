# Models

Makker treats models as opaque `provider:model` strings resolved by
LangChain's `init_chat_model`. Adding a new model is configuration, not a
code change.

## Selection

Precedence, highest first: the `--model` flag, the workspace config, the
user config, `$MAKKER_MODEL`, the built-in default. A resumed session runs
the model it last ran.

In the web UI the current model is shown under the message box, beside the
context ring. Its menu lists your pinned and recent models, the three newest
models of each provider you have a key for, and your installed local models.
**More models…** searches everything, pins a model to the top of the menu,
and takes any `provider:model` typed in; **API keys…** sets a provider's key. `/model` opens the same menu, and
`/model provider:model` switches directly. In the terminal, `/model` opens the
same short list, and typing searches everything.

A switch keeps the conversation (see [context handling](context.md)) and is
saved for the workspace, so the next session there starts on it.

## Settings

Each model can be set to run differently, with only the settings it supports:

- **Effort:** how hard a reasoning model thinks, from the levels the model
  offers (the provider data lists them for cloud models; Ollama reports them for
  local ones). Unset, a model runs at its own default, and OpenAI models at
  medium.
- **Context size:** for a local model, the context it is loaded with; each size
  is listed with the memory it takes on this machine and whether it still fits
  on the GPU. Unset, it is sized automatically (see
  [local models](#local-models-ollama)). For a hosted model, a smaller size caps
  the window Makker compacts against (nothing is sent to the provider), for a
  cheaper session or to see compaction sooner. Any size from 16K up to the
  model's maximum can be typed in.
- **Output length:** the most the model may write in one reply, its reasoning
  included, up to its maximum.

Settings are saved per model in the user config and apply in every workspace.
A change applies to the running session from its next turn. In the web UI the
effort shows after the model's name, and the model menu ends with a row for
each setting the current model supports; each opens its choices beside the
menu. In the terminal, `/effort` lists the levels and `/effort <level>` sets
one.

Providers beyond the bundled five work once their LangChain package is
installed, and Makker names the exact package when it is missing.
API keys are prompted for on first use and stored in the user-global `.env`
([configuration](configuration.md)).

## OpenRouter

OpenRouter is included. Save an API key and start the web UI:

```sh
makker key openrouter
makker
```

Then use `/model` to choose an OpenRouter model.

## What the harness asks of a model

The agent is tool-driven: every Julia evaluation, file edit, and plot is a
tool call, often ten or more per task. Models therefore need solid tool
calling, and weaker models fail in characteristic ways: skipping tools and
answering from memory, or calling a tool with malformed arguments. The
bench's trajectory scorers exist to catch exactly this, and the canary
suite is the quick way to qualify a new model
([evaluation](evaluation.md)).

## Reasoning

Models that can reason get it requested explicitly, because the provider
defaults hide it or skip it: recent OpenAI reasoning models default to
effort *none* (they do not reason at all), and Anthropic's extended
thinking is off without a budget. The builder asks the model's bundled
profile (`BaseChatModel.profile`) whether it reasons and then enables, per
provider:

- OpenAI: `reasoning={"effort": "medium", "summary": "auto"}`, or the effort
  chosen in the [settings](#settings). The model
  reasons when a task warrants it and the summary streams into the TUI's
  reasoning card while it thinks. The `-chat` hybrids, which reject the
  effort parameter, are excluded by the same profile data (they keep
  temperature support; true reasoning models do not).
- Anthropic: extended thinking with a 10k-token budget (output capped at
  24k so thinking leaves room for the answer). The thinking text streams
  into the same card.
- Gemini: `include_thoughts=True`. These models think by default (level
  `high` on Gemini 3+), so the only thing to add is visibility. A Gemini
  model with no profile entry is treated as thinking, since every model
  newer than the bundled data thinks; the legacy non-thinking ones are
  marked explicitly in the data.
- OpenRouter: models whose bundled profile supports reasoning receive
  `reasoning={"effort": "medium", "summary": "auto"}` through the dedicated
  integration, which preserves reasoning across tool calls.
- Ollama: think mode is requested explicitly for models the daemon
  reports as thinking-capable. Left at the daemon default, the thinking
  segment is dropped on the client side, and a turn the model spends
  entirely on thinking then surfaces as an empty reply with no tool
  calls, and the agent falls silent. Requested, the thinking is
  separated and visible.

Because the capability check is profile data shipped with the provider
packages, new models are covered by upgrading the package, and there is no
model list to maintain here. A model whose profile is missing (or a
session without the provider key at build time) falls back to the plain
spec string and simply runs without visible reasoning. Reasoning cards
appear only when the model actually reasons: trivial prompts legitimately
produce none.

The same profile data supplies the context window for `/context` and the
auto-compaction trigger, read without building the model; Gemini models
missing from the data ask the Gemini API for their limit instead. A model no
source knows is assumed to have 200K tokens (see
[Context](context.md)).

## Local models (Ollama)

Local models run through Ollama with no key. What the harness handles for
you:

- Capability gating: the selector checks that a model supports tool
  calling before offering it, and can pull a model that is not installed
  yet. An outdated Ollama can mis-parse a new model's template and lose
  tool support, so keep Ollama itself current.
- Context sizing: a local model is loaded with the largest context, up to
  its maximum, that Ollama keeps entirely on the GPU. Memory per token
  differs widely between models, so this is measured, not estimated: the
  first time a model is used on a machine, Makker loads it at its maximum
  and at 32K (which gives the memory each size takes), and, if the maximum
  spills to the CPU, a few more times to narrow it down (seconds per load). The answer is remembered per model version and
  machine in `ollama-context.json` in the state home; delete it to measure
  again. A model that spills even at 32K tokens is loaded at 32K, partly on
  the CPU, since the agent's own prompt needs that much. A model whose
  Modelfile sets `num_ctx` (such as a `-64k` variant) is loaded at that
  size instead. To choose a size, use the context size in the model's
  [settings](#settings), which shows what each size costs in memory.
  Hosted `:cloud` models run at their full window.
- Malformed tool calls: a weaker model occasionally serializes a tool
  call with unparseable arguments (e.g. two parallel calls merged into
  one). Such a reply would otherwise end the turn silently; instead the
  parse error is fed back and the model retries (see
  [turns](turns.md)). Parallel calls themselves are normal and kept.

Expect local models to be noticeably weaker at multi-step tool use than the
hosted ones. They are best for cheap iteration and offline work, with the
bench as the honest comparison.

## Models in the bench

`makker eval` uses Inspect's model layer, so model ids there take the
`provider/model` form, preserving any slashes or tags in the model ID, and
one run can matrix several models. The agent under test runs against the
bridge regardless of the target provider, which is how the same session
code serves every model. See [evaluation](evaluation.md).
