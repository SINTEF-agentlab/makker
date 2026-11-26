"""makker command-line entrypoint package.

``main.main`` is the entry point ``__main__.py`` and the tests use. The
subcommand modules (``init.py``, ``run.py``, ``transcript.py``) handle one
``argv[0]`` each; ``_helpers.py`` holds shared argparse plumbing.

Nothing is imported here, so the light helpers load without the commands, which
import LangChain (``makker mcp`` must start in seconds).
"""
