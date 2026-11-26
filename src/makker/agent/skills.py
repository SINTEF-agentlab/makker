"""Where a session's skills live, and what each one is called.

Kept apart from the agent builder so a front end that only lists the skills (the
MCP server, as guides for another agent) need not import the agent stack.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from makker.paths import SHARED_SKILLS_DIR
from makker.simulators.base import SimulatorAdapter


def skill_sources(adapter: SimulatorAdapter) -> list[str | tuple[str, str]]:
    """Skill sources for ``SkillsMiddleware``: the real skill directories.

    Bundled skills ship with the package (so they resolve at a real path even
    from a pip install, no repo checkout needed). The middleware reads their
    ``SKILL.md`` through the real-path backend. Labels are explicit. This is the
    seam for user/project skills later (append their real dirs, last wins).
    """

    sources: list[str | tuple[str, str]] = []
    if SHARED_SKILLS_DIR.exists():
        sources.append((str(SHARED_SKILLS_DIR), "Built-in"))
    if adapter.skills_dir.exists():
        sources.append((str(adapter.skills_dir), adapter.display_name))
    return sources


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path  # its SKILL.md


def skill_index(adapter: SimulatorAdapter) -> list[Skill]:
    """The skills in ``skill_sources``, by the name and description they declare."""
    skills: dict[str, Skill] = {}
    for source in skill_sources(adapter):
        root = Path(source if isinstance(source, str) else source[0])
        for path in sorted(root.glob("*/SKILL.md")):
            meta = _frontmatter(path)
            name = str(meta.get("name") or path.parent.name)
            # Last wins, as for the agent's own skills.
            skills[name] = Skill(name, str(meta.get("description") or "").strip(), path)
    return list(skills.values())


def _frontmatter(path: Path) -> dict[str, Any]:
    """A SKILL.md's YAML frontmatter, or nothing if it has none that parses."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    block = text[3:].split("\n---", 1)[0]
    with contextlib.suppress(yaml.YAMLError):
        meta = yaml.safe_load(block)
        if isinstance(meta, dict):
            return meta
    return {}
