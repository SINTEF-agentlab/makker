"""What a session can be started with: the model catalog, provider API keys,
and the simulators (with any installed branding for the welcome screen).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from makker.interfaces.server import protocol
from makker.interfaces.server.deps import DefaultsDep

router = APIRouter()


class CredentialRequest(BaseModel):
    # ``provider`` is a catalog name, label, or model id; the server resolves it
    # to the provider's key variable so the UI never sends a raw env-var name.
    provider: str
    value: str


def _installed_branding() -> dict[str, Any] | None:
    """What the installed extensions call the web UI, or ``None`` when unbranded.

    Read from the entry points rather than from a session, because the welcome
    screen is the first thing the page draws and no session exists yet. Only
    installed capabilities can brand it; the per-session layers a host sends on
    create describe one launch, not the product.
    """
    from makker.agent.capabilities import (
        collect_branding,
        discover_extensions,
        select_for_surface,
    )

    branding = collect_branding(select_for_surface(discover_extensions(), "web"))
    if branding is None:
        return None
    return {
        "display_name": branding.display_name,
        "tagline": branding.tagline,
        "examples": list(branding.example_prompts),
    }


class PinRequest(BaseModel):
    model: str
    pinned: bool


@router.get("/models")
def list_models(defaults: DefaultsDep) -> dict[str, Any]:
    from makker.models import DEFAULT_MODEL, PROVIDERS, discover_models

    # Every model the provider data lists, newest first per provider. Read from
    # the bundled data with no model instantiation, so it is fast at page load.
    models = [info.to_wire() for infos in discover_models().values() for info in infos]
    # Report the server's actual default: the launch ``--model`` if one was given,
    # else the catalog default. The UI seeds its model from this, so it must match
    # what new sessions use.
    return {
        "default": defaults.model or DEFAULT_MODEL,
        "providers": sorted(PROVIDERS),
        "models": models,
        # The wire-contract version for third-party front ends (see
        # docs/server-interface.md); the bundled UI ships in lockstep.
        "protocol": protocol.PROTOCOL_VERSION,
    }


@router.get("/models/live")
async def list_live_models() -> dict[str, Any]:
    """The full catalog refreshed from provider APIs and the Ollama daemon, for
    the "More models" list. Slower than ``/models``, so the page asks lazily."""
    import asyncio

    from makker.models import discover_available_models, discover_ollama_models

    cloud = await asyncio.to_thread(discover_available_models)
    local = await discover_ollama_models()
    models = [info for infos in cloud.values() for info in infos] + local
    return {"models": [info.to_wire() for info in models]}


@router.get("/models/menu")
async def model_menu(current: str | None = None) -> dict[str, Any]:
    """The model menu's short list: pinned, recent, the newest per provider with
    a key, installed local models, and the providers that still need a key."""
    from makker.credentials import load_user_credentials
    from makker.model_choices import load_pinned, missing_keys, short_list
    from makker.models import discover_models, discover_ollama_models
    from makker.recent_models import load_recent_models

    load_user_credentials()
    models = [info for infos in discover_models().values() for info in infos]
    models += await discover_ollama_models()
    sections = short_list(
        models, current=current, pinned=load_pinned(), recent=load_recent_models()
    )
    return {
        "sections": [
            {"title": title, "models": [m.to_wire() for m in rows]} for title, rows in sections
        ],
        "add_key": [{"provider": p.name, "label": p.label} for p in missing_keys()],
        "pinned": load_pinned(),
    }


@router.post("/models/pinned")
def pin_model(req: PinRequest) -> dict[str, Any]:
    """Pin a model to the top of the menu, or unpin it."""
    from makker.model_choices import set_pinned

    return {"pinned": set_pinned(req.model, req.pinned)}


@router.get("/models/window")
def model_window(model: str | None = None) -> dict[str, Any]:
    """A model's context window (for the % indicator), where it came from
    (``provider``, ``ollama`` or ``assumed``), and its compaction point.

    Separate from ``/models`` because a model missing from the provider data is
    looked up live, so the UI asks for just the active model, lazily.
    """
    from makker.context_budget import context_budget
    from makker.models import DEFAULT_MODEL

    budget = context_budget(model or DEFAULT_MODEL)
    return {
        "model": model or DEFAULT_MODEL,
        "window": budget.window,
        "window_source": budget.source,
        "compact_at": budget.compact_at,
    }


@router.get("/credentials")
def list_credentials() -> dict[str, Any]:
    """Which provider keys are configured, so the UI can prompt for a missing one.

    The masked previews are safe to show (a few characters, the rest hidden)
    and let the user confirm which key is saved; full secrets never cross the
    wire. ``shadowed`` flags a saved key that an environment value overrides.
    """
    from makker.credentials import key_status, user_env_path

    return {
        "path": str(user_env_path()),
        "providers": [
            {
                "provider": st.provider,
                "label": st.label,
                "env_var": st.env_var,
                "is_set": st.is_set,
                "masked": st.masked,
                "source": st.source,
                "shadowed": st.shadowed,
            }
            for st in key_status()
        ],
    }


@router.post("/credentials")
def set_credential(req: CredentialRequest) -> dict[str, Any]:
    """Save a provider's API key to the global ``.env`` and use it immediately."""
    from makker.credentials import store_credential_for_provider

    try:
        info, path = store_credential_for_provider(req.provider, req.value)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"unknown provider {req.provider!r}") from exc
    except ValueError as exc:  # empty key
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"provider": info.name, "env_var": info.key_env_var, "path": str(path)}


@router.get("/simulators")
def list_simulators(defaults: DefaultsDep) -> dict[str, Any]:
    from makker.simulators import registry

    names = registry.names()
    details = {}
    for name in names:
        adapter = registry.get(name)
        details[name] = {
            "display_name": adapter.display_name,
            "examples": list(adapter.example_prompts),
        }
    return {
        "simulators": names,
        "default": defaults.sim,
        "details": details,
        "branding": _installed_branding(),
    }
