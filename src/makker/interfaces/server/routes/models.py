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


@router.get("/models")
def list_models(defaults: DefaultsDep) -> dict[str, Any]:
    from makker.models import DEFAULT_MODEL, PROVIDERS, discover_models

    # The selectable models for the UI's model picker (provider profile data,
    # no model instantiation), grouped flat with their provider.
    models = [
        {"id": info.id, "label": info.label, "provider": provider, "note": info.note}
        for provider, infos in discover_models().items()
        for info in infos
    ]
    # Report the server's actual default: the launch ``--model`` if one was given,
    # else the catalog default. The UI seeds its model from this, so it must match
    # what new sessions use, or the UI would show the wrong model, query the wrong
    # context window, and resume a from-disk session onto the catalog default.
    return {
        "default": defaults.model or DEFAULT_MODEL,
        "providers": sorted(PROVIDERS),
        "models": models,
        # The wire-contract version for third-party front ends (see
        # docs/server-interface.md); the bundled UI ships in lockstep.
        "protocol": protocol.PROTOCOL_VERSION,
    }


@router.get("/models/live")
def list_live_models() -> dict[str, Any]:
    """Refresh the selector from provider APIs without delaying page startup."""
    from makker.models import discover_available_models

    return {
        "models": [
            {"id": info.id, "label": info.label, "provider": provider, "note": info.note}
            for provider, infos in discover_available_models().items()
            for info in infos
        ]
    }


@router.get("/models/window")
def model_window(model: str | None = None) -> dict[str, Any]:
    """The context window for a model (for the % indicator), or null if unknown.

    Separate from ``/models`` because it instantiates the model to read its
    profile, so the UI asks for just the active model, lazily.
    """
    from makker.models import DEFAULT_MODEL, context_window

    return {"model": model or DEFAULT_MODEL, "window": context_window(model or DEFAULT_MODEL)}


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
