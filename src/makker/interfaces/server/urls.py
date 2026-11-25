"""Public URLs for a server that may sit under a base path.

A reverse proxy can serve the UI under a prefix such as ``/restricted``; every
site-relative URL the server hands the browser goes through ``public_url``.
"""

from __future__ import annotations


def normalize_base_path(raw: str | None) -> str:
    """Return ``""`` or a leading-slash prefix with no trailing slash (e.g. ``/restricted``)."""
    if not raw:
        return ""
    p = raw.strip()
    if not p or p == "/":
        return ""
    if not p.startswith("/"):
        p = "/" + p
    return p.rstrip("/")


def public_url(base_path: str, path: str) -> str:
    """Prefix a site-relative path with ``base_path``; leave absolute URLs alone."""
    if not path.startswith("/"):
        return path
    prefix = normalize_base_path(base_path)
    return f"{prefix}{path}" if prefix else path
