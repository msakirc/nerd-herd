"""Injected host probes — nerd_herd's ONLY sanctioned way to read app-level
state (local model manager, cloud capacity tracker, model registry,
credential vault) without importing ``src.*``. Same pattern as
``dabidabi.hooks``.

The host wires real implementations once at startup via ``register(...)``
(KutAI: ``src/infra/nerd_herd_hooks.py::wire``, called from ``run.py``).
Every hook is ``None`` until registered; callers null-check and simply skip
that probe — which is also what a standalone nerd_herd does.
"""
from typing import Any, Awaitable, Callable, Optional

# -> {"current_model": str|None, "is_loaded": bool}
local_model_state: Optional[Callable[[], dict]] = None

# -> the local model manager's get_status() dict ({"healthy", "loaded_model", ...}).
# Separate from local_model_state so a status error (e.g. mid-swap) can't
# silence the llama-unhealthy / VRAM-leak checks.
local_model_status: Optional[Callable[[], dict]] = None

# -> {provider_name: circuit_breaker_open}
cloud_breakers: Optional[Callable[[], dict]] = None

# -> set of configured cloud provider names
cloud_providers: Optional[Callable[[], set]] = None

# Side effect run on every health tick (KDV periodic limit restore).
restore_rate_limits: Optional[Callable[[], Any]] = None

# -> awaitable list of credential service names that have expired
expired_credentials: Optional[Callable[[], Awaitable[list]]] = None

_HOOK_NAMES = frozenset({
    "local_model_state", "local_model_status", "cloud_breakers", "cloud_providers",
    "restore_rate_limits", "expired_credentials",
})


def register(**kwargs: Callable) -> None:
    """Set one or more hooks by name. Unknown names raise (typo guard)."""
    g = globals()
    for name, fn in kwargs.items():
        if name not in _HOOK_NAMES:
            raise KeyError(f"unknown nerd_herd hook: {name!r}")
        g[name] = fn


def reset() -> None:
    """Clear all hooks (test isolation)."""
    g = globals()
    for name in _HOOK_NAMES:
        g[name] = None
