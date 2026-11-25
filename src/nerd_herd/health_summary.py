"""health_summary — ported from general_beckman.watchdog.check_resources.

Returns a structured dict describing the current resource health state.
All telegram.send_notification calls are dropped — alerts reach Telegram
via the nerd_herd_health_alert cron marker which spawns mr_roboto.notify_user.

GPU / RAM come from nerd_herd's own collector. App-level probes (local model
manager, cloud circuit breakers, model registry, credential vault) come from
``nerd_herd.hooks`` — registered by the host at startup; an unregistered
probe is skipped. Each probe is isolated: one failing never sinks the rest.
"""
from __future__ import annotations

import logging

from nerd_herd import hooks
from nerd_herd.health import HealthRegistry

_registry = HealthRegistry()
_log = logging.getLogger("nerd_herd.health_summary")
_gpu_collector = None


def _gpu_state():
    global _gpu_collector
    if _gpu_collector is None:
        from nerd_herd.gpu import GPUCollector
        _gpu_collector = GPUCollector()
    return _gpu_collector.get_state()


async def health_summary() -> dict:
    """Check system resource health and return a structured summary.

    Returns:
        {
            "alerts": [str, ...],           # high-severity strings for Telegram alerting
            "issues": [str, ...],           # full issue list for logs / /status
            "providers_degraded": [...],    # list of degraded cloud provider names
            "vram_leak_suspected": bool,
            "gpu_throttling": bool,
            "low_ram": bool,
            "credentials_expired": [...],   # list of expired credential service names
            "local_model_healthy": bool,
        }
    """
    resource_issues: list[str] = []
    alerts: list[str] = []
    providers_degraded: list[str] = []
    vram_leak_suspected = False
    gpu_throttling = False
    low_ram = False
    credentials_expired: list[str] = []
    local_model_healthy = True

    # ── Local model manager snapshot (host probe) ──
    local = None
    if hooks.local_model_state is not None:
        try:
            local = hooks.local_model_state()
        except Exception as e:
            _log.warning(f"Local model check failed: {e}")

    # ── Check llama-server health ──
    if local is not None and local.get("current_model") and not local.get("is_loaded"):
        resource_issues.append(f"llama-server unhealthy (model: {local['current_model']})")
        local_model_healthy = False

    # ── Check GPU health ──
    try:
        gpu_state = _gpu_state()

        if gpu_state.gpu.available:
            if gpu_state.gpu.is_throttling:
                msg = (
                    f"GPU thermal throttling! "
                    f"Temp: {gpu_state.gpu.temperature_c}°C"
                )
                resource_issues.append(msg)
                alerts.append(msg)
                gpu_throttling = True

            if (local is not None and gpu_state.gpu.vram_usage_pct > 95
                    and not local.get("is_loaded")):
                msg = (
                    f"VRAM nearly full ({gpu_state.gpu.vram_usage_pct:.0f}%) "
                    f"but no model loaded — possible leak"
                )
                resource_issues.append(msg)
                alerts.append(msg)
                vram_leak_suspected = True

        if gpu_state.ram_available_mb < 2048:
            msg = f"Low RAM: {gpu_state.ram_available_mb}MB available"
            resource_issues.append(msg)
            alerts.append(msg)
            low_ram = True

    except Exception as e:
        _log.warning(f"GPU health check failed: {e}")

    # ── Check local model status (separate probe: its failure must not
    #    silence the checks above) ──
    if hooks.local_model_status is not None:
        try:
            status = hooks.local_model_status() or {}
            if not status.get("healthy", True) and status.get("loaded_model"):
                resource_issues.append("Local model unhealthy")
                local_model_healthy = False
        except Exception as e:
            _log.warning(f"Local model status check failed: {e}")

    # ── Check circuit breakers (cloud providers) ──
    if hooks.cloud_breakers is not None:
        try:
            degraded = [p for p, is_open in hooks.cloud_breakers().items() if is_open]
            if degraded:
                providers_degraded = degraded
                resource_issues.append(f"Degraded providers: {', '.join(degraded)}")

                all_cloud = set(hooks.cloud_providers()) if hooks.cloud_providers else set()
                if all_cloud and all_cloud.issubset(set(degraded)):
                    critical_msg = (
                        "ALL cloud providers are degraded! "
                        "Only local inference available."
                    )
                    resource_issues.append(critical_msg)
                    alerts.append(critical_msg)
        except Exception as e:
            _log.warning(f"Circuit breaker check failed: {e}")

    # ── Restore rate limits ──
    if hooks.restore_rate_limits is not None:
        try:
            hooks.restore_rate_limits()
        except Exception:
            pass

    # ── Check expiring credentials ──
    if hooks.expired_credentials is not None:
        try:
            for svc in await hooks.expired_credentials():
                msg = f"Credential '{svc}' has expired. Refresh with /credential add."
                resource_issues.append(msg)
                alerts.append(msg)
                credentials_expired.append(svc)
        except Exception as e:
            _log.warning(f"Credential expiry check failed: {e}")

    return {
        "alerts": alerts,
        "issues": resource_issues,
        "providers_degraded": providers_degraded,
        "vram_leak_suspected": vram_leak_suspected,
        "gpu_throttling": gpu_throttling,
        "low_ram": low_ram,
        "credentials_expired": credentials_expired,
        "local_model_healthy": local_model_healthy,
    }
