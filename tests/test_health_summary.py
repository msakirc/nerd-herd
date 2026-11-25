"""health_summary — nerd_herd aggregates; the host app supplies app-level
probes through ``nerd_herd.hooks`` (it used to import five ``src.*`` modules,
one of which merely re-instantiated nerd_herd's own GPUCollector)."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from nerd_herd import hooks

# `import nerd_herd.health_summary as hs` would bind the FUNCTION: the package
# __init__ re-exports `health_summary`, shadowing the submodule attribute.
hs = importlib.import_module("nerd_herd.health_summary")


def _gpu(*, available=True, throttling=False, temp=70, vram_pct=50.0, ram_mb=16000):
    return SimpleNamespace(
        gpu=SimpleNamespace(
            available=available, is_throttling=throttling,
            temperature_c=temp, vram_usage_pct=vram_pct,
        ),
        ram_available_mb=ram_mb,
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    hooks.reset()
    monkeypatch.setattr(hs, "_gpu_state", lambda: _gpu())
    yield
    hooks.reset()


@pytest.mark.asyncio
async def test_no_hooks_registered_is_healthy_and_never_raises():
    out = await hs.health_summary()
    assert out["alerts"] == [] and out["issues"] == []
    assert out["local_model_healthy"] is True
    assert out["providers_degraded"] == [] and out["credentials_expired"] == []


@pytest.mark.asyncio
async def test_gpu_alerts_need_no_host_hooks(monkeypatch):
    monkeypatch.setattr(hs, "_gpu_state", lambda: _gpu(throttling=True, temp=91, ram_mb=1024))
    out = await hs.health_summary()
    assert out["gpu_throttling"] is True and out["low_ram"] is True
    assert any("throttling" in a for a in out["alerts"])
    assert any("Low RAM" in a for a in out["alerts"])


@pytest.mark.asyncio
async def test_vram_leak_needs_local_model_state(monkeypatch):
    monkeypatch.setattr(hs, "_gpu_state", lambda: _gpu(vram_pct=97.0))
    hooks.register(local_model_state=lambda: {"current_model": None, "is_loaded": False})
    out = await hs.health_summary()
    assert out["vram_leak_suspected"] is True
    assert any("possible leak" in a for a in out["alerts"])


@pytest.mark.asyncio
async def test_unhealthy_local_model():
    hooks.register(
        local_model_state=lambda: {"current_model": "qwen", "is_loaded": False},
        local_model_status=lambda: {"healthy": False, "loaded_model": "qwen"},
    )
    out = await hs.health_summary()
    assert out["local_model_healthy"] is False
    assert "llama-server unhealthy (model: qwen)" in out["issues"]
    assert "Local model unhealthy" in out["issues"]


@pytest.mark.asyncio
async def test_status_probe_failing_does_not_silence_the_other_local_checks(monkeypatch):
    """Review regression: one composite hook meant a get_status() error (e.g.
    mid-swap) also skipped the llama-unhealthy and VRAM-leak checks, which the
    pre-inversion code guarded separately."""
    def _boom():
        raise RuntimeError("swap in progress")

    monkeypatch.setattr(hs, "_gpu_state", lambda: _gpu(vram_pct=97.0))
    hooks.register(
        local_model_state=lambda: {"current_model": "qwen", "is_loaded": False},
        local_model_status=_boom,
    )
    out = await hs.health_summary()
    assert "llama-server unhealthy (model: qwen)" in out["issues"]
    assert out["vram_leak_suspected"] is True


@pytest.mark.asyncio
async def test_degraded_and_all_cloud_providers_down():
    hooks.register(
        cloud_breakers=lambda: {"groq": True, "gemini": True, "openrouter": False},
        cloud_providers=lambda: {"groq", "gemini"},
    )
    out = await hs.health_summary()
    assert out["providers_degraded"] == ["groq", "gemini"]
    assert any("ALL cloud providers are degraded" in a for a in out["alerts"])


@pytest.mark.asyncio
async def test_partial_degradation_is_not_an_alert():
    hooks.register(
        cloud_breakers=lambda: {"groq": True, "gemini": False},
        cloud_providers=lambda: {"groq", "gemini"},
    )
    out = await hs.health_summary()
    assert out["providers_degraded"] == ["groq"]
    assert out["alerts"] == []


@pytest.mark.asyncio
async def test_restore_hook_runs_and_expired_credentials_alert():
    calls = []

    async def _expired():
        return ["stripe"]

    hooks.register(restore_rate_limits=lambda: calls.append(1),
                   expired_credentials=_expired)
    out = await hs.health_summary()
    assert calls == [1]
    assert out["credentials_expired"] == ["stripe"]
    assert any("Credential 'stripe' has expired" in a for a in out["alerts"])


@pytest.mark.asyncio
async def test_a_failing_probe_does_not_sink_the_others(monkeypatch):
    def _boom():
        raise RuntimeError("probe down")

    monkeypatch.setattr(hs, "_gpu_state", lambda: _gpu(ram_mb=512))
    hooks.register(local_model_state=_boom, cloud_breakers=_boom)
    out = await hs.health_summary()
    assert out["low_ram"] is True  # GPU probe still ran


def test_unknown_hook_name_is_rejected():
    with pytest.raises(KeyError):
        hooks.register(not_a_hook=lambda: None)
