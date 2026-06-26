from __future__ import annotations

import struct
import sys
import types

import pytest

from livekit.plugins.sanas import AudioFilter, NoiseCancellation

# ---------------------------------------------------------------------------
# Shared fake SDK factory (new CreateRemoteSDK / InitParams / AudioParams API)
# ---------------------------------------------------------------------------


def _make_fake_sdk_module(
    *,
    init_ok: bool = True,
    processor_obj: object | None = None,
) -> types.ModuleType:
    """Return a minimal fake sanas_remote_sdk module matching the 1.1.0 API."""

    class _FakeProcessor:
        def ProcessSamples(self, chunk: list[float]) -> list[float]:
            return [s * 0.5 for s in chunk]

    class _InitSDKResult:
        SUCCESS = "SUCCESS"

    class _CreateProcessorResult:
        SUCCESS = "SUCCESS"

    class _FakeSDK:
        def Initialize(self, _params: object) -> object:
            return _InitSDKResult.SUCCESS if init_ok else "FAILED"

        def CreateAudioProcessor(
            self, _audio_params: object, _cb: object = None
        ) -> tuple[object, object]:
            proc = processor_obj if processor_obj is not None else _FakeProcessor()
            result = _CreateProcessorResult.SUCCESS if proc is not None else "FAILED"
            return proc, result

        def DestroyAudioProcessor(self, _proc: object) -> None:
            pass

        def Shutdown(self) -> None:
            pass

    class _FakeInitParams:
        apiKey: str = ""
        secureMedia: bool = True

    class _FakeAudioParams:
        modelName: str = ""
        sampleRate: int = 16000
        usePcm16: bool = False

    mod = types.ModuleType("sanas_remote_sdk")
    mod.CreateRemoteSDK = _FakeSDK
    mod.InitParams = _FakeInitParams
    mod.AudioParams = _FakeAudioParams
    mod.InitSDKResult = _InitSDKResult
    mod.CreateProcessorResult = _CreateProcessorResult
    return mod


def _inject_fake_sdk(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> None:
    mod = _make_fake_sdk_module(**kwargs)
    monkeypatch.setitem(sys.modules, "sanas_remote_sdk", mod)


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANAS_API_KEY", "sk_svgnv_testkey")


# ---------------------------------------------------------------------------
# Construction and model defaults
# ---------------------------------------------------------------------------


def test_nc_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    nc = NoiseCancellation()
    assert nc.model == "AGENTIC_VI_G_NC"
    assert nc.provider == "sanas"


def test_custom_model_via_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    nc = NoiseCancellation(model_name="AGENTIC_VI_G_NC")
    assert nc.model == "AGENTIC_VI_G_NC"


def test_custom_model_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    monkeypatch.setenv("SANAS_MODEL_NAME", "AGENTIC_VI_G_NC")
    nc = NoiseCancellation()
    assert nc.model == "AGENTIC_VI_G_NC"


def test_missing_config_raises() -> None:
    with pytest.raises(RuntimeError, match="Missing required Sanas config"):
        AudioFilter(model_name="VI_G_NC3.0")  # no SANAS_API_KEY


# ---------------------------------------------------------------------------
# Audio processing — 20ms chunk buffering
# ---------------------------------------------------------------------------


def test_process_buffers_into_20ms_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """_process_buffered must accumulate samples and call ProcessSamples in
    exact 20ms chunks (320 samples at 16 kHz)."""
    _inject_fake_sdk(monkeypatch)
    _env(monkeypatch)

    nc = NoiseCancellation()
    # Manually initialize to avoid blocking the test thread.
    nc._initialize(sample_rate=16000)

    chunk_size = 320  # 20ms @ 16 kHz

    # Feed 100 samples — not enough for one chunk; expect empty output.
    out = nc._process_buffered([0.2] * 100, sample_rate=16000)
    assert out == []

    # Feed 220 more — total 320 samples of 0.2; expect one fully processed chunk.
    out = nc._process_buffered([0.2] * 220, sample_rate=16000)
    assert len(out) == chunk_size
    # FakeProcessor halves each sample: 0.2 * 0.5 = 0.1
    assert all(abs(s - 0.1) < 1e-6 for s in out)


def test_process_returns_silence_on_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_fake_sdk(monkeypatch, init_ok=False)
    _env(monkeypatch)

    nc = NoiseCancellation()
    samples = [0.5] * 320
    out = nc._process_buffered(samples, sample_rate=16000)
    assert nc._init_failed is True
    assert all(s == 0.0 for s in out)


# ---------------------------------------------------------------------------
# _process() — full AudioFrame path (resampling + int16 conversion)
# ---------------------------------------------------------------------------


def _make_frame(samples: list[float], sample_rate: int) -> object:
    """Build a minimal fake rtc.AudioFrame from float32 samples."""
    from livekit import rtc

    data = struct.pack(f"<{len(samples)}h", *[int(s * 32767) for s in samples])
    return rtc.AudioFrame(data, sample_rate, 1, len(samples))


def test_process_frame_passthrough_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_fake_sdk(monkeypatch)
    _env(monkeypatch)

    nc = NoiseCancellation()
    nc.enabled = False
    frame = _make_frame([0.5] * 480, sample_rate=48000)
    result = nc._process(frame)
    assert result is frame  # unchanged


def test_process_frame_stereo_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_fake_sdk(monkeypatch)
    _env(monkeypatch)

    nc = NoiseCancellation()
    from livekit import rtc

    data = struct.pack("<4h", 100, 200, 300, 400)
    frame = rtc.AudioFrame(data, 16000, 2, 2)
    result = nc._process(frame)
    assert result is frame  # stereo passed through unchanged


# ---------------------------------------------------------------------------
# Lifecycle — track change reset vs permanent close
# ---------------------------------------------------------------------------


def test_close_resets_for_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """_close() (track change) must reset state so the filter can re-init."""
    _inject_fake_sdk(monkeypatch)
    _env(monkeypatch)

    nc = NoiseCancellation()
    nc._initialize(sample_rate=16000)
    assert nc._initialized is True

    nc._close()

    assert nc._initialized is False
    assert nc._closed is False  # must NOT be permanently closed
    assert nc._processor is None
    assert nc._sdk is None


def test_permanent_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """close() (session teardown) must set _closed=True permanently."""
    _inject_fake_sdk(monkeypatch)
    _env(monkeypatch)

    nc = NoiseCancellation()
    nc._initialize(sample_rate=16000)
    nc.close()

    assert nc._closed is True
    assert nc._processor is None
    assert nc._sdk is None


def test_close_before_init_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_fake_sdk(monkeypatch)
    _env(monkeypatch)

    nc = NoiseCancellation()
    nc.close()  # must not raise
    assert nc._closed is True
