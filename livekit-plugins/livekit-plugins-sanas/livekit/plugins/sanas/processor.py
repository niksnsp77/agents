# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sanas Noise Cancellation audio filter for LiveKit Agents via the Sanas Remote SDK."""

from __future__ import annotations

import os
import struct
import threading
from typing import Any

from livekit import rtc

from .log import logger
from .models import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    DEFAULT_SAMPLE_RATE,
    SANAS_CHUNK_MS,
    SUPPORTED_SAMPLE_RATES,
    SanasModels,
)


class AudioFilter(rtc.FrameProcessor[rtc.AudioFrame]):
    """Sanas Noise Cancellation audio filter.

    Wraps the Sanas Remote SDK and plugs into LiveKit's ``noise_cancellation``
    pipeline slot.  Prefer the ``NoiseCancellation`` subclass for the standard API::

        sanas.NoiseCancellation()   # default model: AGENTIC_VI_G_NC

    ``AudioFilter`` accepts the same keyword arguments and can be used directly
    when a custom model name is needed.

    Audio flow per frame
    --------------------
    WebRTC frame (48 kHz, mono, int16)
        → resample to sdk_sample_rate (default 16 kHz)
        → buffer into 20 ms chunks
        → Sanas SDK ProcessSamples
        → resample output back to 48 kHz
        → return processed frame to LiveKit pipeline
    """

    def __init__(
        self,
        *,
        model_name: SanasModels | str | None = None,
        api_key: str | None = None,
        secure_media: bool = True,
        use_pcm16: bool | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        # Resolve credentials from kwargs or environment.
        self._model_name: str = str(model_name or os.getenv("SANAS_MODEL_NAME") or "")
        self._api_key: str = api_key or os.getenv("SANAS_API_KEY") or ""
        self._secure_media: bool = secure_media
        if use_pcm16 is None:
            use_pcm16 = os.getenv("SANAS_USE_PCM16", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self._use_pcm16: bool = use_pcm16
        self._sdk_sample_rate: int = (
            sample_rate if sample_rate in SUPPORTED_SAMPLE_RATES else DEFAULT_SAMPLE_RATE
        )
        self._ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS

        missing = [
            name
            for name, value in [
                ("SANAS_API_KEY / api_key", self._api_key),
                ("SANAS_MODEL_NAME / model_name", self._model_name),
            ]
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required Sanas config: {', '.join(missing)}")

        # SDK handles — set in _initialize(), cleared in _reset()/_destroy().
        self._sdk: Any | None = None
        self._processor: Any | None = None

        # Lifecycle state.
        self._initialized: bool = False
        self._init_failed: bool = False
        self._closed: bool = False
        self._buffer: list[float] = []
        # Re-chunks 20ms SDK output back into the original input frame size.
        self._out_buffer: list[float] = []
        self._enabled: bool = True

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "sanas"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # ── Optional eager initialization ─────────────────────────────────────────

    def pre_initialize(self, sample_rate: int | None = None) -> None:
        """Eagerly initialize the Sanas SDK before the first audio frame arrives.

        The SDK initialization (backend connection + READY handshake) is
        synchronous and can take several seconds.  Without eager initialization,
        this blocking work happens on the audio processing thread when the first
        frame arrives, causing a frame backlog and high output jitter.

        Call from the agent entrypoint via ``run_in_executor`` so the blocking
        work runs off the asyncio event loop::

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, nc.pre_initialize)
            # Pass nc to RoomOptions — no frame backlog possible.
            await session.start(
                agent=agent,
                room=ctx.room,
                room_options=room_io.RoomOptions(
                    audio_input=room_io.AudioInputOptions(noise_cancellation=nc),
                ),
            )
        """
        if self._initialized or self._init_failed:
            return
        rate = sample_rate if sample_rate is not None else self._sdk_sample_rate
        self._initialize(rate)

    # ── LiveKit FrameProcessor callbacks ──────────────────────────────────────

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled:
            return frame

        if frame.num_channels != 1:
            logger.warning(
                "Sanas expects mono audio; got %d channels — passing through unchanged",
                frame.num_channels,
            )
            return frame

        # int16 → float32 [-1.0, 1.0]
        in_i16 = struct.unpack(f"<{frame.samples_per_channel}h", frame.data)
        in_f32 = [s / 32768.0 for s in in_i16]

        # Resample to the rate the SDK processes at (default 16 kHz).
        sdk_rate = self._sdk_sample_rate
        if sdk_rate != frame.sample_rate:
            in_f32 = _resample(in_f32, frame.sample_rate, sdk_rate)

        # Send through the SDK in 20 ms chunks.
        out_f32 = self._process_buffered(in_f32, sample_rate=sdk_rate)

        # Resample output back to the incoming frame rate and add to output buffer.
        if out_f32:
            if sdk_rate != frame.sample_rate:
                out_f32 = _resample(out_f32, sdk_rate, frame.sample_rate)
            self._out_buffer.extend(out_f32)

        # Always return a frame with exactly the same number of samples as the
        # input so that downstream VAD and STT receive consistent frame sizes.
        # During the initial buffering period (first ~10ms) we pad with zeros.
        n = frame.samples_per_channel
        if len(self._out_buffer) >= n:
            frame_out = self._out_buffer[:n]
            del self._out_buffer[:n]
        else:
            frame_out = self._out_buffer + [0.0] * (n - len(self._out_buffer))
            self._out_buffer = []

        # float32 → int16, pack into bytes.
        out_i16 = [int(max(-1.0, min(1.0, s)) * 32767.0) for s in frame_out]
        payload = struct.pack(f"<{len(out_i16)}h", *out_i16)

        return rtc.AudioFrame(payload, frame.sample_rate, frame.num_channels, len(out_i16))

    def _close(self) -> None:
        """Called by LiveKit when the subscribed track changes.

        Resets the processor so it re-initializes on the first frame of the
        new track.  The same ``AudioFilter`` object is reused across tracks so
        we must NOT permanently close here.
        """
        logger.info("Sanas AudioFilter: track change — resetting for new track")
        self._reset()

    def close(self) -> None:
        """Permanent session teardown — call when the room disconnects."""
        logger.info("Sanas AudioFilter: session teardown")
        self._destroy()

    # ── SDK lifecycle (private) ────────────────────────────────────────────────

    def _initialize(self, sample_rate: int) -> None:
        """Synchronously initialize the Sanas Remote SDK.

        Blocks until the Sanas backend acknowledges the session (up to
        ``DEFAULT_READY_TIMEOUT_SECONDS``).  Always call via ``pre_initialize()``
        which runs this in a thread executor off the asyncio event loop.
        """
        try:
            import sanas_remote_sdk  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "sanas_remote_sdk is not installed. "
                "Install it from the Sanas Remote SDK distribution."
            ) from exc

        if not all(
            hasattr(sanas_remote_sdk, attr)
            for attr in ("CreateRemoteSDK", "InitParams", "AudioParams")
        ):
            raise RuntimeError(
                "Unsupported sanas_remote_sdk version: missing CreateRemoteSDK / InitParams / AudioParams"
            )

        logger.info(
            "Sanas SDK initializing | model=%s rate=%d Hz",
            self._model_name,
            sample_rate,
        )

        sdk = sanas_remote_sdk.CreateRemoteSDK()
        if not sdk:
            raise RuntimeError("sanas_remote_sdk.CreateRemoteSDK() returned None")

        init_params = sanas_remote_sdk.InitParams()
        init_params.apiKey = self._api_key
        init_params.secureMedia = self._secure_media

        result = sdk.Initialize(init_params)
        if result != sanas_remote_sdk.InitSDKResult.SUCCESS:
            raise RuntimeError(f"Sanas SDK init failed: {result}")

        audio_params = sanas_remote_sdk.AudioParams()
        audio_params.modelName = self._model_name
        audio_params.sampleRate = sample_rate
        audio_params.usePcm16 = self._use_pcm16

        # Wire up state callback so we can wait for READY.
        ready_evt = threading.Event()
        failed_evt = threading.Event()
        failed_reason: dict[str, str] = {"msg": ""}

        ProcessorState = getattr(sanas_remote_sdk, "ProcessorState", None)
        ready_state = getattr(ProcessorState, "READY", None)
        failed_states = {
            s
            for s in (
                getattr(ProcessorState, "FAILED", None),
                getattr(ProcessorState, "DISCONNECTED", None),
            )
            if s is not None
        }

        def _on_state(state: int, reason: str) -> None:
            initializing_state = getattr(ProcessorState, "INITIALIZING", None)
            if initializing_state is not None and state == initializing_state:
                logger.debug("Sanas processor INITIALIZING | reason=%s", reason)
            elif ready_state is not None and state == ready_state:
                ready_evt.set()
            elif state in failed_states:
                failed_reason["msg"] = reason
                failed_evt.set()

        processor, create_result = sdk.CreateAudioProcessor(audio_params, _on_state)
        if create_result != sanas_remote_sdk.CreateProcessorResult.SUCCESS:
            sdk.Shutdown()
            raise RuntimeError(
                f"Failed to create Sanas processor for model '{self._model_name}': {create_result}"
            )

        if ready_state is not None:
            if not ready_evt.wait(timeout=self._ready_timeout):
                sdk.DestroyAudioProcessor(processor)
                sdk.Shutdown()
                if failed_evt.is_set():
                    raise RuntimeError(
                        f"Sanas processor failed to become ready: {failed_reason['msg']}"
                    )
                raise RuntimeError(
                    f"Timeout waiting for Sanas processor READY ({self._ready_timeout:.1f}s)"
                )

        self._sdk = sdk
        self._processor = processor
        self._initialized = True
        logger.info("Sanas SDK ready | model=%s rate=%d Hz", self._model_name, sample_rate)

    def _process_buffered(self, samples: list[float], *, sample_rate: int) -> list[float]:
        """Accumulate samples and call ProcessSamples in strict 20 ms chunks."""
        if self._closed or self._init_failed:
            return [0.0] * len(samples)

        if not self._initialized:
            # Lazy init — blocks the calling thread once.
            logger.warning(
                "Sanas: initializing on first audio frame — "
                "add nc.pre_initialize() to prewarm() to avoid a delay here"
            )
            try:
                self._initialize(sample_rate)
            except Exception:
                self._init_failed = True
                logger.exception("Sanas SDK lazy-init failed — returning silence")
                return [0.0] * len(samples)

        chunk_size = int(sample_rate * SANAS_CHUNK_MS / 1000)
        self._buffer.extend(samples)

        assert self._processor is not None
        output: list[float] = []
        while len(self._buffer) >= chunk_size:
            chunk = self._buffer[:chunk_size]
            del self._buffer[:chunk_size]
            output.extend(self._processor.ProcessSamples(list(chunk)))
        return output

    def _reset(self) -> None:
        """Tear down the current SDK session and prepare for re-initialization."""
        if self._initialized:
            # Gate any in-flight _process calls while tearing down the SDK.
            self._closed = True
            self._destroy_sdk()
        self._closed = False
        self._initialized = False
        self._init_failed = False
        self._buffer = []
        self._out_buffer = []
        self._processor = None
        self._sdk = None

    def _destroy(self) -> None:
        """Permanently close the SDK (session teardown)."""
        self._closed = True
        self._destroy_sdk()
        self._buffer = []
        self._out_buffer = []
        self._processor = None
        self._sdk = None

    def _destroy_sdk(self) -> None:
        if self._sdk is not None and self._processor is not None:
            try:
                self._sdk.DestroyAudioProcessor(self._processor)
            except Exception:
                logger.exception("Error destroying Sanas audio processor")
        if self._sdk is not None and hasattr(self._sdk, "Shutdown"):
            try:
                self._sdk.Shutdown()
            except Exception:
                logger.exception("Error shutting down Sanas SDK")


# ── Module-level resample helper ───────────────────────────────────────────────


def _resample(samples: list[float], from_rate: int, to_rate: int) -> list[float]:
    """Linear-interpolation resample for float32 sample lists."""
    if from_rate == to_rate or not samples:
        return samples
    import numpy as np  # lazy import — only needed when rates differ

    arr = np.asarray(samples, dtype=np.float32)
    out_len = int(len(arr) * to_rate / from_rate)
    src_idx = np.arange(len(arr))
    dst_idx = np.linspace(0, len(arr) - 1, out_len)
    return list(np.interp(dst_idx, src_idx, arr).tolist())


# ── Named subclasses — preferred public API ────────────────────────────────────


class NoiseCancellation(AudioFilter):
    """Sanas Noise Cancellation audio filter.

    Default model: ``AGENTIC_VI_G_NC``.  Override via the ``model_name``
    argument or the ``SANAS_MODEL_NAME`` environment variable.

    Usage::

        sanas_nc = sanas.NoiseCancellation()

        session = AgentSession(
            ...,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(noise_cancellation=sanas_nc),
            ),
        )

        # On room disconnect:
        sanas_nc.close()
    """

    def __init__(
        self,
        *,
        model_name: SanasModels | str | None = None,
        api_key: str | None = None,
        secure_media: bool = True,
        use_pcm16: bool | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            model_name=model_name or os.getenv("SANAS_MODEL_NAME", "AGENTIC_VI_G_NC"),
            api_key=api_key,
            secure_media=secure_media,
            use_pcm16=use_pcm16,
            sample_rate=sample_rate,
        )
