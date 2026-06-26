"""Sanas plugin models and constants."""

from enum import Enum

# Sanas SDK expects 20ms audio chunks
SANAS_CHUNK_MS = 20

# Sample rates natively accepted by the Sanas Remote SDK.
# Anything not in this set is resampled down to DEFAULT_SAMPLE_RATE before processing.
# Update this tuple if Sanas adds support for additional rates in a future SDK release.
SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 48000)

# Default configuration values
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_READY_TIMEOUT_SECONDS = 10.0


class SanasModels(str, Enum):
    """Available Sanas Noise Cancellation models.

    Full model specifications and audio samples:
    https://developer.sanas.ai/Docs/Models/Overview
    """

    # Human ↔ Human: isolates primary speaker, optimised for human listeners.
    # Latency ~40ms, up to 24kHz. Recommended: sample_rate=24000.
    VI_G_NC3_0 = "VI_G_NC3.0"

    # Human ↔ Machine: removes all background noise and non-primary voices
    # for complete speaker isolation. Latency ~100ms, 16kHz. Recommended: sample_rate=16000 (default).
    AGENTIC_VI_G_NC = "AGENTIC_VI_G_NC"

    # Human ↔ Machine: telephony-optimised variant of AGENTIC_VI_G_NC for
    # 8kHz narrowband audio. Latency ~100ms, 8kHz. Recommended: sample_rate=8000.
    AGENTIC_VI_GT_NC = "AGENTIC_VI_GT_NC"

    # Human ↔ Machine: removes background noise while preserving all human
    # speech — use in multi-speaker environments. Latency ~100ms, 16kHz. Recommended: sample_rate=16000 (default).
    AGENTIC_ST_NC = "AGENTIC_ST_NC"

    def __str__(self) -> str:
        return self.value
