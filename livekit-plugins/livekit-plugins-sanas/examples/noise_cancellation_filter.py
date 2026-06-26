"""Minimal test for the Sanas NoiseCancellation audio filter.

Exercises the filter directly — no LiveKit room or full agent required.
Useful for verifying that the Sanas Remote SDK is installed correctly and
that your API key can reach the Sanas backend before wiring up a full agent.

Prerequisites
-------------
1. Install the Sanas Remote SDK::

       cd sanas_remote_sdk_<platform>/
       bash install.sh

2. Set the API key::

       export SANAS_API_KEY=sk_svgnv_...

Run
---
::

    python noise_cancellation_filter.py
"""

import asyncio
import logging
import struct

from livekit import rtc
from livekit.plugins import sanas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanas-nc-filter-test")

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 10
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 160 samples


def _make_frame(value: float = 0.1) -> rtc.AudioFrame:
    """Create a synthetic mono audio frame filled with a constant amplitude."""
    samples = [int(value * 32767)] * SAMPLES_PER_FRAME
    payload = struct.pack(f"<{len(samples)}h", *samples)
    return rtc.AudioFrame(payload, SAMPLE_RATE, 1, SAMPLES_PER_FRAME)


async def main() -> None:
    logger.info("Creating Sanas NoiseCancellation filter | model=AGENTIC_VI_G_NC")
    nc = sanas.NoiseCancellation(model_name=sanas.SanasModels.AGENTIC_VI_G_NC)

    # pre_initialize() is synchronous (blocks for SDK init + READY handshake).
    # Run it off the asyncio event loop so the loop stays responsive.
    logger.info("Pre-initialising — connecting to Sanas backend ...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, nc.pre_initialize)
    logger.info("Filter ready | initialized=%s", nc._initialized)

    logger.info("Processing 20 synthetic frames (200 ms) ...")
    for i in range(20):
        frame_in = _make_frame(value=0.1)
        frame_out = nc._process(frame_in)
        logger.info(
            "Frame %2d | in=%d samples  out=%d samples",
            i + 1,
            frame_in.samples_per_channel,
            frame_out.samples_per_channel,
        )

    logger.info("Closing filter and releasing Sanas processor ...")
    nc.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
