"""Voice agent with Sanas Noise Cancellation.

Demonstrates how to plug ``sanas.NoiseCancellation()`` into the LiveKit audio
pipeline so that background noise is removed **before** the audio reaches VAD,
STT, and the LLM — improving transcription accuracy in noisy environments.

Prerequisites
-------------
1. Install the Sanas Remote SDK from the Sanas distribution::

       cd sanas_remote_sdk_<platform>/
       bash install.sh

2. Install this package::

       pip install livekit-plugins-sanas

3. Set environment variables (or add to a .env file)::

       LIVEKIT_URL=wss://<your-project>.livekit.cloud
       LIVEKIT_API_KEY=...
       LIVEKIT_API_SECRET=...
       SANAS_API_KEY=...

Run
---
::

    python noise_cancellation_agent.py dev

Audio pipeline
--------------
Browser mic → Sanas NoiseCancellation (AGENTIC_VI_G_NC) → VAD → STT
    → Turn Detector → LLM → TTS → Browser speaker
"""

import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
    room_io,
)
from livekit.plugins import sanas, silero

load_dotenv()

logger = logging.getLogger("sanas-nc-agent")


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. "
                "Give short, direct answers — one or two sentences at most. "
                "Never use lists, bullet points, or long explanations."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(allow_interruptions=False)


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()

    # Initialize Sanas NC during process warm-up so that the SDK's background
    # threads complete their startup before any session's room.connect() runs.
    # This prevents GIL contention that can cause LiveKit FFI timeouts on the
    # first call from a new network.
    nc = sanas.NoiseCancellation(model_name=sanas.SanasModels.AGENTIC_VI_G_NC)
    nc.pre_initialize()
    proc.userdata["nc"] = nc
    logger.info("Sanas NoiseCancellation pre-warmed | model=%s", nc.model)


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    nc: sanas.NoiseCancellation = ctx.proc.userdata["nc"]
    logger.info("Sanas NoiseCancellation ready | model=%s", nc.model)

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_detection=inference.TurnDetector(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        # NC reduces background noise, so fewer false interruptions —
        # resume agent speech if an interruption was caused by noise not voice.
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info("Usage: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    # Release the Sanas processor when the room disconnects.
    disconnected = asyncio.Event()
    ctx.room.on("disconnected", lambda *_: disconnected.set())

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Sanas NC sits at the front of the pipeline — VAD and STT
                # both receive clean, noise-cancelled audio.
                noise_cancellation=nc,
            ),
        ),
    )

    await disconnected.wait()
    nc.close()


if __name__ == "__main__":
    cli.run_app(server)
