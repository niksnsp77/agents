# Sanas plugin for LiveKit Agents

Real-time Noise Cancellation for LiveKit voice agents using the [Sanas Remote SDK](https://www.sanas.ai/).

## Features

- **`sanas.NoiseCancellation()`**: Real-time noise cancellation FrameProcessor for audio processing in voice agents.

## Installation

```bash
# 1. Download the Sanas Remote SDK from https://www.sanas.ai/developer-platform
#    then install and activate it:
cd sanas_remote_sdk_<platform>/
bash install.sh
source sanas_remote_sdk/bin/activate

# 2. Install livekit-agents and the plugin into the activated environment
pip install livekit-agents livekit-plugins-sanas
```

**Note:** The `sanas_remote_sdk` package is a proprietary native SDK not available on public PyPI.
It must be obtained from the [Sanas Developer Platform](https://www.sanas.ai/developer-platform) after onboarding.
The `install.sh` script creates a Python virtual environment inside the SDK folder.
All subsequent commands must run inside this activated environment.

## Prerequisites

1. **Sanas Remote SDK**: Obtain from [sanas.ai/developer-platform](https://www.sanas.ai/developer-platform) after onboarding.
2. **API Key**: Set your API key as an environment variable:
   ```bash
   export SANAS_API_KEY=sk_svgnv_...
   ```

## Quick Start

```python
import asyncio
from livekit.agents import AgentSession, AgentServer, JobContext, JobProcess, inference, room_io
from livekit.plugins import sanas

server = AgentServer()

def prewarm(proc: JobProcess) -> None:
    # Initialize Sanas NC during process warm-up — prevents GIL contention
    # on the first session from a new network.
    nc = sanas.NoiseCancellation(model_name=sanas.SanasModels.AGENTIC_VI_G_NC)
    nc.pre_initialize()
    proc.userdata["nc"] = nc

server.setup_fnc = prewarm

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    nc = ctx.proc.userdata["nc"]

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=nc,
            ),
        ),
    )

    # Release the processor when the room disconnects:
    nc.close()
```

**Audio Pipeline:** `Browser mic → Sanas NoiseCancellation → VAD → STT → LLM → TTS → Browser speaker`

## Configuration

### `NoiseCancellation` / `AudioFilter` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | str | `AGENTIC_VI_G_NC` | Noise cancellation model. See models below or set via `SANAS_MODEL_NAME` |
| `api_key` | str | env var | Sanas API key (`SANAS_API_KEY`) |
| `secure_media` | bool | `True` | Enable encrypted media transport |
| `use_pcm16` | bool | `False` | Use PCM16 encoding (has no effect at 24 kHz, always PCM16) |
| `sample_rate` | int | `16000` | SDK processing rate. Supported: 8000, 16000, 24000, 48000 Hz |

### Available Models

See full specifications at [developer.sanas.ai/Docs/Models/Overview](https://developer.sanas.ai/Docs/Models/Overview).

| Model | Use Case | Best For | `sample_rate` |
|---|---|---|---|
| `AGENTIC_VI_G_NC` | Human ↔ Machine | Voice agents, IVR, phone bots (default) | `16000` (default) |
| `AGENTIC_VI_GT_NC` | Human ↔ Machine | Telephony voice agents — 8kHz narrowband | `8000` |
| `AGENTIC_ST_NC` | Human ↔ Machine | Multi-speaker environments | `16000` (default) |
| `VI_G_NC3.0` | Human ↔ Human | Contact centers, conferencing | `24000` |

```python
# Using the enum (type-safe, IDE autocomplete)
nc = sanas.NoiseCancellation(model_name=sanas.SanasModels.AGENTIC_VI_G_NC)

# Or as a plain string for models not yet in the enum
nc = sanas.NoiseCancellation(model_name="AGENTIC_VI_GT_NC")
```

## Examples

| File | Description |
|---|---|
| [`examples/noise_cancellation_agent.py`](examples/noise_cancellation_agent.py) | Full voice agent with Sanas NC in the audio pipeline |
| [`examples/noise_cancellation_filter.py`](examples/noise_cancellation_filter.py) | Minimal standalone test — verifies SDK install and backend connectivity |

## Troubleshooting

### "Missing required Sanas config: SANAS_API_KEY / api_key"
Set the API key environment variable:
```bash
export SANAS_API_KEY=sk_svgnv_...
```

### "sanas_remote_sdk is not installed"
Download and install the SDK from the [Sanas Developer Platform](https://www.sanas.ai/developer-platform):
```bash
bash sanas_remote_sdk_<platform>/install.sh
```

### "Timeout waiting for Sanas processor READY"
The plugin connected to the Sanas backend but did not receive a READY signal within 10 seconds.
Most common causes:
- API key does not have the requested model enabled in the Sanas Developer Console
- Network or firewall blocking connectivity to the Sanas backend

### Agent does not respond to speech when NoiseCancellation is enabled
Verify the Sanas processor reached READY state in the logs:
```
Sanas SDK ready | model=AGENTIC_VI_G_NC rate=16000 Hz
```
If you see `FAILED` or `DISCONNECTED` in the logs, check API key entitlements and network connectivity to the Sanas backend.
