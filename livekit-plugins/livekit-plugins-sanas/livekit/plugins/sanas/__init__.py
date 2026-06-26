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

"""Sanas plugin for LiveKit Agents.

Provides Noise Cancellation via the Sanas Remote SDK.

Typical usage::

    import livekit.plugins.sanas as sanas

    sanas_nc = sanas.NoiseCancellation()                                        # default: AGENTIC_VI_G_NC
    sanas_nc = sanas.NoiseCancellation(model_name="AGENTIC_VI_GT_NC")          # telephony variant
"""

from livekit.agents import Plugin

from .log import logger
from .models import SanasModels
from .processor import AudioFilter, NoiseCancellation
from .version import __version__

__all__ = [
    # Preferred API
    "NoiseCancellation",
    # Base class (for custom models / advanced use)
    "AudioFilter",
    # Model constants
    "SanasModels",
    "__version__",
]


class SanasPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(SanasPlugin())

# Hide private symbols from documentation
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]
__pdoc__ = {}
for n in NOT_IN_ALL:
    __pdoc__[n] = False
