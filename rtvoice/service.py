from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.tools import Tools
from rtvoice.tools.mcp.server import MCPServer
from rtvoice.views import AssistantVoice, RealtimeModel


class Agent(LoggingMixin):
    def __init__(
        self,
        instructions: str = "",
        model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        speech_speed: float = 1.0,
        tools: Tools | None = None,
        mcp_servers: list[MCPServer] | None = None,
    ):
        self._instructions = instructions
        self._model = model
        self._voice = voice
        self._speech_speed = self._clip_speech_speed(speech_speed)
        self._tools = tools or Tools()
        self._mcp_servers = mcp_servers or []

    async def __aenter__(self) -> "Agent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def _clip_speech_speed(self, speed: float) -> float:
        clipped = max(0.5, min(speed, 1.5))

        if speed != clipped:
            self.logger.warning(
                f"Speech speed {speed} is out of range [0.5, 1.5], clipping to {clipped}"
            )

        return clipped
