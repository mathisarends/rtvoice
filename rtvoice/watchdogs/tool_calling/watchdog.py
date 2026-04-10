import json
import logging

from rtvoice.events import EventBus
from rtvoice.realtime.schemas import FunctionCallItem
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.tools import Tools
from rtvoice.watchdogs.tool_calling.helpers import (
    send_function_call_output,
    send_response_event,
    serialize_tool_result,
)

logger = logging.getLogger(__name__)


class ToolCallingWatchdog:
    def __init__(
        self,
        event_bus: EventBus,
        tools: Tools,
        websocket: RealtimeWebSocket,
        subagent_tool_names: set[str] | None = None,
    ) -> None:
        self._tools = tools
        self._websocket = websocket
        self._subagent_tool_names: set[str] = subagent_tool_names or set()

        event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        logger.debug("ToolCallingWatchdog initialized")

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        if self._is_subagent_tool(event.name):
            return

        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        logger.info(
            "Tool call: '%s' [args=%s]",
            event.name,
            json.dumps(event.arguments or {}, ensure_ascii=False),
        )

        result = await self._tools.execute(event.name, event.arguments or {})
        serialized = serialize_tool_result(result)
        logger.info("Tool result: '%s' [result=%s]", event.name, serialized)
        await send_function_call_output(self._websocket, event.call_id, serialized)
        await send_response_event(self._websocket, tool)

    def _is_subagent_tool(self, tool_name: str) -> bool:
        return tool_name in self._subagent_tool_names
