from __future__ import annotations

import json
import logging

from rtvoice.events import EventBus
from rtvoice.realtime.schemas import FunctionCallItem
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.tools import Tools
from rtvoice.watchdogs.tool_calling.helpers import ToolCallWebSocketHelper

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
        self._ws = ToolCallWebSocketHelper(websocket)
        self._subagent_tool_names: set[str] = subagent_tool_names or set()

        event_bus.subscribe(FunctionCallItem, self._handle_tool_call)
        logger.debug("ToolCallingWatchdog initialized")

    def _is_supervisor_tool(self, tool_name: str) -> bool:
        return tool_name in self._subagent_tool_names

    async def _handle_tool_call(self, event: FunctionCallItem) -> None:
        if self._is_supervisor_tool(event.name):
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
        logger.info(
            "Tool result: '%s' [result=%s]", event.name, self._ws.serialize(result)
        )
        await self._ws.send_function_call_output(
            event.call_id, self._ws.serialize(result)
        )
        await self._ws.send_response_event(tool)
