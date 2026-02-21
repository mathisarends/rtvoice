import inspect
from collections.abc import Callable
from typing import Any

from rtvoice.realtime.schemas import FunctionParameters, FunctionTool


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        schema: FunctionParameters,
        result_instruction: str | None = None,
        pending_message: str | None = None,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.result_instruction = result_instruction
        self.pending_message = pending_message

    async def execute(self, arguments: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**arguments)
        else:
            return self.function(**arguments)

    def to_pydantic(self) -> FunctionTool:
        return FunctionTool(
            name=self.name,
            description=self.description,
            parameters=self.schema,
        )
