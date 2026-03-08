import inspect
from collections.abc import Callable
from typing import Any

from rtvoice.realtime.schemas import FunctionParameters, FunctionTool
from rtvoice.tools.views import VoidResult


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        schema: FunctionParameters,
        result_instruction: str | None = None,
        holding_instruction: str | None = None,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.result_instruction = result_instruction
        self.holding_instruction = holding_instruction

    async def execute(self, arguments: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self.function):
            result = await self.function(**arguments)
        else:
            result = self.function(**arguments)

        return result if result is not None else VoidResult()

    def to_pydantic(self) -> FunctionTool:
        return FunctionTool(
            name=self.name,
            description=self.description,
            parameters=self.schema,
        )
