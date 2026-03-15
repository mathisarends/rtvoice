import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

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
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.result_instruction = result_instruction

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tool):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class RealtimeTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        schema: FunctionParameters,
        result_instruction: str | None = None,
        holding_instruction: str | None = None,
    ):
        super().__init__(
            name=name,
            description=description,
            function=function,
            schema=schema,
            result_instruction=result_instruction,
        )
        self.holding_instruction = holding_instruction


class SubAgentTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        schema: FunctionParameters,
        result_instruction: str | None = None,
        status: str | None = None,
        suppress_response: bool = False,
    ):
        super().__init__(
            name=name,
            description=description,
            function=function,
            schema=schema,
            result_instruction=result_instruction,
        )
        self.status = status
        self.suppress_response = suppress_response

    def format_status(self, args: BaseModel | dict[str, Any]) -> str | None:
        if self.status is None:
            return None

        args_dict = (
            args.model_dump(exclude_none=True) if isinstance(args, BaseModel) else args
        )

        try:
            return self.status.format(**args_dict)
        except KeyError:
            return self.status
