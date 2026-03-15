from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Protocol for tool implementations.

    Any class implementing this protocol can be used as a tool.
    Allows for flexible tool definitions beyond Pydantic models.
    """

    @property
    def name(self) -> str: ...

    def to_openai_schema(self) -> dict[str, Any]: ...

    def parse_arguments(self, arguments: str) -> Any: ...
