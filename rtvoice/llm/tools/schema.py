import json
from typing import Any


class RawSchemaTool:
    """Tool implementation for raw JSON schemas.

    Useful when you want full control over the schema or for legacy tools.

    Example:
        tool = RawSchemaTool(
            name="search_web",
            description="Search the web for information",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        )
    """

    def __init__(
        self,
        name: str,
        schema: dict[str, Any],
        description: str = "",
    ):
        self._name = name
        self._schema = schema
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._description,
                "parameters": self._schema,
            },
        }

    def parse_arguments(self, arguments: str) -> dict[str, Any]:
        return json.loads(arguments)
