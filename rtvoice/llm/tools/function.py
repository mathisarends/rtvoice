import inspect
import json
from collections.abc import Callable
from typing import Any, get_type_hints, overload

type _AnyCallable = Callable[..., Any]


class FunctionTool:
    """Tool implementation for Python functions.

    Extracts parameter information from function signature and type hints.

    Example:
        def search_web(query: str, max_results: int = 10) -> str:
            '''Search the web for information'''
            return f"Searching for: {query}"

        tool = FunctionTool(search_web)
        # Or use decorator: @tool
    """

    def __init__(
        self,
        fn: Callable,
        name: str | None = None,
        description: str | None = None,
    ):
        self._fn = fn
        self._name = name or fn.__name__
        self._description = description or (fn.__doc__.strip() if fn.__doc__ else "")

    @property
    def name(self) -> str:
        return self._name

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._description,
                "parameters": self._extract_parameters(),
            },
        }

    def _extract_parameters(self) -> dict[str, Any]:
        sig = inspect.signature(self._fn)

        try:
            hints = get_type_hints(self._fn)
        except Exception:
            hints = {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            type_hint = hints.get(param_name, str)
            properties[param_name] = self._type_to_json_schema(type_hint, param)

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _type_to_json_schema(
        self, type_hint: Any, param: inspect.Parameter
    ) -> dict[str, Any]:
        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }

        origin = getattr(type_hint, "__origin__", None)
        if origin is not None:
            if origin is list:
                return {"type": "array"}
            elif origin is dict:
                return {"type": "object"}

        schema = type_mapping.get(type_hint, {"type": "string"})

        if hasattr(param, "annotation") and hasattr(param.annotation, "__metadata__"):
            metadata = getattr(param.annotation, "__metadata__", ())
            for item in metadata:
                if hasattr(item, "description"):
                    schema["description"] = item.description
                    break

        return schema

    def parse_arguments(self, arguments: str) -> dict[str, Any]:
        return json.loads(arguments)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


@overload
def tool(fn: _AnyCallable) -> FunctionTool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[_AnyCallable], FunctionTool]: ...


def tool(
    fn: _AnyCallable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool | Callable[[_AnyCallable], FunctionTool]:
    def decorator(func: _AnyCallable) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)

    if fn is None:
        return decorator
    return decorator(fn)
