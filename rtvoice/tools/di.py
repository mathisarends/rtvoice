from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Self, final


@final
class _InjectMarker:
    def __repr__(self) -> str:
        return "Inject"


_INJECT_MARKER = _InjectMarker()


if TYPE_CHECKING:
    type Inject[T] = T
else:

    class Inject:
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, _INJECT_MARKER]


class ToolContext:
    def __init__(self, *dependencies: Any) -> None:
        self._dependencies: list[Any] = [dep for dep in dependencies if dep is not None]

    def provide(self, *dependencies: Any) -> Self:
        self._dependencies.extend(dep for dep in dependencies if dep is not None)
        return self

    def clear(self) -> Self:
        self._dependencies.clear()
        return self

    def without(self, *excluded: type) -> Self:
        self._dependencies = [
            dep for dep in self._dependencies if not isinstance(dep, excluded)
        ]
        return self

    def resolve[T](self, expected_type: type[T]) -> T | None:
        for dependency in self._dependencies:
            if isinstance(dependency, expected_type):
                return dependency
        return None

    def __len__(self) -> int:
        return len(self._dependencies)
