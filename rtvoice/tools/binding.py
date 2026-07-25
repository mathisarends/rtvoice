from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtvoice.tools.di import ToolContext


class ToolDescription:
    def __init__(
        self,
        dependency: type,
        render: Callable[[object], str],
        default: str,
    ) -> None:
        self._dependency = dependency
        self._render = render
        self._default = default

    def resolve(self, context: ToolContext | None) -> str:
        dependency = context.resolve(self._dependency) if context is not None else None
        if dependency is None:
            return self._default
        return self._render(dependency)


class ToolAvailability:
    def __init__(self, predicate: Callable[[ToolContext | None], bool]) -> None:
        self._predicate = predicate

    def __call__(self, context: ToolContext | None) -> bool:
        return self._predicate(context)


def described[T](
    dependency: type[T],
    *,
    render: Callable[[T], str],
    default: str,
) -> ToolDescription:
    return ToolDescription(dependency, render, default)  # type: ignore[arg-type]


def provided(dependency: type) -> ToolAvailability:
    def _predicate(context: ToolContext | None) -> bool:
        return context is not None and context.resolve(dependency) is not None

    return ToolAvailability(_predicate)


def requires[T](
    dependency: type[T],
    *,
    predicate: Callable[[T], bool],
) -> ToolAvailability:
    def _predicate(context: ToolContext | None) -> bool:
        if context is None:
            return False
        resolved = context.resolve(dependency)
        return resolved is not None and predicate(resolved)

    return ToolAvailability(_predicate)
