from .port import RealtimeProvider
from .providers import AzureOpenAIProvider, OpenAIProvider

__all__ = [
    "AzureOpenAIProvider",
    "OpenAIProvider",
    "RealtimeProvider",
    "RealtimeSession",
]


def __getattr__(name: str):
    if name == "RealtimeSession":
        from .session import RealtimeSession

        return RealtimeSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
