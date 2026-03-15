from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, overload

import httpx

from rtvoice.llm.messages import Message
from rtvoice.llm.views import ChatInvokeCompletion


class BaseChatModel(ABC):
    def __init__(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        response_format: dict | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        self._default_max_tokens = max_tokens
        self._default_temperature = temperature
        self._default_top_p = top_p
        self._default_frequency_penalty = frequency_penalty
        self._default_presence_penalty = presence_penalty
        self._default_stop = stop
        self._default_seed = seed
        self._default_response_format = response_format
        self._default_timeout = timeout
        self._default_max_retries = max_retries
        self._default_kwargs = kwargs

    def _merge_params(self, method_kwargs: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "max_tokens": self._default_max_tokens,
            "temperature": self._default_temperature,
            "top_p": self._default_top_p,
            "frequency_penalty": self._default_frequency_penalty,
            "presence_penalty": self._default_presence_penalty,
            "stop": self._default_stop,
            "seed": self._default_seed,
            "response_format": self._default_response_format,
        }

        params = {**self._default_kwargs}

        for key, default in defaults.items():
            value = method_kwargs.get(key, default)
            if value is not None:
                params[key] = value

        for key, value in method_kwargs.items():
            if key not in defaults and value is not None:
                params[key] = value

        return params

    @overload
    async def invoke[T](
        self, messages: list[Message], output_format: type[T], **kwargs: Any
    ) -> ChatInvokeCompletion[T]: ...

    @overload
    async def invoke(
        self, messages: list[Message], output_format: None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[str]: ...

    @abstractmethod
    async def invoke[T](
        self,
        messages: list[Message],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...
