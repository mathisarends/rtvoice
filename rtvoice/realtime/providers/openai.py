import os

from rtvoice.realtime.providers.base import RealtimeProvider


class OpenAIProvider(RealtimeProvider):
    """Realtime provider for the OpenAI API.

    This is the default provider used by `RealtimeAgent` when no `provider`
    argument is supplied.

    The API key is resolved in this order:

    1. The `api_key` constructor argument.
    2. The ``OPENAI_API_KEY`` environment variable.

    Example:
        ```python
        from rtvoice import RealtimeAgent, OpenAIProvider

        agent = RealtimeAgent(
            instructions="You are a helpful assistant.",
            provider=OpenAIProvider(api_key="sk-..."),
        )
        ```
    """

    _BASE_URL = "wss://api.openai.com/v1/realtime"

    def __init__(
        self,
        api_key: str | None = None,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    def build_url(self, model: str) -> str:
        return f"{self._BASE_URL}?model={model}"

    def build_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
