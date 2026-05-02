import os

from rtvoice.realtime.port import RealtimeProvider


class OpenAIProvider(RealtimeProvider):
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
