import os

from rtvoice.realtime.port import RealtimeProvider


class AzureOpenAIProvider(RealtimeProvider):
    _DEFAULT_API_VERSION = "2025-04-01-preview"

    def __init__(
        self,
        resource: str | None = None,
        deployment: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
    ):
        self._resource = resource or os.environ.get("AZURE_OPENAI_RESOURCE")
        self._deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._api_version = api_version or self._DEFAULT_API_VERSION

        if not self._resource:
            raise RuntimeError(
                "Azure resource name is required (AZURE_OPENAI_RESOURCE)."
            )
        if not self._deployment:
            raise RuntimeError(
                "Azure deployment name is required (AZURE_OPENAI_DEPLOYMENT)."
            )
        if not self._api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY environment variable is not set.")

    def build_url(self, model: str) -> str:
        # `model` is intentionally ignored — Azure routes via the deployment name,
        # which is bound to a specific model at deployment time.
        return (
            f"wss://{self._resource}.openai.azure.com/openai/realtime"
            f"?api-version={self._api_version}"
            f"&deployment={self._deployment}"
        )

    def build_headers(self) -> dict[str, str]:
        return {"api-key": self._api_key}
