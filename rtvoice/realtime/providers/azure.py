import os
from typing import Annotated

from typing_extensions import Doc

from rtvoice.realtime.providers.base import RealtimeProvider


class AzureOpenAIProvider(RealtimeProvider):
    """Realtime provider for Azure OpenAI.

    Constructs the Azure-specific WebSocket endpoint and authenticates
    using an ``api-key`` header instead of a Bearer token.

    All constructor arguments fall back to environment variables when omitted:

    | Argument     | Environment variable         |
    |--------------|------------------------------|
    | `resource`   | ``AZURE_OPENAI_RESOURCE``    |
    | `deployment` | ``AZURE_OPENAI_DEPLOYMENT``  |
    | `api_key`    | ``AZURE_OPENAI_API_KEY``     |

    Example:
        ```python
        from rtvoice import RealtimeAgent, AzureOpenAIProvider

        agent = RealtimeAgent(
            instructions="You are a helpful assistant.",
            provider=AzureOpenAIProvider(
                resource="my-resource",
                deployment="gpt-4o-realtime-preview",
            ),
        )
        ```
    """

    _DEFAULT_API_VERSION = "2025-04-01-preview"

    def __init__(
        self,
        resource: Annotated[
            str | None,
            Doc(
                "Azure OpenAI resource name (the subdomain of ``openai.azure.com``). "
                "Falls back to the ``AZURE_OPENAI_RESOURCE`` environment variable."
            ),
        ] = None,
        deployment: Annotated[
            str | None,
            Doc(
                "Deployment name that maps to a specific model. "
                "Falls back to the ``AZURE_OPENAI_DEPLOYMENT`` environment variable."
            ),
        ] = None,
        api_key: Annotated[
            str | None,
            Doc(
                "Azure OpenAI API key. "
                "Falls back to the ``AZURE_OPENAI_API_KEY`` environment variable."
            ),
        ] = None,
        api_version: Annotated[
            str | None,
            Doc("API version string. Defaults to ``2025-04-01-preview``."),
        ] = None,
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
