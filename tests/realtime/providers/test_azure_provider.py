from unittest.mock import patch

import pytest

from rtvoice.realtime.providers.azure import AzureOpenAIProvider


class TestInit:
    def test_uses_explicit_arguments(self) -> None:
        provider = AzureOpenAIProvider(
            resource="my-resource",
            deployment="gpt-4o",
            api_key="my-key",
        )

        assert provider._resource == "my-resource"
        assert provider._deployment == "gpt-4o"
        assert provider._api_key == "my-key"

    def test_falls_back_to_env_vars(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_RESOURCE": "env-resource",
                "AZURE_OPENAI_DEPLOYMENT": "env-deployment",
                "AZURE_OPENAI_API_KEY": "env-key",
            },
        ):
            provider = AzureOpenAIProvider()

        assert provider._resource == "env-resource"
        assert provider._deployment == "env-deployment"
        assert provider._api_key == "env-key"

    def test_uses_default_api_version(self) -> None:
        provider = AzureOpenAIProvider(resource="r", deployment="d", api_key="k")

        assert provider._api_version == AzureOpenAIProvider._DEFAULT_API_VERSION

    def test_uses_explicit_api_version(self) -> None:
        provider = AzureOpenAIProvider(
            resource="r", deployment="d", api_key="k", api_version="2024-01-01"
        )

        assert provider._api_version == "2024-01-01"

    def test_raises_when_resource_missing(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="AZURE_OPENAI_RESOURCE"),
        ):
            AzureOpenAIProvider(deployment="d", api_key="k")

    def test_raises_when_deployment_missing(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="AZURE_OPENAI_DEPLOYMENT"),
        ):
            AzureOpenAIProvider(resource="r", api_key="k")

    def test_raises_when_api_key_missing(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"),
        ):
            AzureOpenAIProvider(resource="r", deployment="d")


class TestBuildUrl:
    def test_includes_resource_and_deployment(self) -> None:
        provider = AzureOpenAIProvider(
            resource="my-resource", deployment="gpt-4o", api_key="k"
        )

        url = provider.build_url("ignored-model")

        assert "my-resource.openai.azure.com" in url
        assert "deployment=gpt-4o" in url

    def test_ignores_model_argument(self) -> None:
        provider = AzureOpenAIProvider(resource="r", deployment="d", api_key="k")

        url_a = provider.build_url("gpt-4o")
        url_b = provider.build_url("gpt-4o-mini")

        assert url_a == url_b

    def test_includes_api_version(self) -> None:
        provider = AzureOpenAIProvider(
            resource="r", deployment="d", api_key="k", api_version="2024-05-01"
        )

        url = provider.build_url("any")

        assert "api-version=2024-05-01" in url

    def test_uses_wss_scheme(self) -> None:
        provider = AzureOpenAIProvider(resource="r", deployment="d", api_key="k")

        assert provider.build_url("any").startswith("wss://")


class TestBuildHeaders:
    def test_returns_api_key_header(self) -> None:
        provider = AzureOpenAIProvider(
            resource="r", deployment="d", api_key="my-secret"
        )

        headers = provider.build_headers()

        assert headers == {"api-key": "my-secret"}
