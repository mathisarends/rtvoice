from unittest.mock import patch

import pytest

from rtvoice.realtime.providers.openai import OpenAIProvider


class TestInit:
    def test_uses_explicit_api_key(self) -> None:
        provider = OpenAIProvider(api_key="sk-explicit")

        assert provider._api_key == "sk-explicit"

    def test_falls_back_to_env_var(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
            provider = OpenAIProvider()

        assert provider._api_key == "sk-from-env"

    def test_explicit_key_takes_precedence_over_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
            provider = OpenAIProvider(api_key="sk-explicit")

        assert provider._api_key == "sk-explicit"

    def test_raises_when_api_key_missing(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
        ):
            OpenAIProvider()


class TestBuildUrl:
    def test_includes_model_in_url(self) -> None:
        provider = OpenAIProvider(api_key="k")

        url = provider.build_url("gpt-4o-realtime-preview")

        assert "gpt-4o-realtime-preview" in url

    def test_uses_wss_scheme(self) -> None:
        provider = OpenAIProvider(api_key="k")

        assert provider.build_url("any").startswith("wss://")

    def test_different_models_produce_different_urls(self) -> None:
        provider = OpenAIProvider(api_key="k")

        assert provider.build_url("gpt-4o") != provider.build_url("gpt-4o-mini")


class TestBuildHeaders:
    def test_returns_bearer_authorization_header(self) -> None:
        provider = OpenAIProvider(api_key="sk-secret")

        headers = provider.build_headers()

        assert headers == {"Authorization": "Bearer sk-secret"}
