import logging
from unittest.mock import patch

import pytest

from rtvoice.shared.decorators import timed


class TestTimedExecution:
    @pytest.mark.asyncio
    async def test_returns_function_result(self) -> None:
        @timed()
        async def func() -> str:
            return "result"

        assert await func() == "result"

    @pytest.mark.asyncio
    async def test_passes_args_to_function(self) -> None:
        @timed()
        async def add(a: int, b: int) -> int:
            return a + b

        assert await add(2, 3) == 5

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_function(self) -> None:
        @timed()
        async def greet(name: str = "World") -> str:
            return f"Hello {name}"

        assert await greet(name="Mathis") == "Hello Mathis"

    @pytest.mark.asyncio
    async def test_preserves_function_name(self) -> None:
        @timed()
        async def my_function() -> None: ...

        assert my_function.__name__ == "my_function"

    @pytest.mark.asyncio
    async def test_preserves_function_module(self) -> None:
        @timed()
        async def my_function() -> None: ...

        assert my_function.__module__ == __name__


class TestTimedLogging:
    @pytest.mark.asyncio
    async def test_logs_when_duration_exceeds_threshold(self, caplog) -> None:
        @timed(min_duration_to_log=0.0)
        async def func() -> None: ...

        with caplog.at_level(logging.DEBUG):
            await func()

        assert any("func" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_does_not_log_when_duration_below_threshold(self, caplog) -> None:
        @timed(min_duration_to_log=9999.0)
        async def func() -> None: ...

        with caplog.at_level(logging.DEBUG):
            await func()

        assert not caplog.records

    @pytest.mark.asyncio
    async def test_uses_additional_text_as_label(self, caplog) -> None:
        @timed(additional_text="my-label", min_duration_to_log=0.0)
        async def func() -> None: ...

        with caplog.at_level(logging.DEBUG):
            await func()

        assert any("my-label" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_falls_back_to_function_name_when_no_additional_text(
        self, caplog
    ) -> None:
        @timed(min_duration_to_log=0.0)
        async def my_named_func() -> None: ...

        with caplog.at_level(logging.DEBUG):
            await my_named_func()

        assert any("my_named_func" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_additional_text_strips_dashes_from_label(self, caplog) -> None:
        @timed(additional_text="-stripped-", min_duration_to_log=0.0)
        async def func() -> None: ...

        with caplog.at_level(logging.DEBUG):
            await func()

        assert any("stripped" in record.message for record in caplog.records)
        assert not any("--" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_logs_to_function_module_logger(self, caplog) -> None:
        @timed(min_duration_to_log=0.0)
        async def func() -> None: ...

        with caplog.at_level(logging.DEBUG, logger=__name__):
            await func()

        assert any(record.name == __name__ for record in caplog.records)

    @pytest.mark.asyncio
    async def test_log_message_contains_duration(self, caplog) -> None:
        @timed(min_duration_to_log=0.0)
        async def func() -> None: ...

        with caplog.at_level(logging.DEBUG):
            await func()

        assert any("s" in record.message for record in caplog.records)


class TestTimedDefaults:
    @pytest.mark.asyncio
    async def test_default_threshold_does_not_log_fast_function(self, caplog) -> None:
        @timed()
        async def fast() -> None: ...

        with caplog.at_level(logging.DEBUG):
            await fast()

        assert not caplog.records

    @pytest.mark.asyncio
    async def test_default_additional_text_is_empty(self, caplog) -> None:
        @timed()
        async def my_func() -> None: ...

        with (
            caplog.at_level(logging.DEBUG, logger=__name__),
            patch("time.perf_counter", side_effect=[0.0, 1.0]),
        ):
            await my_func()

        assert any("my_func" in record.message for record in caplog.records)
