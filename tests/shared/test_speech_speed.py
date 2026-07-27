import logging

import pytest
from pydantic import BaseModel

from rtvoice.shared.speech_speed import DEFAULT_SPEECH_SPEED, SpeechSpeed


class TestClamping:
    def test_value_within_range_is_unchanged(self) -> None:
        assert SpeechSpeed(1.2) == 1.2

    def test_value_below_minimum_is_clamped(self) -> None:
        assert SpeechSpeed(0.1) == 0.25

    def test_value_above_maximum_is_clamped(self) -> None:
        assert SpeechSpeed(2.0) == 1.5

    def test_exact_bounds_are_not_clamped(self) -> None:
        assert SpeechSpeed(0.25) == 0.25
        assert SpeechSpeed(1.5) == 1.5

    def test_out_of_range_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            SpeechSpeed(3.0)
        assert any("out of range" in r.message for r in caplog.records)

    def test_in_range_does_not_log_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            SpeechSpeed(1.2)
        assert not any("out of range" in r.message for r in caplog.records)

    def test_default_is_one(self) -> None:
        assert DEFAULT_SPEECH_SPEED == 1.0


class TestFloatBehaviour:
    def test_arithmetic_works_like_a_float(self) -> None:
        assert 1000 * SpeechSpeed(1.5) == 1500.0

    def test_is_a_float(self) -> None:
        assert isinstance(SpeechSpeed(1.0), float)


class TestPydanticIntegration:
    def test_model_field_clamps_on_construction(self) -> None:
        class Model(BaseModel):
            speed: SpeechSpeed

        assert Model(speed=3.0).speed == 1.5

    def test_model_field_serializes_as_number(self) -> None:
        class Model(BaseModel):
            speed: SpeechSpeed

        assert Model(speed=0.5).model_dump() == {"speed": 0.5}
