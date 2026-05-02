from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from rtvoice.llm.views import ChatInvokeUsage
from rtvoice.token.views import (
    TokenUsageBreakdown,
    TokenUsageCost,
    TokenUsageModelSummary,
    TokenUsageRecord,
    TokenUsageSummary,
)

TOKENS_PER_MILLION = 1_000_000
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class TokenRate:
    input_per_million: float | None = None
    cached_input_per_million: float | None = None
    output_per_million: float | None = None


@dataclass(frozen=True)
class ModelPricing:
    text: TokenRate | None = None
    audio: TokenRate | None = None
    image: TokenRate | None = None
    transcription: TokenRate | None = None
    duration_per_minute: float | None = None


class TokenPricingCatalog:
    def __init__(self, prices: dict[str, ModelPricing] | None = None) -> None:
        self._prices = prices or _default_prices()

    def price_for(self, model: str) -> ModelPricing | None:
        return self._prices.get(model)


class TokenTracker:
    def __init__(self, pricing_catalog: TokenPricingCatalog | None = None) -> None:
        self._pricing_catalog = pricing_catalog or TokenPricingCatalog()
        self._records: list[TokenUsageRecord] = []

    def track_chat_usage(
        self, *, model: str, usage: ChatInvokeUsage | None, source: str = "chat"
    ) -> None:
        if usage is None:
            return

        cached_tokens = usage.prompt_cached_tokens or 0
        input_tokens = max(usage.prompt_tokens - cached_tokens, 0)
        breakdown = TokenUsageBreakdown(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            input_text_tokens=input_tokens,
            cached_input_text_tokens=cached_tokens,
            output_text_tokens=usage.completion_tokens,
        )
        self._append_record(source=source, model=model, usage=breakdown)

    def track_realtime_response_usage(
        self, *, model: str, usage: BaseModel | None, source: str = "realtime"
    ) -> None:
        if usage is None:
            return

        breakdown = self._breakdown_realtime_usage(usage)
        self._append_record(source=source, model=model, usage=breakdown)

    def track_transcription_usage(
        self, *, model: str, usage: BaseModel | None, source: str = "transcription"
    ) -> None:
        if usage is None:
            return

        breakdown = self._breakdown_transcription_usage(usage)
        self._append_record(source=source, model=model, usage=breakdown)

    def summary(self) -> TokenUsageSummary:
        records = list(self._records)
        total_usage = TokenUsageBreakdown()
        total_cost = TokenUsageCost()
        by_model: dict[str, TokenUsageModelSummary] = {}

        for record in records:
            self._add_usage(total_usage, record.usage)
            self._add_cost(total_cost, record.cost)

            model_summary = by_model.setdefault(
                record.model,
                TokenUsageModelSummary(model=record.model),
            )
            self._add_usage(model_summary.usage, record.usage)
            self._add_cost(model_summary.cost, record.cost)
            model_summary.price_available = (
                model_summary.price_available and record.price_available
            )

        return TokenUsageSummary(
            usage=total_usage,
            cost=total_cost,
            by_model=list(by_model.values()),
            records=records,
            has_unpriced_usage=any(not record.price_available for record in records),
        )

    def _append_record(
        self, *, source: str, model: str, usage: TokenUsageBreakdown
    ) -> None:
        cost, price_available = self._calculate_cost(model, usage)
        self._records.append(
            TokenUsageRecord(
                source=source,
                model=model,
                usage=usage,
                cost=cost,
                price_available=price_available,
            )
        )

    def _breakdown_realtime_usage(self, usage: BaseModel) -> TokenUsageBreakdown:
        input_details = getattr(usage, "input_token_details", None)
        output_details = getattr(usage, "output_token_details", None)
        cached_details = getattr(input_details, "cached_tokens_details", None)

        input_text_tokens = self._int_field(input_details, "text_tokens")
        input_audio_tokens = self._int_field(input_details, "audio_tokens")
        input_image_tokens = self._int_field(input_details, "image_tokens")

        cached_text_tokens = self._int_field(cached_details, "text_tokens")
        cached_audio_tokens = self._int_field(cached_details, "audio_tokens")
        cached_image_tokens = self._int_field(cached_details, "image_tokens")
        cached_input_tokens = self._int_field(input_details, "cached_tokens")

        if cached_input_tokens and not any(
            [cached_text_tokens, cached_audio_tokens, cached_image_tokens]
        ):
            cached_text_tokens = cached_input_tokens
        else:
            cached_input_tokens = (
                cached_text_tokens + cached_audio_tokens + cached_image_tokens
            )

        output_text_tokens = self._int_field(output_details, "text_tokens")
        output_audio_tokens = self._int_field(output_details, "audio_tokens")

        input_tokens = self._int_field(usage, "input_tokens")
        output_tokens = self._int_field(usage, "output_tokens")

        if not any([input_text_tokens, input_audio_tokens, input_image_tokens]):
            input_text_tokens = max(input_tokens - cached_input_tokens, 0)
        if not any([output_text_tokens, output_audio_tokens]):
            output_text_tokens = output_tokens

        uncached_text_tokens = max(input_text_tokens - cached_text_tokens, 0)
        uncached_audio_tokens = max(input_audio_tokens - cached_audio_tokens, 0)
        uncached_image_tokens = max(input_image_tokens - cached_image_tokens, 0)

        return TokenUsageBreakdown(
            input_tokens=input_tokens - cached_input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            total_tokens=self._int_field(usage, "total_tokens"),
            input_text_tokens=uncached_text_tokens,
            cached_input_text_tokens=cached_text_tokens,
            output_text_tokens=output_text_tokens,
            input_audio_tokens=uncached_audio_tokens,
            cached_input_audio_tokens=cached_audio_tokens,
            output_audio_tokens=output_audio_tokens,
            input_image_tokens=uncached_image_tokens,
            cached_input_image_tokens=cached_image_tokens,
        )

    def _breakdown_transcription_usage(self, usage: BaseModel) -> TokenUsageBreakdown:
        if getattr(usage, "type", None) == "duration":
            return TokenUsageBreakdown(duration_seconds=float(usage.seconds))

        input_details = getattr(usage, "input_token_details", None)
        input_tokens = self._int_field(usage, "input_tokens")
        output_tokens = self._int_field(usage, "output_tokens")
        return TokenUsageBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=self._int_field(usage, "total_tokens"),
            input_audio_tokens=self._int_field(input_details, "audio_tokens"),
            input_text_tokens=self._int_field(input_details, "text_tokens"),
            output_text_tokens=output_tokens,
        )

    def _calculate_cost(
        self, model: str, usage: TokenUsageBreakdown
    ) -> tuple[TokenUsageCost, bool]:
        pricing = self._pricing_catalog.price_for(model)
        if pricing is None:
            return TokenUsageCost(), self._is_zero_usage(usage)

        input_usd = 0.0
        cached_input_usd = 0.0
        output_usd = 0.0

        if pricing.transcription:
            input_usd += self._token_cost(
                usage.input_tokens,
                pricing.transcription.input_per_million,
            )
            output_usd += self._token_cost(
                usage.output_tokens,
                pricing.transcription.output_per_million,
            )
        else:
            input_usd += self._modality_input_cost(
                usage.input_text_tokens, pricing.text
            )
            input_usd += self._modality_input_cost(
                usage.input_audio_tokens, pricing.audio
            )
            input_usd += self._modality_input_cost(
                usage.input_image_tokens, pricing.image
            )
            cached_input_usd += self._modality_cached_cost(
                usage.cached_input_text_tokens, pricing.text
            )
            cached_input_usd += self._modality_cached_cost(
                usage.cached_input_audio_tokens, pricing.audio
            )
            cached_input_usd += self._modality_cached_cost(
                usage.cached_input_image_tokens, pricing.image
            )
            output_usd += self._modality_output_cost(
                usage.output_text_tokens, pricing.text
            )
            output_usd += self._modality_output_cost(
                usage.output_audio_tokens, pricing.audio
            )

        duration_usd = self._duration_cost(usage.duration_seconds, pricing)
        total_usd = input_usd + cached_input_usd + output_usd + duration_usd
        return (
            TokenUsageCost(
                input_usd=input_usd,
                cached_input_usd=cached_input_usd,
                output_usd=output_usd,
                duration_usd=duration_usd,
                total_usd=total_usd,
            ),
            self._has_price_for_usage(pricing, usage),
        )

    def _has_price_for_usage(
        self, pricing: ModelPricing, usage: TokenUsageBreakdown
    ) -> bool:
        if pricing.transcription:
            return all(
                [
                    not usage.input_tokens
                    or pricing.transcription.input_per_million is not None,
                    not usage.output_tokens
                    or pricing.transcription.output_per_million is not None,
                    not usage.duration_seconds
                    or pricing.duration_per_minute is not None,
                ]
            )

        return all(
            [
                self._has_input_price(usage.input_text_tokens, pricing.text),
                self._has_cached_price(usage.cached_input_text_tokens, pricing.text),
                self._has_output_price(usage.output_text_tokens, pricing.text),
                self._has_input_price(usage.input_audio_tokens, pricing.audio),
                self._has_cached_price(usage.cached_input_audio_tokens, pricing.audio),
                self._has_output_price(usage.output_audio_tokens, pricing.audio),
                self._has_input_price(usage.input_image_tokens, pricing.image),
                self._has_cached_price(usage.cached_input_image_tokens, pricing.image),
                not usage.duration_seconds or pricing.duration_per_minute is not None,
            ]
        )

    def _has_input_price(self, tokens: int, rate: TokenRate | None) -> bool:
        return not tokens or (rate is not None and rate.input_per_million is not None)

    def _has_cached_price(self, tokens: int, rate: TokenRate | None) -> bool:
        return not tokens or (
            rate is not None and rate.cached_input_per_million is not None
        )

    def _has_output_price(self, tokens: int, rate: TokenRate | None) -> bool:
        return not tokens or (rate is not None and rate.output_per_million is not None)

    def _modality_input_cost(self, tokens: int, rate: TokenRate | None) -> float:
        return self._token_cost(tokens, rate.input_per_million if rate else None)

    def _modality_cached_cost(self, tokens: int, rate: TokenRate | None) -> float:
        return self._token_cost(tokens, rate.cached_input_per_million if rate else None)

    def _modality_output_cost(self, tokens: int, rate: TokenRate | None) -> float:
        return self._token_cost(tokens, rate.output_per_million if rate else None)

    def _token_cost(self, tokens: int, price_per_million: float | None) -> float:
        if not tokens or price_per_million is None:
            return 0.0
        return tokens * price_per_million / TOKENS_PER_MILLION

    def _duration_cost(self, seconds: float, pricing: ModelPricing) -> float:
        if not seconds or pricing.duration_per_minute is None:
            return 0.0
        return seconds * pricing.duration_per_minute / SECONDS_PER_MINUTE

    def _is_zero_usage(self, usage: TokenUsageBreakdown) -> bool:
        return not any(
            [
                usage.input_tokens,
                usage.cached_input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                usage.duration_seconds,
            ]
        )

    def _int_field(self, value: Any, field_name: str) -> int:
        if value is None:
            return 0
        field_value = getattr(value, field_name, 0)
        return int(field_value or 0)

    def _add_usage(
        self, target: TokenUsageBreakdown, addition: TokenUsageBreakdown
    ) -> None:
        for field_name in type(target).model_fields:
            value = getattr(target, field_name) + getattr(addition, field_name)
            setattr(target, field_name, value)

    def _add_cost(self, target: TokenUsageCost, addition: TokenUsageCost) -> None:
        for field_name in type(target).model_fields:
            value = getattr(target, field_name) + getattr(addition, field_name)
            setattr(target, field_name, value)


def _default_prices() -> dict[str, ModelPricing]:
    gpt_4o = ModelPricing(text=TokenRate(2.50, 1.25, 10.00))
    gpt_4o_mini = ModelPricing(text=TokenRate(0.15, 0.075, 0.60))
    gpt_realtime = ModelPricing(
        text=TokenRate(4.00, 0.40, 16.00),
        audio=TokenRate(32.00, 0.40, 64.00),
        image=TokenRate(5.00, 0.50, None),
    )

    return {
        "gpt-4o": gpt_4o,
        "gpt-4o-2024-05-13": gpt_4o,
        "gpt-4o-2024-08-06": gpt_4o,
        "gpt-4o-2024-11-20": gpt_4o,
        "gpt-4o-mini": gpt_4o_mini,
        "gpt-4o-mini-2024-07-18": gpt_4o_mini,
        "gpt-5.5": ModelPricing(text=TokenRate(5.00, 0.50, 30.00)),
        "gpt-5.4": ModelPricing(text=TokenRate(2.50, 0.25, 15.00)),
        "gpt-5.4-mini": ModelPricing(text=TokenRate(0.75, 0.075, 4.50)),
        "gpt-realtime": gpt_realtime,
        "gpt-realtime-2025-08-28": gpt_realtime,
        "gpt-realtime-1.5": gpt_realtime,
        "gpt-realtime-mini": ModelPricing(
            text=TokenRate(0.60, 0.06, 2.40),
            audio=TokenRate(10.00, 0.30, 20.00),
            image=TokenRate(0.80, 0.08, None),
        ),
        "gpt-4o-transcribe": ModelPricing(
            transcription=TokenRate(2.50, None, 10.00), duration_per_minute=0.006
        ),
        "gpt-4o-mini-transcribe": ModelPricing(
            transcription=TokenRate(1.25, None, 5.00), duration_per_minute=0.003
        ),
        "whisper-1": ModelPricing(duration_per_minute=0.006),
    }
