from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from rtvoice.tokens.models import (
    CostEstimate,
    CostLineItem,
    RealtimeTokenTotals,
    TokenTotals,
    TranscriptionTokenTotals,
)

_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class RealtimeRates:
    text_input: Decimal
    text_cached_input: Decimal
    text_output: Decimal
    audio_input: Decimal
    audio_cached_input: Decimal
    audio_output: Decimal
    image_input: Decimal | None = None
    image_cached_input: Decimal | None = None


@dataclass(frozen=True)
class TranscriptionRates:
    input_tokens: Decimal | None = None
    output_tokens: Decimal | None = None
    minute: Decimal | None = None


class PricingCatalog:
    """OpenAI's USD rate card; token rates are per million tokens."""

    AS_OF = date(2026, 7, 25)
    SOURCE = "https://developers.openai.com/api/docs/models"

    def __init__(
        self,
        *,
        realtime: dict[str, RealtimeRates] | None = None,
        transcription: dict[str, TranscriptionRates] | None = None,
    ) -> None:
        self._realtime = realtime or _REALTIME_RATES
        self._transcription = transcription or _TRANSCRIPTION_RATES

    def estimate(
        self,
        totals: TokenTotals,
        *,
        realtime_model: str,
        transcription_model: str | None,
    ) -> CostEstimate:
        items: list[CostLineItem] = []
        notes: list[str] = []
        complete = self._realtime_items(totals.realtime, realtime_model, items, notes)
        transcription_complete = self._transcription_items(
            totals.transcription,
            transcription_model,
            items,
            notes,
        )

        return CostEstimate(
            total=sum((item.cost for item in items), Decimal(0)),
            line_items=items,
            pricing_as_of=self.AS_OF,
            pricing_source=self.SOURCE,
            is_complete=complete and transcription_complete,
            notes=notes,
        )

    def _realtime_items(
        self,
        usage: RealtimeTokenTotals,
        model: str,
        items: list[CostLineItem],
        notes: list[str],
    ) -> bool:
        rates = self._realtime.get(model)
        if rates is None:
            if usage.total_tokens:
                notes.append(f"No realtime pricing configured for {model}.")
                return False
            return True

        cached = {
            "text": usage.cached_input_text_tokens,
            "audio": usage.cached_input_audio_tokens,
            "image": usage.cached_input_image_tokens,
        }
        inputs = {
            "text": usage.input_text_tokens,
            "audio": usage.input_audio_tokens,
            "image": usage.input_image_tokens,
        }
        complete = True
        detailed_cached = sum(cached.values())
        if usage.cached_input_tokens != detailed_cached:
            notes.append(
                f"{usage.cached_input_tokens - detailed_cached} cached realtime "
                "input tokens could not be assigned to a modality."
            )
            complete = False
        for modality, total in inputs.items():
            if cached[modality] > total:
                notes.append(
                    f"Cached {modality} tokens exceed input {modality} tokens."
                )
                complete = False

        self._add_tokens(
            items,
            "realtime.input.text",
            max(0, inputs["text"] - cached["text"]),
            rates.text_input,
        )
        self._add_tokens(
            items,
            "realtime.input.text.cached",
            cached["text"],
            rates.text_cached_input,
        )
        self._add_tokens(
            items,
            "realtime.input.audio",
            max(0, inputs["audio"] - cached["audio"]),
            rates.audio_input,
        )
        self._add_tokens(
            items,
            "realtime.input.audio.cached",
            cached["audio"],
            rates.audio_cached_input,
        )
        complete = (
            self._add_optional_tokens(
                items,
                "realtime.input.image",
                max(0, inputs["image"] - cached["image"]),
                rates.image_input,
                notes,
            )
            and complete
        )
        complete = (
            self._add_optional_tokens(
                items,
                "realtime.input.image.cached",
                cached["image"],
                rates.image_cached_input,
                notes,
            )
            and complete
        )
        self._add_tokens(
            items,
            "realtime.output.text",
            usage.output_text_tokens,
            rates.text_output,
        )
        self._add_tokens(
            items,
            "realtime.output.audio",
            usage.output_audio_tokens,
            rates.audio_output,
        )

        detailed_input = sum(inputs.values())
        detailed_output = usage.output_text_tokens + usage.output_audio_tokens
        if usage.input_tokens != detailed_input:
            notes.append(
                f"{usage.input_tokens - detailed_input} realtime input tokens "
                "could not be assigned to a modality."
            )
            complete = False
        if usage.output_tokens != detailed_output:
            notes.append(
                f"{usage.output_tokens - detailed_output} realtime output tokens "
                "could not be assigned to a modality."
            )
            complete = False
        return complete

    def _transcription_items(
        self,
        usage: TranscriptionTokenTotals,
        model: str | None,
        items: list[CostLineItem],
        notes: list[str],
    ) -> bool:
        if not usage.transcriptions:
            return True
        if model is None or model not in self._transcription:
            notes.append(f"No transcription pricing configured for {model}.")
            return False

        rates = self._transcription[model]
        if rates.minute is not None:
            seconds = usage.duration_seconds
            estimated = False
            if not seconds and usage.input_audio_tokens:
                seconds = Decimal(usage.input_audio_tokens) / Decimal(10)
                estimated = True
            if not seconds:
                notes.append(f"No duration available to price {model} transcription.")
                return False
            if estimated:
                notes.append(
                    "Whisper duration was estimated at 100 ms per audio token."
                )
            self._add_minutes(
                items,
                "transcription.duration",
                seconds / Decimal(60),
                rates.minute,
            )
            return True

        complete = True
        input_tokens = usage.input_text_tokens + usage.input_audio_tokens
        if usage.input_tokens != input_tokens:
            notes.append(
                f"{usage.input_tokens - input_tokens} transcription input tokens "
                "could not be assigned to a modality."
            )
            complete = False
        if rates.input_tokens is None or rates.output_tokens is None:
            notes.append(f"Incomplete transcription pricing configured for {model}.")
            return False
        self._add_tokens(
            items,
            "transcription.input",
            input_tokens,
            rates.input_tokens,
        )
        self._add_tokens(
            items,
            "transcription.output",
            usage.output_tokens,
            rates.output_tokens,
        )
        return complete

    def _add_optional_tokens(
        self,
        items: list[CostLineItem],
        category: str,
        tokens: int,
        rate: Decimal | None,
        notes: list[str],
    ) -> bool:
        if not tokens:
            return True
        if rate is None:
            notes.append(f"No price configured for {category}.")
            return False
        self._add_tokens(items, category, tokens, rate)
        return True

    def _add_tokens(
        self,
        items: list[CostLineItem],
        category: str,
        tokens: int,
        rate: Decimal,
    ) -> None:
        if not tokens:
            return
        quantity = Decimal(tokens)
        items.append(
            CostLineItem(
                category=category,
                quantity=quantity,
                unit="tokens",
                rate=rate,
                rate_unit="million_tokens",
                cost=quantity / _MILLION * rate,
            )
        )

    def _add_minutes(
        self,
        items: list[CostLineItem],
        category: str,
        minutes: Decimal,
        rate: Decimal,
    ) -> None:
        items.append(
            CostLineItem(
                category=category,
                quantity=minutes,
                unit="minutes",
                rate=rate,
                rate_unit="minute",
                cost=minutes * rate,
            )
        )


_STANDARD = RealtimeRates(
    text_input=Decimal("4"),
    text_cached_input=Decimal("0.40"),
    text_output=Decimal("16"),
    audio_input=Decimal("32"),
    audio_cached_input=Decimal("0.40"),
    audio_output=Decimal("64"),
    image_input=Decimal("5"),
    image_cached_input=Decimal("0.50"),
)
_REASONING = RealtimeRates(
    text_input=Decimal("4"),
    text_cached_input=Decimal("0.40"),
    text_output=Decimal("24"),
    audio_input=Decimal("32"),
    audio_cached_input=Decimal("0.40"),
    audio_output=Decimal("64"),
    image_input=Decimal("5"),
    image_cached_input=Decimal("0.50"),
)
_MINI = RealtimeRates(
    text_input=Decimal("0.60"),
    text_cached_input=Decimal("0.06"),
    text_output=Decimal("2.40"),
    audio_input=Decimal("10"),
    audio_cached_input=Decimal("0.30"),
    audio_output=Decimal("20"),
    image_input=Decimal("0.80"),
    image_cached_input=Decimal("0.08"),
)

_REALTIME_RATES = {
    "gpt-realtime-2.1": _REASONING,
    "gpt-realtime-2": _REASONING,
    "gpt-realtime-1.5": _STANDARD,
    "gpt-realtime": _STANDARD,
    "gpt-realtime-2.1-mini": _MINI,
    "gpt-realtime-mini": _MINI,
}

_TRANSCRIPTION_RATES = {
    "whisper-1": TranscriptionRates(minute=Decimal("0.006")),
    "gpt-4o-transcribe": TranscriptionRates(
        input_tokens=Decimal("2.50"),
        output_tokens=Decimal("10"),
    ),
    "gpt-4o-mini-transcribe": TranscriptionRates(
        input_tokens=Decimal("1.25"),
        output_tokens=Decimal("5"),
    ),
}
