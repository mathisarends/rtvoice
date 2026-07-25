from decimal import Decimal

from transitbus import EventBus

from rtvoice.realtime.schemas import (
    DurationUsage,
    InputAudioTranscriptionCompleted,
    ResponseDoneEvent,
    TokenUsage,
)
from rtvoice.tokens.models import (
    TokenTotals,
    TranscriptionTokenTotals,
    UsageReport,
)
from rtvoice.tokens.pricing import PricingCatalog


class TokenTracker:
    def __init__(
        self,
        *,
        event_bus: EventBus,
        realtime_model: str,
        transcription_model: str | None = None,
        pricing_catalog: PricingCatalog | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._realtime_model = realtime_model
        self._transcription_model = transcription_model
        self._pricing_catalog = pricing_catalog or PricingCatalog()
        self._totals = TokenTotals()
        self._response_ids: set[str] = set()
        self._transcription_ids: set[tuple[str, int]] = set()

        self._event_bus.on(ResponseDoneEvent, self._on_response_done)
        self._event_bus.on(
            InputAudioTranscriptionCompleted,
            self._on_transcription_completed,
        )

    @property
    def totals(self) -> TokenTotals:
        return self._totals.model_copy(deep=True)

    def report(self) -> UsageReport:
        totals = self.totals
        return UsageReport(
            tokens=totals,
            cost=self._pricing_catalog.estimate(
                totals,
                realtime_model=self._realtime_model,
                transcription_model=self._transcription_model,
            ),
        )

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        if event.response_id in self._response_ids or event.response.usage is None:
            return
        self._response_ids.add(event.response_id)
        usage = event.response.usage
        totals = self._totals.realtime
        totals.responses += 1
        totals.total_tokens += usage.total_tokens or 0
        totals.input_tokens += usage.input_tokens or 0
        totals.output_tokens += usage.output_tokens or 0

        if details := usage.input_token_details:
            totals.input_text_tokens += details.text_tokens or 0
            totals.input_audio_tokens += details.audio_tokens or 0
            totals.input_image_tokens += details.image_tokens or 0
            totals.cached_input_tokens += details.cached_tokens or 0
            if cached := details.cached_tokens_details:
                totals.cached_input_text_tokens += cached.text_tokens or 0
                totals.cached_input_audio_tokens += cached.audio_tokens or 0
                totals.cached_input_image_tokens += cached.image_tokens or 0
        if details := usage.output_token_details:
            totals.output_text_tokens += details.text_tokens or 0
            totals.output_audio_tokens += details.audio_tokens or 0

    async def _on_transcription_completed(
        self, event: InputAudioTranscriptionCompleted
    ) -> None:
        key = (event.item_id, event.content_index)
        if key in self._transcription_ids or event.usage is None:
            return
        self._transcription_ids.add(key)
        totals = self._totals.transcription
        totals.transcriptions += 1
        usage = event.usage

        if isinstance(usage, DurationUsage):
            totals.duration_seconds += Decimal(str(usage.seconds))
            return
        self._add_transcription_tokens(totals, usage)

    def _add_transcription_tokens(
        self,
        totals: TranscriptionTokenTotals,
        usage: TokenUsage,
    ) -> None:
        totals.total_tokens += usage.total_tokens or 0
        totals.input_tokens += usage.input_tokens or 0
        totals.output_tokens += usage.output_tokens or 0
        if details := usage.input_token_details:
            totals.input_text_tokens += details.text_tokens or 0
            totals.input_audio_tokens += details.audio_tokens or 0
