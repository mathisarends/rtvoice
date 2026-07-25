from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Currency(StrEnum):
    USD = "USD"


class RealtimeTokenTotals(BaseModel):
    responses: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    input_image_tokens: int = 0
    cached_input_tokens: int = 0
    cached_input_text_tokens: int = 0
    cached_input_audio_tokens: int = 0
    cached_input_image_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0


class TranscriptionTokenTotals(BaseModel):
    transcriptions: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    duration_seconds: Decimal = Decimal(0)


class TokenTotals(BaseModel):
    realtime: RealtimeTokenTotals = Field(default_factory=RealtimeTokenTotals)
    transcription: TranscriptionTokenTotals = Field(
        default_factory=TranscriptionTokenTotals
    )


class CostLineItem(BaseModel):
    category: str
    quantity: Decimal
    unit: Literal["tokens", "minutes"]
    rate: Decimal
    rate_unit: Literal["million_tokens", "minute"]
    cost: Decimal


class CostEstimate(BaseModel):
    currency: Currency = Currency.USD
    total: Decimal
    line_items: list[CostLineItem]
    pricing_as_of: date
    pricing_source: str
    is_complete: bool = True
    notes: list[str] = Field(default_factory=list)


class UsageReport(BaseModel):
    tokens: TokenTotals
    cost: CostEstimate
