from pydantic import BaseModel, Field


class TokenUsageCost(BaseModel):
    input_usd: float = 0.0
    cached_input_usd: float = 0.0
    output_usd: float = 0.0
    duration_usd: float = 0.0
    total_usd: float = 0.0


class TokenUsageBreakdown(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_text_tokens: int = 0
    cached_input_text_tokens: int = 0
    output_text_tokens: int = 0
    input_audio_tokens: int = 0
    cached_input_audio_tokens: int = 0
    output_audio_tokens: int = 0
    input_image_tokens: int = 0
    cached_input_image_tokens: int = 0
    duration_seconds: float = 0.0


class TokenUsageRecord(BaseModel):
    source: str
    model: str
    usage: TokenUsageBreakdown = Field(default_factory=TokenUsageBreakdown)
    cost: TokenUsageCost = Field(default_factory=TokenUsageCost)
    price_available: bool = True


class TokenUsageModelSummary(BaseModel):
    model: str
    usage: TokenUsageBreakdown = Field(default_factory=TokenUsageBreakdown)
    cost: TokenUsageCost = Field(default_factory=TokenUsageCost)
    price_available: bool = True


class TokenUsageSummary(BaseModel):
    usage: TokenUsageBreakdown = Field(default_factory=TokenUsageBreakdown)
    cost: TokenUsageCost = Field(default_factory=TokenUsageCost)
    by_model: list[TokenUsageModelSummary] = Field(default_factory=list)
    records: list[TokenUsageRecord] = Field(default_factory=list)
    has_unpriced_usage: bool = False
