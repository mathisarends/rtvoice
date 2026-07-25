from typing import TYPE_CHECKING, Any

from .models import (
    CostEstimate,
    CostLineItem,
    Currency,
    RealtimeTokenTotals,
    TokenTotals,
    TranscriptionTokenTotals,
    UsageReport,
)
from .pricing import (
    PricingCatalog,
    RealtimeRates,
    TranscriptionRates,
)

if TYPE_CHECKING:
    from .tracker import TokenTracker

__all__ = [
    "CostEstimate",
    "CostLineItem",
    "Currency",
    "PricingCatalog",
    "RealtimeRates",
    "RealtimeTokenTotals",
    "TokenTotals",
    "TokenTracker",
    "TranscriptionRates",
    "TranscriptionTokenTotals",
    "UsageReport",
]


def __getattr__(name: str) -> Any:
    if name == "TokenTracker":
        from .tracker import TokenTracker

        return TokenTracker
    raise AttributeError(name)
