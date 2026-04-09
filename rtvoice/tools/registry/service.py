"""Backward-compatible imports for older rtvoice.tools.registry.service path."""

from .registry import RealtimeToolRegistry, SubAgentToolRegistry, ToolRegistry

__all__ = ["RealtimeToolRegistry", "SubAgentToolRegistry", "ToolRegistry"]
