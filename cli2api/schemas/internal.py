"""Internal models for provider communication."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderChunk:
    """A single chunk of data from a streaming provider response."""

    content: str
    is_final: bool = False
    usage: Optional[dict] = None
    tool_calls: Optional[list[dict]] = None
    reasoning: Optional[str] = None  # Reasoning/thinking text


@dataclass
class ProviderResult:
    """Complete result from a non-streaming provider response."""

    content: str
    session_id: Optional[str] = None
    usage: Optional[dict] = field(default_factory=dict)
    tool_calls: Optional[list[dict]] = None
