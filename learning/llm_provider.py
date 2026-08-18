from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class GenerateResult(str):
    """String-compatible output carrying non-private completion metadata."""

    def __new__(cls, value, *, incomplete_reason=None, output_tokens=None, limit=None):
        instance = super().__new__(cls, value or "")
        instance.incomplete_reason = incomplete_reason
        instance.output_tokens = output_tokens
        instance.limit = limit
        return instance


@dataclass(frozen=True)
class GenerateRequest:
    instructions: str
    input: str
    max_output_tokens: int
    safety_identifier: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SummarizeRequest:
    instructions: str
    input: str
    max_output_tokens: int = 240
    safety_identifier: str | None = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def available(self) -> bool:
        ...

    def generate(self, request: GenerateRequest) -> str | None:
        ...

    def summarize(self, request: SummarizeRequest) -> dict | None:
        ...
