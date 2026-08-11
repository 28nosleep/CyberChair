from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


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
    max_output_tokens: int = 500
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
