"""Common interface for the things that turn prompt tokens into completion tokens."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenParams:
    adapter: str                      # "maker" or "solver": which LoRA to apply
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    think_budget: int = 1024          # max tokens inside <think> before </think> is forced
    think_bias: dict = field(default_factory=dict)  # token id -> logit bias, ONLY while thinking
    seed: int | None = None


@dataclass
class GenResult:
    token_ids: list[int]
    loss_mask: list[int]              # 0 for tokens we inserted (forced </think>), 1 otherwise
    finish_reason: str                # "stop" | "length"
    think_tokens: int = 0
    forced_think_end: bool = False
    marker_tokens: int = 0            # how many penalised marker tokens still got sampled


class Engine(ABC):
    """Every engine must implement thinking-only logit bias and the think budget."""

    max_parallel: int = 1

    def start(self):
        pass

    def stop(self):
        pass

    def pause(self):
        """Free memory during the training phase (only if the profile needs it)."""

    def resume(self):
        pass

    def sync_adapters(self, adapter_dirs: dict[str, str], restart: bool = True):
        """Load freshly trained LoRA adapters ({"maker": dir, "solver": dir})."""

    @abstractmethod
    def generate(self, prompt_ids: list[int], params: GenParams, think_end: int | None) -> GenResult:
        """think_end: token that closes the thinking block the prompt opened (None = no thinking)."""
        ...


def count_markers(ids: list[int], bias: dict) -> int:
    return sum(1 for t in ids if t in bias) if bias else 0
