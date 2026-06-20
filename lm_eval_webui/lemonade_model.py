"""lm-eval model plugin for OpenAI-compatible chat completions."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import cached_property
from operator import itemgetter
from typing import Any, TypeVar, cast

from .telemetry import append_timing_events

T = TypeVar("T", bound=type[Any])
RegisterModel = Callable[..., Callable[[T], T]]


class _FallbackLocalChatCompletion:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._header = kwargs.get("header")


def _fallback_register_model(*_names: str) -> Callable[[T], T]:
    def decorate(cls: T) -> T:
        return cls

    return decorate


def _load_lm_eval_symbols() -> tuple[RegisterModel, type[Any]]:
    try:
        registry_module = importlib.import_module("lm_eval.api.registry")
        completions_module = importlib.import_module(
            "lm_eval.models.openai_completions"
        )
    except ModuleNotFoundError:
        return _fallback_register_model, _FallbackLocalChatCompletion
    return (
        cast(RegisterModel, registry_module.__dict__["register_model"]),
        cast(type[Any], completions_module.__dict__["LocalChatCompletion"]),
    )


register_model, LocalChatCompletionBase = _load_lm_eval_symbols()
_CURRENT_TELEMETRY_PATH: str | None = None


@register_model("openai-compatible-chat-completions", "lemonade-chat-completions")
class OpenAICompatibleChatCompletion(LocalChatCompletionBase):
    def __init__(
        self, *args: Any, telemetry_path: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        global _CURRENT_TELEMETRY_PATH
        _CURRENT_TELEMETRY_PATH = str(telemetry_path) if telemetry_path else None

    @cached_property
    def header(self) -> dict[str, str]:
        return self._header or {"Content-Type": "application/json"}

    @staticmethod
    def parse_generations(outputs: Any, **_kwargs: Any) -> list[str]:
        append_timing_events(_CURRENT_TELEMETRY_PATH, outputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        generations: list[str] = []
        for output in outputs:
            choices = sorted(output.get("choices", []), key=itemgetter("index"))
            for choice in choices:
                message = choice.get("message") or {}
                content = message.get("content")
                if content in (None, ""):
                    content = message.get("reasoning_content", "")
                generations.append(content or "")
        return generations


LemonadeChatCompletion = OpenAICompatibleChatCompletion
