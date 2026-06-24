"""lm-eval model plugin for OpenAI-compatible chat completions."""

from __future__ import annotations

import copy
import importlib
import json
import time
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


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def add_runtime_options(
    payload: dict[str, Any], llamacpp_backend: Any = None
) -> dict[str, Any]:
    backend = str(llamacpp_backend or "").strip().lower()
    if backend and backend not in {"auto", "default"}:
        payload["llamacpp_backend"] = backend
        recipe_options = payload.get("recipe_options")
        if not isinstance(recipe_options, dict):
            recipe_options = {}
        recipe_options["llamacpp_backend"] = backend
        payload["recipe_options"] = recipe_options
    return payload


def stream_response_json(
    response: Any,
    started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Consume an OpenAI-compatible SSE response and return chat JSON + timings."""

    first_headers = clock()
    first_event = first_content = None
    model = None
    usage = None
    timings: dict[str, Any] = {}
    choices: dict[int, dict[str, Any]] = {}
    for raw_line in response.iter_lines(decode_unicode=True):
        now = clock()
        line = (
            raw_line.decode("utf-8", "replace")
            if isinstance(raw_line, bytes)
            else raw_line
        )
        line = str(line).strip()
        if not line or line.startswith(":"):
            continue
        if first_event is None:
            first_event = now
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        model = model or chunk.get("model")
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        if isinstance(chunk.get("timings"), dict):
            timings.update(chunk["timings"])
        for choice in chunk.get("choices") or []:
            index = int(choice.get("index", 0))
            stored = choices.setdefault(
                index,
                {"index": index, "message": {"role": "assistant", "content": ""}},
            )
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if content:
                if first_content is None:
                    first_content = now
                stored["message"]["content"] += content
            if reasoning:
                if first_content is None:
                    first_content = now
                stored["message"]["reasoning_content"] = (
                    stored["message"].get("reasoning_content", "") + reasoning
                )
            if choice.get("finish_reason") is not None:
                stored["finish_reason"] = choice.get("finish_reason")
    timings.update(
        {
            "time_to_headers_s": first_headers - started,
            "time_to_first_event_s": None
            if first_event is None
            else first_event - started,
            "ttft_s": None if first_content is None else first_content - started,
        }
    )
    if timings["ttft_s"] is None and timings["time_to_first_event_s"] is not None:
        timings["ttft_s"] = timings["time_to_first_event_s"]
        timings["ttft_source"] = "first_event_no_content"
    elif timings["ttft_s"] is not None:
        timings["ttft_source"] = "first_content"
    output: dict[str, Any] = {
        "choices": [choices[index] for index in sorted(choices)],
        "timings": timings,
    }
    if model is not None:
        output["model"] = model
    if usage is not None:
        output["usage"] = usage
    return output


@register_model("openai-compatible-chat-completions", "lemonade-chat-completions")
class OpenAICompatibleChatCompletion(LocalChatCompletionBase):
    def __init__(
        self,
        *args: Any,
        telemetry_path: str | None = None,
        stream_responses: Any = False,
        llamacpp_backend: Any = None,
        **kwargs: Any,
    ) -> None:
        self._stream_responses = truthy(stream_responses)
        self._llamacpp_backend = llamacpp_backend
        super().__init__(*args, **kwargs)
        global _CURRENT_TELEMETRY_PATH
        _CURRENT_TELEMETRY_PATH = str(telemetry_path) if telemetry_path else None

    @cached_property
    def header(self) -> dict[str, str]:
        return self._header or {"Content-Type": "application/json"}

    def model_call(
        self,
        messages: Any,
        *,
        generate: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if not generate or not self._stream_responses:
            return super().model_call(
                messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
            )
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        add_runtime_options(payload, self._llamacpp_backend)
        payload["stream"] = True
        requests_module = importlib.import_module("requests")

        started = time.perf_counter()
        response = requests_module.post(
            self.base_url,
            json=payload,
            headers=self.header,
            verify=self.verify_certificate,
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return stream_response_json(response, started)

    @staticmethod
    def parse_generations(outputs: Any, **_kwargs: Any) -> list[str]:
        append_timing_events(_CURRENT_TELEMETRY_PATH, outputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        generations: list[str] = []
        for output in outputs:
            choices = sorted(output.get("choices", []), key=itemgetter("index"))
            if not choices:
                generations.append("")
                continue
            for choice in choices:
                message = choice.get("message") or {}
                content = message.get("content")
                if content in (None, ""):
                    content = message.get("reasoning_content", "")
                generations.append(content or "")
        return generations


LemonadeChatCompletion = OpenAICompatibleChatCompletion
