"""lm-eval model plugin for OpenAI-compatible chat completions."""

from __future__ import annotations

import copy
import importlib
import json
import time
from collections.abc import Callable
from contextvars import ContextVar
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
_CURRENT_STOP_SEQUENCES: ContextVar[tuple[str, ...]] = ContextVar(
    "lm_eval_stop_sequences", default=()
)
_REASONING_FIELDS = ("reasoning", "reasoning_content", "analysis")
_THINKING_TAGS = (("<think>", "</think>"), ("<analysis>", "</analysis>"))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def enable_streaming_usage(payload: dict[str, Any]) -> dict[str, Any]:
    """Request the final OpenAI usage chunk needed for client-side throughput."""

    payload["stream"] = True
    raw_options = payload.get("stream_options")
    stream_options = dict(raw_options) if isinstance(raw_options, dict) else {}
    stream_options["include_usage"] = True
    payload["stream_options"] = stream_options
    return payload


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


def text_value(value: Any) -> str:
    """Normalize text fragments used by OpenAI-compatible response variants."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text_value(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return text_value(value[key])
    return ""


def reasoning_text(message: dict[str, Any]) -> str:
    """Read reasoning emitted by vLLM, llama.cpp, and legacy providers."""

    for field in _REASONING_FIELDS:
        if text := text_value(message.get(field)):
            return text
    return ""


def split_thinking_content(value: Any) -> tuple[str, str]:
    """Split providers that place ``<think>`` blocks inside normal content."""

    content = text_value(value)
    for opening, closing in _THINKING_TAGS:
        if closing in content:
            before, final = content.rsplit(closing, 1)
            reasoning = before.rsplit(opening, 1)[-1] if opening in before else before
            return reasoning.strip(), final.lstrip()
        if opening in content:
            return content.rsplit(opening, 1)[-1].strip(), ""
    return "", content


def normalize_stop_sequences(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = []
    return tuple(candidate for item in candidates if (candidate := text_value(item)))


def truncate_at_stop(text: str, stop_sequences: tuple[str, ...]) -> str:
    positions = [
        position for stop in stop_sequences if (position := text.find(stop)) >= 0
    ]
    return text[: min(positions)] if positions else text


def final_answer_text(
    message: dict[str, Any], stop_sequences: tuple[str, ...] = ()
) -> str:
    """Return only the final answer, never an unfinished reasoning trace."""

    embedded_reasoning, content = split_thinking_content(message.get("content"))
    if embedded_reasoning or reasoning_text(message):
        content = content.lstrip("\r\n")
    final = truncate_at_stop(content, stop_sequences)
    return final if final.strip() else ""


def response_metadata(
    choices: list[dict[str, Any]], stop_sequences: tuple[str, ...] = ()
) -> dict[str, Any]:
    final_content_chars = 0
    reasoning_chars = 0
    finish_reasons: list[str] = []
    for choice in choices:
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            continue
        embedded_reasoning, _content = split_thinking_content(message.get("content"))
        final_content_chars += len(final_answer_text(message, stop_sequences))
        reasoning_chars += len(reasoning_text(message) or embedded_reasoning)
        if choice.get("finish_reason") is not None:
            finish_reasons.append(str(choice["finish_reason"]))
    return {
        "has_final_content": final_content_chars > 0,
        "has_reasoning": reasoning_chars > 0,
        "final_content_chars": final_content_chars,
        "reasoning_chars": reasoning_chars,
        "finish_reasons": finish_reasons,
        "hit_generation_limit": any(
            reason.lower() in {"length", "max_tokens"} for reason in finish_reasons
        ),
    }


def record_stop_sequences(
    output: Any, stop_sequences: tuple[str, ...]
) -> dict[str, Any] | None:
    if not isinstance(output, dict):
        return None
    output["_lm_eval_stop_sequences"] = list(stop_sequences)
    return output


def prepare_generation_payload(
    payload: dict[str, Any], minimum_gen_toks: Any
) -> dict[str, Any]:
    """Let thinking finish before applying task stop strings to final content."""

    payload.pop("stop", None)
    try:
        minimum = max(1, int(minimum_gen_toks))
    except (TypeError, ValueError, OverflowError):
        minimum = 1
    token_key = (
        "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
    )
    try:
        requested = int(payload.get(token_key) or 0)
    except (TypeError, ValueError, OverflowError):
        requested = 0
    payload[token_key] = max(minimum, requested)
    return payload


def stream_response_json(
    response: Any,
    started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Consume an OpenAI-compatible SSE response and return chat JSON + timings."""

    first_headers = clock()
    ended = first_headers
    first_event = first_token = None
    first_token_source = None
    model = None
    usage = None
    timings: dict[str, Any] = {}
    choices: dict[int, dict[str, Any]] = {}
    for raw_line in response.iter_lines(decode_unicode=True):
        now = clock()
        ended = now
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
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(chunk, dict):
            continue
        model = model or chunk.get("model")
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        if isinstance(chunk.get("timings"), dict):
            timings.update(chunk["timings"])
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            try:
                index = int(choice.get("index", 0))
            except (TypeError, ValueError, OverflowError):
                continue
            stored = choices.setdefault(
                index,
                {"index": index, "message": {"role": "assistant", "content": ""}},
            )
            delta = choice.get("delta") or choice.get("message") or {}
            content = text_value(delta.get("content"))
            reasoning = reasoning_text(delta)
            if content:
                if first_token is None:
                    first_token = now
                    first_token_source = "first_content"
                stored["message"]["content"] += content
            if reasoning:
                if first_token is None:
                    first_token = now
                    first_token_source = "first_reasoning"
                stored["message"]["reasoning"] = (
                    stored["message"].get("reasoning", "") + reasoning
                )
            if choice.get("finish_reason") is not None:
                stored["finish_reason"] = choice.get("finish_reason")
    timings.update(
        {
            "request_elapsed_s": ended - started,
            "generation_elapsed_s": None
            if first_token is None
            else ended - first_token,
            "time_to_headers_s": first_headers - started,
            "time_to_first_event_s": None
            if first_event is None
            else first_event - started,
            "ttft_s": None if first_token is None else first_token - started,
        }
    )
    if timings["ttft_s"] is None and timings["time_to_first_event_s"] is not None:
        timings["ttft_s"] = timings["time_to_first_event_s"]
        timings["ttft_source"] = "first_event_no_content"
    elif timings["ttft_s"] is not None:
        timings["ttft_source"] = first_token_source
    client_timing = False
    if usage is not None and timings.get("predicted_n") is None:
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is not None:
            timings["predicted_n"] = completion_tokens
            client_timing = True
    if first_token is not None and timings.get("predicted_ms") is None:
        timings["predicted_ms"] = max(0.0, ended - first_token) * 1000.0
        client_timing = True
    if client_timing:
        timings["generation_timing_source"] = "client_stream"
    ordered_choices = [choices[index] for index in sorted(choices)]
    output: dict[str, Any] = {
        "choices": ordered_choices,
        "timings": timings,
        "response": response_metadata(ordered_choices),
    }
    if model is not None:
        output["model"] = model
    if usage is not None:
        output["usage"] = usage
    return output


async def astream_response_json(
    response: Any,
    started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Capture asynchronous SSE line arrival times and normalize the response."""

    first_headers = clock()
    lines: list[bytes] = []
    arrival_times: list[float] = []
    while True:
        raw_line = await response.content.readline()
        if not raw_line:
            break
        lines.append(raw_line)
        arrival_times.append(clock())

    class BufferedResponse:
        @staticmethod
        def iter_lines(decode_unicode: bool = False) -> list[Any]:
            if decode_unicode:
                return [line.decode("utf-8", "replace") for line in lines]
            return lines

    fallback_time = arrival_times[-1] if arrival_times else first_headers
    observed_times = iter([first_headers, *arrival_times])
    return stream_response_json(
        BufferedResponse(),
        started,
        clock=lambda: next(observed_times, fallback_time),
    )


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
        self._telemetry_path = str(telemetry_path) if telemetry_path else None
        super().__init__(*args, **kwargs)
        global _CURRENT_TELEMETRY_PATH
        _CURRENT_TELEMETRY_PATH = self._telemetry_path

    @cached_property
    def header(self) -> dict[str, str]:
        return self._header or {"Content-Type": "application/json"}

    def _create_payload(
        self,
        messages: Any,
        generate: bool = False,
        gen_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        generation_kwargs = copy.deepcopy(gen_kwargs) or {}
        _CURRENT_STOP_SEQUENCES.set(
            normalize_stop_sequences(generation_kwargs.pop("until", None))
        )
        payload = super()._create_payload(
            messages,
            generate=generate,
            gen_kwargs=generation_kwargs,
            **kwargs,
        )
        if generate:
            prepare_generation_payload(payload, getattr(self, "_max_gen_toks", 1))
        return add_runtime_options(payload, self._llamacpp_backend)

    def model_call(
        self,
        messages: Any,
        *,
        generate: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        stop_sequences = normalize_stop_sequences((gen_kwargs or {}).get("until"))
        if not generate or not self._stream_responses:
            output = super().model_call(
                messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
            )
            return record_stop_sequences(output, stop_sequences)
        generation_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=generation_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        enable_streaming_usage(payload)
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
        output = stream_response_json(response, started)
        record_stop_sequences(output, stop_sequences)
        append_timing_events(self._telemetry_path, output)
        output["_lm_eval_telemetry_recorded"] = True
        return output

    async def amodel_call(
        self,
        session: Any,
        sem: Any,
        messages: Any,
        *,
        generate: bool = True,
        cache_keys: list[Any] | None = None,
        ctxlens: list[int] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not generate or not self._stream_responses:
            return await LocalChatCompletionBase.amodel_call(
                self,
                session,
                sem,
                messages,
                generate=generate,
                cache_keys=cache_keys,
                ctxlens=ctxlens,
                gen_kwargs=gen_kwargs,
                **kwargs,
            )

        stop_sequences = normalize_stop_sequences((gen_kwargs or {}).get("until"))
        payload = self._create_payload(
            self.create_message(messages),
            generate=True,
            gen_kwargs=copy.deepcopy(gen_kwargs),
            seed=self._seed,
            **kwargs,
        )
        enable_streaming_usage(payload)
        acquired = await sem.acquire()
        try:
            started = time.perf_counter()
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                response.raise_for_status()
                output = await astream_response_json(response, started)
            record_stop_sequences(output, stop_sequences)
            append_timing_events(self._telemetry_path, output)
            output["_lm_eval_telemetry_recorded"] = True
            answers = self.parse_generations(output)
            if cache_keys:
                for answer, cache_key in zip(answers, cache_keys, strict=False):
                    self.cache_hook.add_partial("generate_until", cache_key, answer)
            return answers
        finally:
            if acquired:
                sem.release()

    def parse_generations(self, outputs: Any, **_kwargs: Any) -> list[str]:
        if not isinstance(outputs, list):
            outputs = [outputs]
        generations: list[str] = []
        unrecorded_outputs: list[dict[str, Any]] = []
        for output in outputs:
            if not isinstance(output, dict):
                generations.append("")
                continue
            if not output.pop("_lm_eval_telemetry_recorded", False):
                unrecorded_outputs.append(output)
            choices = sorted(output.get("choices", []), key=itemgetter("index"))
            stop_sequences = (
                normalize_stop_sequences(output.get("_lm_eval_stop_sequences"))
                if "_lm_eval_stop_sequences" in output
                else _CURRENT_STOP_SEQUENCES.get()
            )
            output["response"] = response_metadata(choices, stop_sequences)
            if not choices:
                generations.append("")
                continue
            for choice in choices:
                message = choice.get("message") or {}
                generations.append(final_answer_text(message, stop_sequences))
        telemetry_path = getattr(self, "_telemetry_path", _CURRENT_TELEMETRY_PATH)
        append_timing_events(telemetry_path, unrecorded_outputs)
        return generations


LemonadeChatCompletion = OpenAICompatibleChatCompletion
