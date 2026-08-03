"""Telemetry helpers for Lemonade-backed benchmark runs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .lemonade import openai_api_url


def append_timing_events(
    telemetry_path: str | Path | None, outputs: Any, source: str = "lm_eval"
) -> None:
    if not telemetry_path:
        return
    path = Path(telemetry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_list = outputs if isinstance(outputs, list) else [outputs]
    with path.open("a", encoding="utf-8") as handle:
        for output in output_list:
            if not isinstance(output, dict):
                continue
            timings = output.get("timings")
            if not isinstance(timings, dict):
                continue
            handle.write(
                json.dumps(
                    {
                        "source": source,
                        "timestamp": time.time(),
                        "model": output.get("model"),
                        "timings": timings,
                        "usage": output.get("usage"),
                        "response": output.get("response"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def load_telemetry_events(telemetry_path: str | Path | None) -> list[dict[str, Any]]:
    if not telemetry_path:
        return []
    path = Path(telemetry_path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def aggregate_telemetry_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    generated_tokens = prompt_tokens = prompt_ms = 0.0
    generated_rate_tokens = generation_seconds = 0.0
    response_metadata_count = final_content_count = reasoning_count = 0
    generation_limit_count = 0
    ttft_values: list[float] = []
    for event in events:
        response = event.get("response") if isinstance(event, dict) else None
        if isinstance(response, dict):
            response_metadata_count += 1
            if response.get("has_final_content"):
                final_content_count += 1
            if response.get("has_reasoning"):
                reasoning_count += 1
            if response.get("hit_generation_limit"):
                generation_limit_count += 1
        timings = event.get("timings") if isinstance(event, dict) else None
        if not isinstance(timings, dict):
            timings = event if isinstance(event, dict) else None
        if not isinstance(timings, dict):
            continue
        usage = event.get("usage") if isinstance(event, dict) else None
        if not isinstance(usage, dict):
            usage = {}
        generated_count = _number(timings.get("predicted_n"))
        if generated_count is None:
            generated_count = _number(usage.get("completion_tokens"))
        prompt_count = _number(timings.get("prompt_n"))
        if prompt_count is None:
            prompt_count = _number(usage.get("prompt_tokens"))
        generated_tokens += generated_count or 0.0
        generated_duration_s = None
        generated_ms = _number(timings.get("predicted_ms"))
        if generated_ms is not None and generated_ms > 0:
            generated_duration_s = generated_ms / 1000.0
        else:
            client_elapsed = _number(timings.get("generation_elapsed_s"))
            if client_elapsed is not None and client_elapsed > 0:
                generated_duration_s = client_elapsed
        if generated_count is not None and generated_duration_s is not None:
            generated_rate_tokens += generated_count
            generation_seconds += generated_duration_s
        prompt_tokens += prompt_count or 0.0
        prompt_ms += _number(timings.get("prompt_ms")) or 0.0
        ttft = _number(timings.get("ttft_s") or timings.get("time_to_first_token_s"))
        if ttft is not None:
            ttft_values.append(ttft)
    aggregate: dict[str, Any] = {"request_count": len(events)}
    if response_metadata_count:
        aggregate.update(
            {
                "response_metadata_count": response_metadata_count,
                "final_content_response_count": final_content_count,
                "reasoning_response_count": reasoning_count,
                "empty_response_count": response_metadata_count - final_content_count,
                "generation_limited_response_count": generation_limit_count,
            }
        )
    if generated_tokens:
        aggregate["generated_tokens"] = _integer(generated_tokens)
    if generated_rate_tokens and generation_seconds:
        aggregate["generation_tok_s"] = generated_rate_tokens / generation_seconds
    if prompt_tokens:
        aggregate["prompt_tokens"] = _integer(prompt_tokens)
    if prompt_tokens and prompt_ms:
        aggregate["prompt_tok_s"] = prompt_tokens / (prompt_ms / 1000.0)
    if ttft_values:
        aggregate["ttft_s"] = sum(ttft_values) / len(ttft_values)
    return aggregate


def aggregate_telemetry_file(telemetry_path: str | Path | None) -> dict[str, Any]:
    return aggregate_telemetry_events(load_telemetry_events(telemetry_path))


def probe_lemonade_chat_telemetry(
    base_url: str, model_id: str, timeout: int = 300
) -> dict[str, Any]:
    started = time.perf_counter()
    first_headers = first_event = first_content = None
    final_timings: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Write exactly this sentence: red blue green."}
        ],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        openai_api_url(base_url, "/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            first_headers = time.perf_counter()
            for raw_line in response:
                now = time.perf_counter()
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
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
                if isinstance(chunk.get("timings"), dict):
                    final_timings = chunk["timings"]
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") or choice.get("message") or {}
                    if not isinstance(delta, dict):
                        continue
                    text = (
                        delta.get("content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or delta.get("analysis")
                        or ""
                    )
                    if text and first_content is None:
                        first_content = now
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return {"error": str(exc)}
    ended = time.perf_counter()
    result: dict[str, Any] = {
        "probe_elapsed_s": ended - started,
        "time_to_headers_s": None if first_headers is None else first_headers - started,
        "time_to_first_event_s": None if first_event is None else first_event - started,
        "ttft_s": None if first_content is None else first_content - started,
    }
    if first_content is None and first_event is not None:
        result["ttft_s"] = first_event - started
        result["ttft_source"] = "first_event_no_content"
    elif first_content is not None:
        result["ttft_source"] = "first_content"
    combined_timings = dict(final_timings or {})
    if first_content is not None:
        combined_timings["generation_elapsed_s"] = ended - first_content
    if combined_timings or usage:
        rates = aggregate_telemetry_events(
            [{"timings": combined_timings, "usage": usage}]
        )
        for key in (
            "generated_tokens",
            "generation_tok_s",
            "prompt_tokens",
            "prompt_tok_s",
        ):
            if key in rates:
                result[f"probe_{key}"] = rates[key]
    return result


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None
