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
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def aggregate_telemetry_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    generated_tokens = generated_ms = prompt_tokens = prompt_ms = 0.0
    ttft_values: list[float] = []
    for event in events:
        timings = event.get("timings") if isinstance(event, dict) else None
        if not isinstance(timings, dict):
            timings = event if isinstance(event, dict) else None
        if not isinstance(timings, dict):
            continue
        generated_tokens += _number(timings.get("predicted_n")) or 0.0
        generated_ms += _number(timings.get("predicted_ms")) or 0.0
        prompt_tokens += _number(timings.get("prompt_n")) or 0.0
        prompt_ms += _number(timings.get("prompt_ms")) or 0.0
        ttft = _number(timings.get("ttft_s") or timings.get("time_to_first_token_s"))
        if ttft is not None:
            ttft_values.append(ttft)
    aggregate: dict[str, Any] = {"request_count": len(events)}
    if generated_tokens:
        aggregate["generated_tokens"] = int(generated_tokens)
    if generated_tokens and generated_ms:
        aggregate["generation_tok_s"] = generated_tokens / (generated_ms / 1000.0)
    if prompt_tokens:
        aggregate["prompt_tokens"] = int(prompt_tokens)
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
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Write exactly this sentence: red blue green."}
        ],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
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
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("timings"), dict):
                    final_timings = chunk["timings"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or choice.get("message") or {}
                    text = delta.get("content") or delta.get("reasoning_content") or ""
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
    if final_timings:
        rates = aggregate_telemetry_events([{"timings": final_timings}])
        for key in (
            "generated_tokens",
            "generation_tok_s",
            "prompt_tokens",
            "prompt_tok_s",
        ):
            if key in rates:
                result[f"probe_{key}"] = rates[key]
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
