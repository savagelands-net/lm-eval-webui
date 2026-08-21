"""Lemonade CLI benchmark command construction and result parsing."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .lemonade import lemonade_management_base_url

LEMONADE_BENCH_SUITE = "lemonade_bench"
DEFAULT_LEMONADE_BENCH_RUNS = 3
DEFAULT_LEMONADE_BENCH_WARMUP = 0
DEFAULT_LEMONADE_BENCH_TIMEOUT = 300
LEMONADE_BENCH_RESULT_NAME = "results.json"
LEMONADE_BENCH_RESPONSE_LOG_NAME = "responses.jsonl"
LEMONADE_BENCH_SCENARIO_RE = re.compile(
    r"^\s*Scenario:\s+(.+?)\s+\(([^)]+)\)\s*$", re.MULTILINE
)
LEMONADE_BENCH_CONFIGURATION_RE = re.compile(
    r"^===\s+\[(.+?)\]\s+(.+?)\s+===$", re.MULTILINE
)

_FALLBACK_SCENARIOS = (
    ("chat-short", "chat", 20),
    ("chat-long-output", "chat", 256),
    ("code-short", "coding", 60),
    ("code-explain", "coding", 128),
    ("code-debug", "coding", 100),
    ("context-32k", "long-context", 20),
    ("context-64k", "long-context", 20),
    ("context-128k", "long-context", 20),
    ("context-multi-turn", "long-context", 100),
    ("embed-small-string", "embed", None),
    ("embed-small-array", "embed", None),
    ("embed-long-string", "embed", None),
    ("embed-long-array", "embed", None),
    ("image-text0", "imagegen", None),
    ("image-text1", "imagegen", None),
    ("image-text2", "imagegen", None),
    ("image-text3", "imagegen", None),
    ("image-text4", "imagegen", None),
    ("image-text5", "imagegen", None),
    ("image-text6", "imagegen", None),
    ("image-small-cartoon", "imagegen", None),
    ("image-big-cartoon", "imagegen", None),
    ("image-small-no-style", "imagegen", None),
    ("image-big-no-style", "imagegen", None),
    ("image-small-photo", "imagegen", None),
    ("image-big-photo", "imagegen", None),
)
_CATEGORY_LABELS = {
    "chat": "Chat",
    "coding": "Coding",
    "long-context": "Long context",
    "embed": "Embeddings",
    "imagegen": "Image generation",
    "vision": "Vision",
}


@dataclass(frozen=True)
class LemonadeBenchRequest:
    """Inputs needed to run one model through ``lemonade bench``."""

    model_id: str
    scenarios: list[str]
    output_path: str
    openai_base_url: str
    lemonade_model_id: str | None = None
    backends: list[str] = field(default_factory=list)
    context_sizes: list[int] = field(default_factory=list)
    measurement_runs: int = DEFAULT_LEMONADE_BENCH_RUNS
    warmup_runs: int = DEFAULT_LEMONADE_BENCH_WARMUP
    timeout: int = DEFAULT_LEMONADE_BENCH_TIMEOUT
    memory_tracking: bool = True
    reload_between_runs: bool = True
    log_responses: bool = False
    lemonade_cli: str | None = None


def find_lemonade_cli(explicit: str | None = None) -> str:
    """Resolve the Lemonade CLI without accepting a path from an API payload."""

    return (
        explicit
        or os.environ.get("LEMONADE_CLI")
        or shutil.which("lemonade")
        or "lemonade"
    )


def lemonade_cli_target(openai_base_url: str) -> str:
    """Convert an OpenAI ``/v1`` URL into the CLI's HTTPS-aware server target."""

    management_url = lemonade_management_base_url(openai_base_url)
    parsed = urlsplit(management_url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def build_lemonade_bench_command(
    request: LemonadeBenchRequest,
) -> tuple[list[str], dict[str, str]]:
    """Build a deterministic Lemonade CLI benchmark invocation."""

    output_path = Path(request.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        find_lemonade_cli(request.lemonade_cli),
        "--host",
        lemonade_cli_target(request.openai_base_url),
        "bench",
        "--json",
        "--output",
        str(output_path),
        "--runs",
        str(request.measurement_runs),
        "--warmup",
        str(request.warmup_runs),
        "--timeout",
        str(request.timeout),
    ]
    for backend in request.backends:
        command.extend(["--backend", backend])
    if request.context_sizes:
        command.append("--ctx-size")
        command.extend(str(size) for size in request.context_sizes)
    for scenario in request.scenarios:
        command.extend(["--scenarios", scenario])
    if not request.memory_tracking:
        command.append("--no-memory")
    if not request.reload_between_runs:
        command.append("--no-reload")
    if request.log_responses:
        command.extend(
            [
                "--response-log",
                str(output_path.with_name(LEMONADE_BENCH_RESPONSE_LOG_NAME)),
            ]
        )
    command.append(request.lemonade_model_id or request.model_id)
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    return command, env


def _scenario_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    if configured := os.environ.get("LEMONADE_BENCH_SCENARIOS"):
        candidates.append(Path(configured))
    executable = shutil.which("lemonade")
    if executable:
        candidates.append(
            Path(executable).resolve().parent / "resources" / "bench_scenarios.json"
        )
    candidates.extend(
        [
            Path("/opt/lemonade/resources/bench_scenarios.json"),
            Path("/usr/share/lemonade-server/resources/bench_scenarios.json"),
            Path("/usr/share/lemonade/resources/bench_scenarios.json"),
        ]
    )
    return candidates


def _load_scenario_payload(path: str | Path | None = None) -> dict[str, Any] | None:
    candidates = [Path(path)] if path is not None else _scenario_file_candidates()
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
            return payload
    return None


def find_lemonade_bench_scenarios(
    scenario_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return selectable scenario metadata from Lemonade's bundled catalog."""

    payload = _load_scenario_payload(scenario_file)
    raw_scenarios: list[Any]
    if payload is None:
        raw_scenarios = [
            {"name": name, "category": category, "max_tokens": max_tokens}
            for name, category, max_tokens in _FALLBACK_SCENARIOS
        ]
    else:
        raw_scenarios = payload["scenarios"]

    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_scenarios:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        category = str(raw.get("category") or "general").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        max_tokens = _positive_int(raw.get("max_tokens"))
        description = _CATEGORY_LABELS.get(category, category.replace("-", " ").title())
        if max_tokens is not None:
            description = f"{description} · up to {max_tokens:,} output tokens"
        scenarios.append(
            {
                "name": name,
                "description": description,
                "category": _CATEGORY_LABELS.get(
                    category, category.replace("-", " ").title()
                ),
                "scenario_category": category,
                "suite": LEMONADE_BENCH_SUITE,
                "compatibility": "compatible",
                "kind": "scenario",
                "default_selected": category not in {"long-context", "embed"},
            }
        )
    return scenarios


def find_lemonade_bench_result_files(run_dir: str | Path) -> list[Path]:
    result_path = Path(run_dir) / LEMONADE_BENCH_RESULT_NAME
    return [result_path] if result_path.is_file() else []


def _model_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("models")
    if isinstance(models, list):
        return [model for model in models if isinstance(model, dict)]
    return [payload] if isinstance(payload.get("results"), list) else []


def _measurement_runs(
    model_result: dict[str, Any], default: int = DEFAULT_LEMONADE_BENCH_RUNS
) -> int:
    config = model_result.get("config")
    configured = config.get("measurement_runs") if isinstance(config, dict) else None
    return _positive_int(configured, default) or default


def _backend_results(
    payload: dict[str, Any], fallback_model_id: str
) -> list[tuple[str, int, Any, dict[str, Any]]]:
    results: list[tuple[str, int, Any, dict[str, Any]]] = []
    for model_result in _model_results(payload):
        model_name = _public_model_id(model_result.get("model"), fallback_model_id)
        measurement_runs = _measurement_runs(model_result)
        timestamp = model_result.get("timestamp") or payload.get("timestamp")
        results.extend(
            (model_name, measurement_runs, timestamp, backend_result)
            for backend_result in model_result.get("results") or []
            if isinstance(backend_result, dict)
        )
    return results


def _public_model_id(model_id: Any, fallback: str) -> str:
    value = str(model_id or fallback).strip()
    return value[len("user.") :] if value.startswith("user.") else value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any, default: int | None = None) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _configuration(
    recipe: str, backend: str, context_size: int | None, backend_args: str
) -> str:
    label = "/".join(part for part in (recipe, backend) if part) or "default"
    if context_size:
        label += f" · {context_size:,} ctx"
    if backend_args:
        label += f" · {backend_args}"
    return label


def _scenario_base_row(
    job: dict[str, Any],
    model_name: str,
    backend_result: dict[str, Any],
    scenario: dict[str, Any],
    successful_runs: int,
) -> dict[str, Any]:
    recipe = str(backend_result.get("recipe") or "")
    backend = str(backend_result.get("backend") or "")
    context_size = _positive_int(backend_result.get("ctx_size"))
    backend_args = str(backend_result.get("backend_args") or "")
    return {
        "suite": LEMONADE_BENCH_SUITE,
        "job_id": str(job.get("id") or ""),
        "model": model_name,
        "task": str(scenario.get("name") or "unknown scenario"),
        "scenario_category": str(scenario.get("category") or "general"),
        "recipe": recipe,
        "backend": backend,
        "provider_backend": backend,
        "context_window": context_size,
        "backend_args": backend_args,
        "configuration": _configuration(recipe, backend, context_size, backend_args),
        "samples": successful_runs,
        "runtime_seconds": _finite_float(job.get("runtime_seconds")),
    }


def _append_metric(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    metric: str,
    value: Any,
) -> None:
    numeric = _finite_float(value)
    if numeric is not None:
        rows.append({**base, "metric": metric, "value": numeric})


def extract_lemonade_bench_result_rows(
    job: dict[str, Any], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Flatten Lemonade's model/backend/scenario JSON into detailed rows."""

    rows: list[dict[str, Any]] = []
    for model_name, measurement_runs, _timestamp, backend_result in _backend_results(
        payload, str(job["model_id"])
    ):
        for scenario in backend_result.get("scenarios") or []:
            if not isinstance(scenario, dict):
                continue
            failed_runs = _nonnegative_int(scenario.get("failed_runs"))
            successful_runs = max(0, measurement_runs - failed_runs)
            base = _scenario_base_row(
                job, model_name, backend_result, scenario, successful_runs
            )
            for source, prefix in (
                ("ttft_ms", "ttft"),
                ("tps", "tps"),
                ("duration_ms", "duration"),
            ):
                stats = scenario.get(source)
                if not isinstance(stats, dict):
                    continue
                for statistic in ("mean", "p50", "p95", "min", "max"):
                    suffix = "_ms" if source != "tps" else ""
                    _append_metric(
                        rows,
                        base,
                        f"{prefix}_{statistic}{suffix}",
                        stats.get(statistic),
                    )
            for source in (
                "vram_peak_gb",
                "memory_peak_gb",
                "input_tokens",
                "output_tokens",
                "failed_runs",
            ):
                _append_metric(rows, base, source, scenario.get(source))
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _scenario_stat(
    scenarios: list[dict[str, Any]], field: str, statistic: str = "mean"
) -> list[float]:
    values: list[float] = []
    for scenario in scenarios:
        stats = scenario.get(field)
        value = stats.get(statistic) if isinstance(stats, dict) else None
        if (numeric := _finite_float(value)) is not None:
            values.append(numeric)
    return values


def _peak(scenarios: list[dict[str, Any]], field: str) -> float | None:
    values = [
        numeric
        for scenario in scenarios
        if (numeric := _finite_float(scenario.get(field))) is not None
    ]
    return max(values) if values else None


def _measured_duration_seconds(
    scenarios: list[dict[str, Any]], measurement_runs: int
) -> float | None:
    total_ms = 0.0
    found = False
    for scenario in scenarios:
        duration = scenario.get("duration_ms")
        mean_ms = duration.get("mean") if isinstance(duration, dict) else None
        numeric = _finite_float(mean_ms)
        if numeric is None:
            continue
        successful_runs = max(
            0, measurement_runs - _nonnegative_int(scenario.get("failed_runs"))
        )
        total_ms += numeric * successful_runs
        found = True
    return total_ms / 1000.0 if found else None


def extract_lemonade_bench_leaderboard_entries(
    job: dict[str, Any], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Create one speed leaderboard entry per backend/context combination."""

    entries: list[dict[str, Any]] = []
    for model_name, measurement_runs, timestamp, backend_result in _backend_results(
        payload, str(job["model_id"])
    ):
        scenarios = [
            scenario
            for scenario in backend_result.get("scenarios") or []
            if isinstance(scenario, dict)
        ]
        if not scenarios:
            continue
        failed_runs = sum(
            _nonnegative_int(scenario.get("failed_runs")) for scenario in scenarios
        )
        total_runs = sum(
            max(
                0,
                measurement_runs - _nonnegative_int(scenario.get("failed_runs")),
            )
            for scenario in scenarios
        )
        successful_scenarios = sum(
            not bool(scenario.get("all_runs_failed")) for scenario in scenarios
        )
        average_tps = _mean(_scenario_stat(scenarios, "tps"))
        average_ttft = _mean(_scenario_stat(scenarios, "ttft_ms"))
        recipe = str(backend_result.get("recipe") or "")
        backend = str(backend_result.get("backend") or "")
        context_size = _positive_int(backend_result.get("ctx_size"))
        backend_args = str(backend_result.get("backend_args") or "")
        entries.append(
            {
                "suite": LEMONADE_BENCH_SUITE,
                "job_id": str(job.get("id") or ""),
                "model": model_name,
                "model_id": model_name,
                "status": job.get("status"),
                "partial": failed_runs > 0 or successful_scenarios < len(scenarios),
                "recipe": recipe,
                "backend": LEMONADE_BENCH_SUITE,
                "provider_backend": backend,
                "lemonade_backend": backend,
                "context_window": context_size,
                "backend_args": backend_args,
                "configuration": _configuration(
                    recipe, backend, context_size, backend_args
                ),
                "scenario_count": len(scenarios),
                "successful_scenarios": successful_scenarios,
                "total_runs": total_runs,
                "failed_runs": failed_runs,
                "average_ttft_ms": average_ttft,
                "p95_ttft_ms": _mean(_scenario_stat(scenarios, "ttft_ms", "p95")),
                "average_tps": average_tps,
                "p95_tps": _mean(_scenario_stat(scenarios, "tps", "p95")),
                "vram_peak_gb": _peak(scenarios, "vram_peak_gb"),
                "memory_peak_gb": _peak(scenarios, "memory_peak_gb"),
                "measured_duration_seconds": _measured_duration_seconds(
                    scenarios, measurement_runs
                ),
                "runtime_seconds": _finite_float(job.get("runtime_seconds")),
                "timestamp": timestamp,
                "overall_score": average_tps,
            }
        )
    return entries


def summarize_lemonade_bench_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract compact job-level performance telemetry from a bench result."""

    scenarios: list[dict[str, Any]] = []
    measurement_runs = DEFAULT_LEMONADE_BENCH_RUNS
    for model_result in _model_results(payload):
        measurement_runs = _measurement_runs(model_result, measurement_runs)
        for backend_result in model_result.get("results") or []:
            if isinstance(backend_result, dict):
                scenarios.extend(
                    scenario
                    for scenario in backend_result.get("scenarios") or []
                    if isinstance(scenario, dict)
                )
    telemetry: dict[str, Any] = {
        "request_count": sum(
            max(
                0,
                measurement_runs - _nonnegative_int(scenario.get("failed_runs")),
            )
            for scenario in scenarios
        ),
        "failed_request_count": sum(
            _nonnegative_int(scenario.get("failed_runs")) for scenario in scenarios
        ),
    }
    if (ttft_ms := _mean(_scenario_stat(scenarios, "ttft_ms"))) is not None:
        telemetry["ttft_s"] = ttft_ms / 1000.0
    if (generation_rate := _mean(_scenario_stat(scenarios, "tps"))) is not None:
        telemetry["generation_tok_s"] = generation_rate
    if (vram_peak := _peak(scenarios, "vram_peak_gb")) is not None:
        telemetry["vram_peak_gb"] = vram_peak
    if (memory_peak := _peak(scenarios, "memory_peak_gb")) is not None:
        telemetry["memory_peak_gb"] = memory_peak
    return telemetry
