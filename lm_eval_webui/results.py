"""Utilities for parsing lm-eval result JSON files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_META_KEYS = {"alias", "name", "sample_len"}
_CATEGORY_ORDER = [
    "Reasoning",
    "Math",
    "Coding / Structured Output",
    "Instruction Following",
    "Other",
]
_TASK_CATEGORIES = {
    "arc_challenge_chat": "Reasoning",
    "bbh_cot_zeroshot": "Reasoning",
    "truthfulqa_gen": "Reasoning",
    "gsm8k": "Math",
    "jsonschema_bench_easy": "Coding / Structured Output",
    "ifeval": "Instruction Following",
}
_TASK_CATEGORY_PATTERNS = [
    ("Math", ("gsm8k", "math", "aime", "amc", "minerva")),
    (
        "Coding / Structured Output",
        ("json", "schema", "code", "humaneval", "mbpp", "repobench", "longbench_lcc"),
    ),
    ("Instruction Following", ("ifeval", "instruction")),
    ("Reasoning", ("arc", "bbh", "truthful", "mmlu", "hellaswag", "winogrande")),
]
_TASK_SCORE_METRICS = {
    "gsm8k": [
        "exact_match,strict-match",
        "exact_match,flexible-extract",
    ],
    "ifeval": [
        "prompt_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_strict_acc,none",
        "inst_level_loose_acc,none",
    ],
    "truthfulqa_gen": [
        "bleu_acc,none",
        "rouge1_acc,none",
        "rouge2_acc,none",
        "rougeL_acc,none",
    ],
    "arc_challenge_chat": ["exact_match,remove_whitespace"],
    "jsonschema_bench_easy": ["json_validity,none", "schema_compliance,none"],
}
_FALLBACK_METRIC_BASES = (
    "acc_norm",
    "acc",
    "exact_match",
    "f1",
    "schema_compliance",
    "json_validity",
)


def _model_name(result_json: dict[str, Any]) -> str:
    config = result_json.get("config") or {}
    model_args = config.get("model_args") or {}
    return str(
        result_json.get("model_name")
        or config.get("model_name")
        or model_args.get("model")
        or config.get("model")
        or "unknown"
    )


def extract_result_rows(
    job_id: str, result_json: dict[str, Any]
) -> list[dict[str, Any]]:
    model = _model_name(result_json)
    limit = (result_json.get("config") or {}).get("limit")
    rows: list[dict[str, Any]] = []
    for task, metrics in (result_json.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        samples = _samples_for_task(str(task), result_json, metrics)
        for metric, value in metrics.items():
            if not _is_numeric_metric(metric, value):
                continue
            rows.append(
                {
                    "job_id": job_id,
                    "model": model,
                    "task": str(task),
                    "metric": str(metric),
                    "value": float(value),
                    "samples": samples,
                    "limit": limit,
                }
            )
    return rows


def extract_leaderboard_entry(
    job: dict[str, Any],
    result_json: dict[str, Any],
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_model_metadata = model_metadata or job.get("model_metadata") or {}
    model_metadata = raw_model_metadata if isinstance(raw_model_metadata, dict) else {}
    raw_config = result_json.get("config") or {}
    config = raw_config if isinstance(raw_config, dict) else {}
    raw_telemetry = job.get("telemetry") or {}
    telemetry = raw_telemetry if isinstance(raw_telemetry, dict) else {}
    task_scores: list[dict[str, Any]] = []
    for task, metrics in (result_json.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        scored_metrics = _scored_metrics(str(task), metrics)
        if not scored_metrics:
            continue
        score = sum(
            _score_value(metric, value) for metric, value in scored_metrics
        ) / len(scored_metrics)
        task_scores.append(
            {
                "task": str(task),
                "category": _task_category(str(task)),
                "metric": " + ".join(metric for metric, _value in scored_metrics),
                "metrics": [metric for metric, _value in scored_metrics],
                "values": dict(scored_metrics),
                "score": score,
                "samples": _samples_for_task(str(task), result_json, metrics),
            }
        )
    score_values = [task["score"] for task in task_scores]
    provider_backend = (
        model_metadata.get("runtime_backend")
        or model_metadata.get("llamacpp_backend")
        or job.get("runtime_backend")
        or job.get("llamacpp_backend")
        or job.get("requested_llamacpp_backend")
        or job.get("provider_backend")
        or job.get("lemonade_backend")
        or model_metadata.get("recipe")
        or job.get("recipe")
        or job.get("backend")
    )
    return {
        "job_id": job.get("id"),
        "model": _model_name(result_json),
        "model_id": job.get("model_id") or _model_name(result_json),
        "backend": str(config.get("model") or job.get("backend") or ""),
        "provider_backend": provider_backend,
        "lemonade_backend": provider_backend,
        "context_window": model_metadata.get("context_window")
        or job.get("context_window"),
        "status": job.get("status"),
        "limit": config.get("limit"),
        "total_evaluation_time_seconds": result_json.get(
            "total_evaluation_time_seconds"
        ),
        "generation_tok_s": telemetry.get("generation_tok_s")
        or telemetry.get("probe_generation_tok_s"),
        "prompt_tok_s": telemetry.get("prompt_tok_s")
        or telemetry.get("probe_prompt_tok_s"),
        "ttft_s": telemetry.get("ttft_s"),
        "overall_score": sum(score_values) / len(score_values)
        if score_values
        else None,
        "category_scores": _category_scores(task_scores),
        "task_scores": task_scores,
    }


def _category_scores(task_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for task in task_scores:
        if task.get("score") is None:
            continue
        by_category.setdefault(str(task.get("category") or "Other"), []).append(task)
    scores: list[dict[str, Any]] = []
    for category in [
        *_CATEGORY_ORDER,
        *sorted(set(by_category) - set(_CATEGORY_ORDER)),
    ]:
        tasks = by_category.get(category, [])
        if not tasks:
            continue
        values = [float(task["score"]) for task in tasks]
        scores.append(
            {
                "category": category,
                "score": sum(values) / len(values),
                "tasks": sorted(str(task["task"]) for task in tasks),
            }
        )
    return scores


def _task_category(task: str) -> str:
    if task in _TASK_CATEGORIES:
        return _TASK_CATEGORIES[task]
    normalized = task.lower()
    for category, needles in _TASK_CATEGORY_PATTERNS:
        if any(needle in normalized for needle in needles):
            return category
    return "Other"


def _scored_metrics(task: str, metrics: dict[str, Any]) -> list[tuple[str, float]]:
    numeric = {
        metric: float(value)
        for metric, value in metrics.items()
        if _is_numeric_metric(metric, value)
    }
    configured = _TASK_SCORE_METRICS.get(task, [])
    scored: list[tuple[str, float]] = []
    for metric in configured:
        if metric in numeric:
            scored.append((metric, numeric[metric]))
    for metric in configured:
        if metric in numeric:
            continue
        for candidate, value in numeric.items():
            if (
                candidate not in {name for name, _value in scored}
                and _metric_base(candidate) == metric
            ):
                scored.append((candidate, value))
                break
    if scored:
        return scored
    for fallback_base in _FALLBACK_METRIC_BASES:
        matches = [
            (metric, value)
            for metric, value in numeric.items()
            if _metric_base(metric) == fallback_base
        ]
        if matches:
            return matches
    return list(numeric.items())


def _is_numeric_metric(metric: str, value: Any) -> bool:
    if metric in _META_KEYS or metric.endswith("_stderr") or "_stderr," in metric:
        return False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _metric_base(metric: str) -> str:
    return metric.split(",", 1)[0]


def _samples_for_task(
    task: str, result_json: dict[str, Any], metrics: dict[str, Any]
) -> Any:
    sample_info = result_json.get("n-samples") or {}
    if isinstance(sample_info.get(task), dict):
        return sample_info[task].get("effective")
    return metrics.get("sample_len")


def _score_value(metric: str, value: float) -> float:
    if _metric_base(metric) in {"smoothed_bleu_4"}:
        return value
    return value * 100 if 0 <= value <= 1 else value


def load_result_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_result_files(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    if not root.exists():
        return []
    return sorted(root.glob("**/results_*.json"), key=lambda path: path.stat().st_mtime)
