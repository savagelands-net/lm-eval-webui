"""Utilities for parsing lm-eval result JSON files."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

PROFILE_VERSION = 1
SCORE_METHOD = "category-balanced-v1"
BALANCED_SCORE_CATEGORIES = (
    "Reasoning",
    "Math",
    "Coding / Structured Output",
    "Instruction Following",
)
BALANCED_PROFILE_TASKS = (
    "ifeval",
    "gsm8k",
    "minerva_math500",
    "mmlu_pro_computer_science",
    "mmlu_pro_engineering",
    "bbh_cot_zeroshot_logical_deduction_five_objects",
    "arc_challenge_chat",
    "jsonschema_bench_easy",
)
_PROFILE_OPTION_KEYS = (
    "limit",
    "num_fewshot",
    "batch_size",
    "max_gen_toks",
    "num_concurrent",
    "timeout",
    "apply_chat_template",
    "fewshot_as_multiturn",
    "log_samples",
    "predict_only",
    "task_batch_size",
    "max_concurrent_jobs",
)
_INTEGER_PROFILE_OPTIONS = {
    "limit",
    "num_fewshot",
    "max_gen_toks",
    "num_concurrent",
    "timeout",
    "task_batch_size",
    "max_concurrent_jobs",
}
_BOOLEAN_PROFILE_OPTIONS = {
    "apply_chat_template",
    "fewshot_as_multiturn",
    "log_samples",
    "predict_only",
}
_SHARED_PROFILE_SETTINGS: dict[str, Any] = {
    "num_fewshot": None,
    "batch_size": "1",
    "max_gen_toks": 32768,
    "num_concurrent": 2,
    "timeout": 7200,
    "apply_chat_template": True,
    "fewshot_as_multiturn": False,
    "log_samples": True,
    "predict_only": False,
    "task_batch_size": 4,
    "max_concurrent_jobs": 1,
}
_PROFILE_SPECS = (
    {
        "id": "strix-balanced-quick-v1",
        "label": "Quick Screen",
        "description": "Up to 50 examples from each balanced task.",
        "limit": 50,
        "warning": None,
    },
    {
        "id": "strix-balanced-standard-v1",
        "label": "Standard Compare",
        "description": "Up to 200 examples from each balanced task.",
        "limit": 200,
        "warning": None,
    },
    {
        "id": "strix-balanced-full-v1",
        "label": "Full Validation",
        "description": "Every available example from each balanced task.",
        "limit": None,
        "warning": "Full validation can take multiple days per model.",
    },
)


def lm_eval_profiles() -> list[dict[str, Any]]:
    """Return independent public profile definitions for API and UI consumers."""

    profiles: list[dict[str, Any]] = []
    for spec in _PROFILE_SPECS:
        settings = {"limit": spec["limit"], **_SHARED_PROFILE_SETTINGS}
        limit = spec["limit"]
        profiles.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "version": PROFILE_VERSION,
                "description": spec["description"],
                "tasks": list(BALANCED_PROFILE_TASKS),
                "settings": settings,
                "maximum_samples": (
                    len(BALANCED_PROFILE_TASKS) * limit
                    if isinstance(limit, int)
                    else None
                ),
                "warning": spec["warning"],
            }
        )
    return deepcopy(profiles)


def classify_lm_eval_profile(
    tasks: list[str] | tuple[str, ...], options: dict[str, Any]
) -> dict[str, Any]:
    """Classify a normalized job recipe against the versioned built-ins."""

    canonical_tasks = sorted(str(task) for task in tasks)
    canonical_options = _canonical_profile_options(options)
    for profile in lm_eval_profiles():
        if canonical_tasks != sorted(profile["tasks"]):
            continue
        if canonical_options != _canonical_profile_options(profile["settings"]):
            continue
        return _profile_identity(profile)
    return custom_profile()


def benchmark_profile_for_job(job: dict[str, Any]) -> dict[str, Any]:
    """Return persisted profile identity, or mark old jobs as custom legacy jobs."""

    raw_profile = job.get("benchmark_profile")
    if isinstance(raw_profile, dict) and raw_profile.get("id"):
        return deepcopy(raw_profile)
    return custom_profile(legacy=True)


def profile_definition(profile_id: Any) -> dict[str, Any] | None:
    requested = str(profile_id or "")
    for profile in lm_eval_profiles():
        if profile["id"] == requested:
            return profile
    return None


def profile_expected_tasks(profile: dict[str, Any]) -> tuple[str, ...]:
    definition = profile_definition(profile.get("id"))
    if not definition:
        return ()
    return tuple(str(task) for task in definition["tasks"])


def custom_profile(*, legacy: bool = False) -> dict[str, Any]:
    return {
        "id": "custom",
        "label": "Custom (legacy)" if legacy else "Custom",
        "version": None,
        "custom": True,
        "legacy": legacy,
    }


def _profile_identity(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": profile["id"],
        "label": profile["label"],
        "version": profile["version"],
        "custom": False,
        "legacy": False,
    }


def _canonical_profile_options(options: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _canonical_profile_option(key, options.get(key))
        for key in _PROFILE_OPTION_KEYS
    }


def _canonical_profile_option(key: str, value: Any) -> Any:
    if key == "batch_size":
        return str(value or "1")
    if key in _BOOLEAN_PROFILE_OPTIONS:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if key in _INTEGER_PROFILE_OPTIONS:
        if value in (None, ""):
            return None
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return str(value)
        return number if number.is_finite() else str(value)
    return value


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
    "gsm8k": ["exact_match,flexible-extract"],
    "minerva_math500": ["math_verify,none"],
    "mmlu_pro_computer_science": ["exact_match,custom-extract"],
    "mmlu_pro_engineering": ["exact_match,custom-extract"],
    "bbh_cot_zeroshot_logical_deduction_five_objects": ["exact_match,flexible-extract"],
    "ifeval": ["prompt_level_strict_acc,none"],
    "truthfulqa_gen": [
        "bleu_acc,none",
        "rouge1_acc,none",
        "rouge2_acc,none",
        "rougeL_acc,none",
    ],
    "arc_challenge_chat": ["exact_match,remove_whitespace"],
    "jsonschema_bench_easy": ["schema_compliance,none"],
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


def _concrete_backend(value: Any) -> str | None:
    if value in (None, ""):
        return None
    backend = str(value)
    return None if backend == "llamacpp" else backend


def _recipe_backend(recipe: Any) -> str | None:
    if recipe in (None, ""):
        return None
    backend = str(recipe)
    return "system" if backend == "llamacpp" else backend


def _provider_backend(job: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    for value in (
        metadata.get("llamacpp_backend"),
        metadata.get("runtime_backend"),
        job.get("llamacpp_backend"),
        job.get("requested_llamacpp_backend"),
        job.get("runtime_backend"),
        job.get("provider_backend"),
        job.get("lemonade_backend"),
    ):
        backend = _concrete_backend(value)
        if backend:
            return backend
    for recipe in (metadata.get("recipe"), job.get("recipe")):
        backend = _recipe_backend(recipe)
        if backend:
            return backend
    return _concrete_backend(job.get("backend"))


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def extract_result_rows(
    job_id: str,
    result_json: dict[str, Any],
    *,
    benchmark_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    model = _model_name(result_json)
    limit = (result_json.get("config") or {}).get("limit")
    profile = benchmark_profile or custom_profile(legacy=True)
    rows: list[dict[str, Any]] = []
    for task, metrics in (result_json.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        samples = _samples_for_task(str(task), result_json, metrics)
        for metric, value in metrics.items():
            numeric_value = _finite_float(value)
            if not _is_numeric_metric(metric, value) or numeric_value is None:
                continue
            rows.append(
                {
                    "job_id": job_id,
                    "model": model,
                    "task": str(task),
                    "metric": str(metric),
                    "value": numeric_value,
                    "samples": samples,
                    "limit": limit,
                    "profile_id": profile.get("id"),
                    "profile_label": profile.get("label"),
                    "profile_version": profile.get("version"),
                }
            )
    return rows


def merge_result_jsons(result_jsons: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    total_time = 0.0
    has_total_time = False
    for result_json in result_jsons:
        if not isinstance(result_json, dict):
            continue
        if not merged:
            merged = {
                key: value
                for key, value in result_json.items()
                if key
                not in {
                    "results",
                    "n-samples",
                    "versions",
                    "configs",
                    "total_evaluation_time_seconds",
                }
            }
            merged["results"] = {}
        for key in ("results", "n-samples", "versions", "configs"):
            value = result_json.get(key)
            if isinstance(value, dict):
                merged.setdefault(key, {}).update(value)
        elapsed = _finite_float(result_json.get("total_evaluation_time_seconds"))
        if elapsed is not None:
            total_time += elapsed
            has_total_time = True
    if has_total_time:
        merged["total_evaluation_time_seconds"] = total_time
    return merged


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
    raw_results = result_json.get("results") or {}
    results = raw_results if isinstance(raw_results, dict) else {}
    benchmark_profile = benchmark_profile_for_job(job)
    task_scores: list[dict[str, Any]] = []
    for task, metrics in results.items():
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
    category_scores = _category_scores(task_scores)
    expected_tasks = set(profile_expected_tasks(benchmark_profile))
    scored_tasks = {str(task["task"]) for task in task_scores}
    profile_complete = expected_tasks.issubset(scored_tasks) if expected_tasks else None
    profile_incomplete = profile_complete is not None and not profile_complete
    overall_score = _balanced_overall_score(category_scores)
    if profile_incomplete:
        overall_score = None
    provider_backend = _provider_backend(job, model_metadata)
    raw_tasks = job.get("tasks")
    requested_task_count = len(raw_tasks) if isinstance(raw_tasks, list) else None
    result_task_count = len(results)
    status = job.get("status")
    partial = (
        status != "succeeded"
        or (
            requested_task_count is not None
            and result_task_count < requested_task_count
        )
        or profile_incomplete
    )
    rank_eligible = (
        not bool(benchmark_profile.get("custom"))
        and not partial
        and bool(profile_complete)
        and overall_score is not None
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
        "status": status,
        "partial": partial,
        "result_task_count": result_task_count,
        "requested_task_count": requested_task_count,
        "limit": config.get("limit"),
        "total_evaluation_time_seconds": result_json.get(
            "total_evaluation_time_seconds"
        ),
        "generation_tok_s": telemetry.get("generation_tok_s")
        or telemetry.get("probe_generation_tok_s"),
        "prompt_tok_s": telemetry.get("prompt_tok_s")
        or telemetry.get("probe_prompt_tok_s"),
        "ttft_s": telemetry.get("ttft_s"),
        "benchmark_profile": benchmark_profile,
        "score_method": SCORE_METHOD,
        "profile_complete": profile_complete,
        "rank_eligible": rank_eligible,
        "overall_score": overall_score,
        "category_scores": category_scores,
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
        values = [
            score
            for score in (_finite_float(task.get("score")) for task in tasks)
            if score is not None
        ]
        if not values:
            continue
        scores.append(
            {
                "category": category,
                "score": sum(values) / len(values),
                "tasks": sorted(str(task["task"]) for task in tasks),
            }
        )
    return scores


def _balanced_overall_score(category_scores: list[dict[str, Any]]) -> float | None:
    by_category = {
        str(category.get("category")): _finite_float(category.get("score"))
        for category in category_scores
    }
    values = [
        score
        for category in BALANCED_SCORE_CATEGORIES
        if (score := by_category.get(category)) is not None
    ]
    return sum(values) / len(values) if values else None


def _task_category(task: str) -> str:
    if task in _TASK_CATEGORIES:
        return _TASK_CATEGORIES[task]
    normalized = task.lower()
    for category, needles in _TASK_CATEGORY_PATTERNS:
        if any(needle in normalized for needle in needles):
            return category
    return "Other"


def _scored_metrics(task: str, metrics: dict[str, Any]) -> list[tuple[str, float]]:
    numeric: dict[str, float] = {}
    for metric, value in metrics.items():
        numeric_value = _finite_float(value)
        if _is_numeric_metric(metric, value) and numeric_value is not None:
            numeric[metric] = numeric_value
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
    return _finite_float(value) is not None


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
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid result JSON: {path}") from exc
    return payload if isinstance(payload, dict) else {}


def find_result_files(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    if not root.exists():
        return []
    return sorted(root.glob("**/results_*.json"), key=lambda path: path.stat().st_mtime)
