"""SWE Mini / pi-bench integration helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lemonade import DEFAULT_OPENAI_BASE_URL, normalize_openai_base_url


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pi_bench_dir(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else repo_root()
    return Path(os.environ.get("PI_BENCH_DIR", root / "third_party" / "pi-bench"))


DEFAULT_PI_BENCH_DIR = default_pi_bench_dir()
DEFAULT_SWE_MINI_PLATFORM = "lemonade-swe"
DEFAULT_SWE_MINI_JUDGE_PROVIDER = "lemonade"
DEFAULT_SWE_MINI_JUDGE_MODEL_ID = "gpt-oss-120b-mxfp-GGUF"
DEFAULT_SWE_MINI_JUDGE_MODEL = (
    f"{DEFAULT_SWE_MINI_JUDGE_PROVIDER}/{DEFAULT_SWE_MINI_JUDGE_MODEL_ID}"
)
SWE_MINI_SUITE = "swe_mini"
WEBUI_TASKSET_DIR = ".webui-tasksets"
LAUNCH_CWD_ENV = "LMEVAL_WEBUI_LAUNCH_CWD"
SWE_OUTPUT_ENV = "SWE_MINI_OUTPUT_PATH"


def normalize_swe_mini_judge_model(
    judge_model: str | None,
    provider: str = DEFAULT_SWE_MINI_JUDGE_PROVIDER,
) -> str:
    """Return a provider-qualified Lemonade judge model name for pi-bench."""

    raw_model = str(judge_model or DEFAULT_SWE_MINI_JUDGE_MODEL_ID).strip()
    if not raw_model:
        raw_model = DEFAULT_SWE_MINI_JUDGE_MODEL_ID
    prefix = f"{provider}/"
    if raw_model.startswith(prefix):
        return raw_model
    return f"{prefix}{raw_model}"


def swe_mini_judge_model_id(
    judge_model: str | None,
    provider: str = DEFAULT_SWE_MINI_JUDGE_PROVIDER,
) -> str:
    """Return the Lemonade model id portion of a provider-qualified judge."""

    qualified = normalize_swe_mini_judge_model(judge_model, provider)
    prefix = f"{provider}/"
    return qualified[len(prefix) :] if qualified.startswith(prefix) else qualified


@dataclass(slots=True)
class SweMiniRequest:
    model_id: str
    task_target: str
    output_path: str
    pi_bench_dir: str | Path = DEFAULT_PI_BENCH_DIR
    project_root: str | Path = repo_root()
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    provider: str = "lemonade"
    judge_model: str = DEFAULT_SWE_MINI_JUDGE_MODEL
    platform: str = DEFAULT_SWE_MINI_PLATFORM
    model_tag: str | None = None
    timeout_minutes: int = 30
    pass_count: int = 1
    context_window: int | None = None
    extra_args: list[str] | None = None
    models_json_path: str | Path | None = None


def write_swe_mini_models_json(
    output_dir: str | Path,
    base_url: str,
    model_id: str,
    context_window: int | None = None,
    judge_model_id: str | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models_path = output / "models.json"
    model_ids: list[str] = []
    for candidate in (model_id, judge_model_id):
        candidate_id = str(candidate or "").strip()
        if candidate_id and candidate_id not in model_ids:
            model_ids.append(candidate_id)
    payload = {
        "providers": {
            "lemonade": {
                "name": "Lemonade",
                "baseUrl": normalize_openai_base_url(base_url),
                "api": "openai-completions",
                "apiKey": "lemonade",
                "compat": {
                    "supportsStore": False,
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsUsageInStreaming": False,
                    "maxTokensField": "max_tokens",
                    "supportsStrictMode": False,
                },
                "models": [
                    _lemonade_model_entry(model, context_window) for model in model_ids
                ],
            }
        }
    }
    models_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return models_path


def _lemonade_model_entry(
    model_id: str, context_window: int | None = None
) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": f"{model_id} (Lemonade)",
        "reasoning": False,
        "input": ["text"],
        "contextWindow": context_window or 131072,
        "maxTokens": 65536,
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
        },
    }


def swe_mini_output_path(
    model_id: str,
    model_tag: str | None,
    platform: str | None = DEFAULT_SWE_MINI_PLATFORM,
    pi_bench_dir: str | Path = DEFAULT_PI_BENCH_DIR,
) -> Path:
    """Return the pi-bench output directory for a provider/model/tag run."""

    base_name = f"{str(model_id).replace('/', '_')}_results"
    if model_tag:
        base_name = base_name.replace("_results", f"-{model_tag}_results")
    root = Path(pi_bench_dir)
    if platform:
        return root / "benchmark_results" / platform / base_name
    return root / base_name


def build_swe_mini_command(request: SweMiniRequest) -> tuple[list[str], dict[str, str]]:
    """Build a pi-bench SWE Mini command and non-secret launch environment."""

    pi_bench_dir = Path(request.pi_bench_dir)
    project_root = Path(request.project_root)
    wrapper = project_root / "scripts" / "run-swe-mini.sh"
    judge_model = normalize_swe_mini_judge_model(request.judge_model)
    models_path = (
        Path(request.models_json_path)
        if request.models_json_path
        else write_swe_mini_models_json(
            Path(request.output_path).parent
            / ".webui-models"
            / (request.model_tag or "default"),
            base_url=request.openai_base_url,
            model_id=request.model_id,
            context_window=request.context_window,
            judge_model_id=swe_mini_judge_model_id(judge_model),
        )
    )
    command = [str(wrapper), request.task_target]

    command.extend(["--provider", request.provider])
    command.extend(["--model", request.model_id])
    if judge_model:
        command.extend(["--judge-model", judge_model])
    if request.platform:
        command.extend(["--platform", request.platform])
    if request.model_tag:
        command.extend(["--model-tag", request.model_tag])
    command.extend(["--timeout", str(_positive_int(request.timeout_minutes, 30))])
    command.extend(["--pass", str(_positive_int(request.pass_count, 1))])
    if request.context_window:
        command.extend(["--context", str(_positive_int(request.context_window, 1))])
    if request.extra_args:
        command.extend([str(arg) for arg in request.extra_args])

    env = os.environ.copy()
    env[LAUNCH_CWD_ENV] = str(project_root)
    env[SWE_OUTPUT_ENV] = str(request.output_path)
    env["PI_BENCH_DIR"] = str(pi_bench_dir)
    env["PI_BENCH_MODELS_JSON"] = str(models_path)
    return command, env


def find_swe_mini_tasks(
    pi_bench_dir: str | Path = DEFAULT_PI_BENCH_DIR,
) -> list[dict[str, Any]]:
    """List pi-bench SWE-bench Verified Mini task files for the WebUI."""

    task_dir = Path(pi_bench_dir) / "tasks" / "verified-mini"
    tasks: list[dict[str, Any]] = []
    for task_file in sorted(task_dir.glob("*.json")):
        try:
            data = _load_json(task_file)
        except (OSError, json.JSONDecodeError):
            continue
        task_id = str(data.get("id") or task_file.stem)
        repo = str(data.get("repo") or "")
        prompt = str(data.get("prompt") or "")
        description_parts = [part for part in (repo, _shorten(prompt)) if part]
        tasks.append(
            {
                "name": task_id,
                "description": " — ".join(description_parts),
                "compatibility": "compatible",
                "category": "Coding / SWE",
                "language_scope": "english",
                "kind": "task",
                "suite": SWE_MINI_SUITE,
                "repo": repo,
            }
        )
    return tasks


def materialize_swe_mini_task_target(
    pi_bench_dir: str | Path,
    task_names: list[str],
    job_id: str,
) -> str:
    """Return a pi-bench-relative task file/dir target for selected task ids."""

    root = Path(pi_bench_dir).resolve()
    task_files = [_resolve_task_file(root, task_name) for task_name in task_names]
    missing = [
        task for task, path in zip(task_names, task_files, strict=False) if path is None
    ]
    if missing:
        raise ValueError(f"Unknown SWE Mini task(s): {', '.join(missing)}")
    resolved_files = [path for path in task_files if path is not None]
    if not resolved_files:
        raise ValueError("At least one SWE Mini task is required")
    if len(resolved_files) == 1:
        return _relative_to_root(root, resolved_files[0])

    taskset_dir = root / WEBUI_TASKSET_DIR / job_id
    if taskset_dir.exists():
        try:
            shutil.rmtree(taskset_dir)
        except OSError as exc:
            raise ValueError(f"Could not reset SWE Mini taskset: {exc}") from exc
    taskset_dir.mkdir(parents=True)
    for task_file in resolved_files:
        shutil.copy2(task_file, taskset_dir / task_file.name)
    return _relative_to_root(root, taskset_dir)


def cleanup_swe_mini_task_target(
    pi_bench_dir: str | Path, task_target: str | None
) -> None:
    if not task_target or not task_target.startswith(f"{WEBUI_TASKSET_DIR}/"):
        return
    target = (Path(pi_bench_dir) / task_target).resolve()
    root = (Path(pi_bench_dir) / WEBUI_TASKSET_DIR).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return
    if target.is_dir():
        try:
            shutil.rmtree(target, ignore_errors=True)
        except OSError:
            return


def find_swe_mini_result_files(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir)
    summary = root / "summary.json"
    if summary.exists():
        return [summary]
    return sorted(root.glob("results-*.json"), key=lambda path: path.stat().st_mtime)


def extract_swe_mini_result_rows(
    job: dict[str, Any], summary_json: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model = str(job.get("model_id") or "unknown")
    for result in _summary_results(summary_json):
        task = str(result.get("task") or "unknown")
        score = _finite_float(result.get("judgeScore"))
        if score is not None:
            rows.append(_swe_row(job, model, task, "judge_score", score))
        duration_ms = _finite_float(result.get("durationMs"))
        if duration_ms is not None:
            rows.append(
                _swe_row(job, model, task, "duration_seconds", duration_ms / 1000)
            )
    return rows


def extract_swe_mini_leaderboard_entry(
    job: dict[str, Any], summary_json: dict[str, Any]
) -> dict[str, Any]:
    results = _summary_results(summary_json)
    total_tasks = _int_or_len(summary_json.get("totalTasks"), results)
    passed_tasks = _int_value(summary_json.get("passedTasks"))
    if passed_tasks is None:
        passed_tasks = sum(1 for result in results if result.get("judgeScore") == 1)
    pass_rate = _finite_float(summary_json.get("passRate"))
    if pass_rate is None:
        pass_rate = passed_tasks / total_tasks if total_tasks else 0.0
    overall_score = pass_rate * 100
    task_scores = [_task_score(result) for result in results]
    raw_swe_options = job.get("swe_options")
    swe_options: dict[str, Any] = (
        raw_swe_options if isinstance(raw_swe_options, dict) else {}
    )
    provider_backend = job.get("provider_backend") or job.get("runtime_backend")
    task_names = [str(score["task"]) for score in task_scores]
    return {
        "suite": SWE_MINI_SUITE,
        "job_id": job.get("id"),
        "model": job.get("model_id") or "unknown",
        "model_id": job.get("model_id") or "unknown",
        "backend": "swe_mini",
        "provider_backend": provider_backend,
        "lemonade_backend": provider_backend,
        "context_window": job.get("context_window"),
        "status": job.get("status"),
        "overall_score": overall_score,
        "category_scores": [
            {"category": "SWE Mini", "score": overall_score, "tasks": task_names}
        ],
        "task_scores": task_scores,
        "total_tasks": total_tasks,
        "passed_tasks": passed_tasks,
        "pass_rate": pass_rate,
        "average_duration_ms": summary_json.get("averageDurationMs"),
        "total_duration_ms": summary_json.get("totalDurationMs"),
        "judge_model": swe_options.get("judge_model"),
        "platform": swe_options.get("platform"),
        "pass_count": swe_options.get("pass_count"),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise exc
    return data if isinstance(data, dict) else {}


def _shorten(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _resolve_task_file(root: Path, task_name: str) -> Path | None:
    verified_dir = root / "tasks" / "verified-mini"
    candidate = verified_dir / f"{task_name}.json"
    if candidate.is_file():
        return candidate.resolve()
    raw = Path(task_name)
    if raw.suffix != ".json":
        return None
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _relative_to_root(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _summary_results(summary_json: dict[str, Any]) -> list[dict[str, Any]]:
    results = summary_json.get("results")
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict)]


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, parsed)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _int_or_len(value: Any, results: list[dict[str, Any]]) -> int:
    int_value = _int_value(value)
    return int_value if int_value is not None else len(results)


def _swe_row(
    job: dict[str, Any], model: str, task: str, metric: str, value: float
) -> dict[str, Any]:
    return {
        "suite": SWE_MINI_SUITE,
        "job_id": job.get("id"),
        "model": model,
        "task": task,
        "metric": metric,
        "value": value,
        "samples": 1,
        "limit": None,
    }


def _task_score(result: dict[str, Any]) -> dict[str, Any]:
    score_value = _finite_float(result.get("judgeScore"))
    attempts = result.get("attempts")
    return {
        "task": str(result.get("task") or "unknown"),
        "category": "SWE Mini",
        "metric": "judgeScore",
        "metrics": ["judgeScore"],
        "values": {"judgeScore": score_value},
        "score": score_value * 100 if score_value is not None else None,
        "duration_ms": result.get("durationMs"),
        "judge_rationale": result.get("judgeRationale"),
        "succeeded_at_attempt": result.get("succeededAtAttempt"),
        "attempts": len(attempts) if isinstance(attempts, list) else None,
    }
