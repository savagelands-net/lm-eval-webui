"""Persistent benchmark job management."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .lemonade import DEFAULT_OPENAI_BASE_URL
from .results import (
    extract_leaderboard_entry,
    extract_result_rows,
    find_result_files,
    load_result_file,
    merge_result_jsons,
)
from .runner import EvalRequest, build_eval_command
from .swe_mini import (  # type: ignore[reportMissingImports]
    DEFAULT_SWE_MINI_JUDGE_MODEL,
    DEFAULT_SWE_MINI_PLATFORM,
    LAUNCH_CWD_ENV,
    SWE_MINI_SUITE,
    SweMiniRequest,
    build_swe_mini_command,
    cleanup_swe_mini_task_target,
    default_pi_bench_dir,
    extract_swe_mini_leaderboard_entry,
    extract_swe_mini_result_rows,
    find_swe_mini_result_files,
    materialize_swe_mini_task_target,
    normalize_swe_mini_judge_model,
    swe_mini_output_path,
)
from .telemetry import aggregate_telemetry_file

Launcher = Callable[[list[str], dict[str, str], Path], int]
TelemetryProbe = Callable[[str, str], dict[str, Any]]
ModelMetadataProbe = Callable[[str, str], dict[str, Any]]
LLAMACPP_BACKENDS = {"system", "vulkan", "rocm"}


def default_launcher(command: list[str], env: dict[str, str], log_path: Path) -> int:
    launch_cwd = env.get(LAUNCH_CWD_ENV) or None
    process_env = {key: value for key, value in env.items() if key != LAUNCH_CWD_ENV}
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=process_env,
            cwd=launch_cwd,
            text=True,
        )
        return process.wait()


class JobManager:
    def __init__(
        self,
        data_dir: str | Path = "data",
        project_root: str | Path | None = None,
        launcher: Launcher = default_launcher,
        run_async: bool = True,
        lm_eval_python: str | None = None,
        openai_base_url: str = DEFAULT_OPENAI_BASE_URL,
        lemonade_base_url: str | None = None,
        telemetry_probe: TelemetryProbe | None = None,
        model_metadata_probe: ModelMetadataProbe | None = None,
        max_concurrent_jobs: int = 1,
        pi_bench_dir: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.launcher = launcher
        self.run_async = run_async
        self.lm_eval_python = lm_eval_python
        self.openai_base_url = (lemonade_base_url or openai_base_url).rstrip("/")
        self.lemonade_base_url = self.openai_base_url
        self.telemetry_probe = telemetry_probe
        self.model_metadata_probe = model_metadata_probe
        self.max_concurrent_jobs = self._int_or_default(max_concurrent_jobs, 1)
        self.pi_bench_dir = (
            Path(pi_bench_dir)
            if pi_bench_dir
            else default_pi_bench_dir(self.project_root)
        )
        self._active_jobs = 0
        self._scheduler = threading.Condition(threading.RLock())
        self.jobs_dir = self.data_dir / "jobs"
        self.logs_dir = self.data_dir / "logs"
        self.runs_dir = self.data_dir / "runs"
        self.telemetry_dir = self.data_dir / "telemetry"
        for directory in (
            self.jobs_dir,
            self.logs_dir,
            self.runs_dir,
            self.telemetry_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create_jobs(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        model_ids = payload.get("model_ids") or payload.get("models") or []
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if not model_ids and payload.get("model_id"):
            model_ids = [payload["model_id"]]
        tasks = payload.get("tasks") or []
        if isinstance(tasks, str):
            tasks = [task for task in tasks.replace(",", " ").split() if task]
        if not model_ids:
            raise ValueError("At least one model is required")
        if not tasks:
            raise ValueError("At least one task is required")
        requested_concurrency = self._optional_int(payload.get("max_concurrent_jobs"))
        if requested_concurrency is not None:
            self.set_max_concurrent_jobs(requested_concurrency)

        created: list[dict[str, Any]] = []
        for model_id in model_ids:
            job = self._create_job(
                str(model_id), [str(task) for task in tasks], payload
            )
            created.append(job)
            if self.run_async:
                threading.Thread(
                    target=self._run_job_with_limit, args=(job["id"],), daemon=True
                ).start()
            else:
                self._run_job(job["id"])
        return created

    def rerun_jobs(self, job_ids: list[str]) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        for job_id in [str(job_id) for job_id in job_ids if str(job_id).strip()]:
            try:
                job = self.get_job(job_id)
            except FileNotFoundError:
                continue
            payload = self._rerun_payload(job)
            if not payload.get("model_ids") or not payload.get("tasks"):
                continue
            created.extend(self.create_jobs(payload))
        return created

    def set_max_concurrent_jobs(self, value: int) -> None:
        with self._scheduler:
            self.max_concurrent_jobs = self._int_or_default(value, 1)
            self._scheduler.notify_all()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [
                self._read_job(path) for path in sorted(self.jobs_dir.glob("*.json"))
            ]
        return sorted(jobs, key=lambda job: job.get("created_at", 0))

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._read_job(self.jobs_dir / f"{job_id}.json")

    def get_log(self, job_id: str, max_chars: int = 20000) -> str:
        log_path = self.logs_dir / f"{job_id}.log"
        if not log_path.exists():
            return ""
        return log_path.read_text(encoding="utf-8", errors="replace")[-max_chars:]

    def result_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for job in self.list_jobs():
            result_files = (
                self._swe_mini_result_files(job)
                if self._job_suite(job) == SWE_MINI_SUITE
                else [Path(path) for path in job.get("result_files", [])]
            )
            for result_file in result_files:
                try:
                    result_json = load_result_file(result_file)
                    if self._job_suite(job) == SWE_MINI_SUITE:
                        rows.extend(extract_swe_mini_result_rows(job, result_json))
                    else:
                        rows.extend(extract_result_rows(job["id"], result_json))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        return rows

    def leaderboard_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for job in self.list_jobs():
            if self._job_suite(job) == SWE_MINI_SUITE:
                for result_file in self._swe_mini_result_files(job):
                    try:
                        result_json = load_result_file(result_file)
                        entries.append(
                            extract_swe_mini_leaderboard_entry(job, result_json)
                        )
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
                continue
            result_jsons = []
            for result_file in job.get("result_files", []):
                try:
                    result_jsons.append(load_result_file(result_file))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
            if result_jsons:
                merged = (
                    merge_result_jsons(result_jsons)
                    if len(result_jsons) > 1
                    else result_jsons[0]
                )
                entries.append(extract_leaderboard_entry(job, merged))
        return sorted(
            entries,
            key=lambda entry: (
                entry.get("overall_score") is not None,
                entry.get("overall_score") or 0,
            ),
            reverse=True,
        )

    def clear_jobs(self, job_ids: list[str]) -> int:
        selected = {str(job_id) for job_id in job_ids if str(job_id).strip()}
        if not selected:
            return 0
        cleared = 0
        with self._lock:
            for job in self.list_jobs():
                if job.get("id") not in selected:
                    continue
                self._remove_job_artifacts(job)
                cleared += 1
        return cleared

    def clear_failed_jobs(self) -> int:
        return self.clear_jobs(
            [job["id"] for job in self.list_jobs() if job.get("status") == "failed"]
        )

    def _create_job(
        self, model_id: str, tasks: list[str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        if self._payload_suite(payload) == SWE_MINI_SUITE:
            return self._create_swe_mini_job(model_id, tasks, payload)
        job_id = uuid.uuid4().hex[:12]
        output_path = self.runs_dir / job_id
        log_path = self.logs_dir / f"{job_id}.log"
        telemetry_path = self.telemetry_dir / f"{job_id}.jsonl"
        backend = payload.get("backend", "openai-compatible-chat-completions")
        openai_base_url = payload.get(
            "openai_base_url", payload.get("lemonade_base_url", self.openai_base_url)
        )
        llamacpp_backend = self._optional_llamacpp_backend(
            payload.get("llamacpp_backend")
        )
        task_batch_size = self._positive_optional_int(payload.get("task_batch_size"))
        request = EvalRequest(
            model_id=model_id,
            tasks=tasks,
            output_path=str(output_path),
            lm_eval_python=payload.get("lm_eval_python") or self.lm_eval_python,
            openai_base_url=openai_base_url,
            backend=backend,
            limit=payload.get("limit"),
            num_fewshot=self._optional_int(payload.get("num_fewshot")),
            batch_size=str(payload.get("batch_size", "1")),
            max_gen_toks=self._int_or_default(payload.get("max_gen_toks"), 256),
            num_concurrent=self._int_or_default(payload.get("num_concurrent"), 1),
            timeout=self._int_or_default(payload.get("timeout"), 300),
            apply_chat_template=bool(payload.get("apply_chat_template", True)),
            fewshot_as_multiturn=bool(payload.get("fewshot_as_multiturn", False)),
            log_samples=bool(payload.get("log_samples", False)),
            predict_only=bool(payload.get("predict_only", False)),
            telemetry_path=str(telemetry_path),
            llamacpp_backend=llamacpp_backend,
        )
        command, env = build_eval_command(request, self.project_root)
        now = time.time()
        eval_options = self._eval_options(request, task_batch_size=task_batch_size)
        job = {
            "id": job_id,
            "model_id": model_id,
            "tasks": tasks,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "command": command,
            "output_path": str(output_path),
            "log_path": str(log_path),
            "telemetry_path": str(telemetry_path),
            "openai_base_url": str(openai_base_url).rstrip("/"),
            "lemonade_base_url": str(openai_base_url).rstrip("/"),
            "backend": backend,
            "eval_options": eval_options,
            "telemetry": {},
            "result_files": [],
            "returncode": None,
            "error": None,
            "_env": env,
        }
        if task_batch_size:
            job["task_batch_size"] = task_batch_size
        if payload.get("rerun_of"):
            job["rerun_of"] = str(payload["rerun_of"])
        if llamacpp_backend:
            job.update(
                {
                    "requested_llamacpp_backend": llamacpp_backend,
                    "provider_backend": llamacpp_backend,
                    "lemonade_backend": llamacpp_backend,
                    "runtime_backend": llamacpp_backend,
                    "llamacpp_backend": llamacpp_backend,
                }
            )
        self._write_job(job)
        return self._public_job(job)

    def _create_swe_mini_job(
        self, model_id: str, tasks: list[str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        log_path = self.logs_dir / f"{job_id}.log"
        platform = str(payload.get("platform") or DEFAULT_SWE_MINI_PLATFORM)
        output_path = swe_mini_output_path(
            model_id, job_id, platform, pi_bench_dir=self.pi_bench_dir
        )
        task_target = materialize_swe_mini_task_target(self.pi_bench_dir, tasks, job_id)
        judge_model = normalize_swe_mini_judge_model(
            str(payload.get("judge_model") or DEFAULT_SWE_MINI_JUDGE_MODEL)
        )
        timeout_minutes = self._int_or_default(
            payload.get("swe_timeout", payload.get("timeout_minutes")), 30
        )
        pass_count = self._int_or_default(payload.get("pass_count"), 1)
        context_window = self._optional_int(payload.get("context_window"))
        provider = str(payload.get("swe_provider") or "lemonade")
        openai_base_url = payload.get(
            "openai_base_url", payload.get("lemonade_base_url", self.openai_base_url)
        )
        request = SweMiniRequest(
            model_id=model_id,
            task_target=task_target,
            output_path=str(output_path),
            pi_bench_dir=self.pi_bench_dir,
            project_root=self.project_root,
            openai_base_url=str(openai_base_url),
            provider=provider,
            judge_model=judge_model,
            platform=platform,
            model_tag=job_id,
            timeout_minutes=timeout_minutes,
            pass_count=pass_count,
            context_window=context_window,
        )
        command, env = build_swe_mini_command(request)
        now = time.time()
        llamacpp_backend = self._optional_llamacpp_backend(
            payload.get("llamacpp_backend")
        )
        swe_options = {
            "provider": provider,
            "judge_model": judge_model,
            "platform": platform,
            "model_tag": job_id,
            "timeout_minutes": timeout_minutes,
            "pass_count": pass_count,
            "context_window": context_window,
            "task_target": task_target,
            "pi_bench_dir": str(self.pi_bench_dir),
            "openai_base_url": str(openai_base_url),
        }
        job: dict[str, Any] = {
            "id": job_id,
            "suite": SWE_MINI_SUITE,
            "model_id": model_id,
            "tasks": tasks,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "command": command,
            "output_path": str(output_path),
            "log_path": str(log_path),
            "openai_base_url": str(openai_base_url).rstrip("/"),
            "lemonade_base_url": str(openai_base_url).rstrip("/"),
            "backend": SWE_MINI_SUITE,
            "swe_options": swe_options,
            "telemetry": {},
            "result_files": [],
            "returncode": None,
            "error": None,
            "_env": env,
        }
        if payload.get("rerun_of"):
            job["rerun_of"] = str(payload["rerun_of"])
        if context_window:
            job["context_window"] = context_window
        if llamacpp_backend:
            job.update(
                {
                    "requested_llamacpp_backend": llamacpp_backend,
                    "provider_backend": llamacpp_backend,
                    "lemonade_backend": llamacpp_backend,
                    "runtime_backend": llamacpp_backend,
                    "llamacpp_backend": llamacpp_backend,
                }
            )
        self._write_job(job)
        return self._public_job(job)

    def _run_job_with_limit(self, job_id: str) -> None:
        with self._scheduler:
            while self._active_jobs >= self.max_concurrent_jobs:
                self._scheduler.wait()
            self._active_jobs += 1
        try:
            self._run_job(job_id)
        except FileNotFoundError:
            return
        finally:
            with self._scheduler:
                self._active_jobs = max(0, self._active_jobs - 1)
                self._scheduler.notify_all()

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        env = job.pop("_env", None) or self._launch_env_for_job(job)
        job["status"] = "running"
        job["updated_at"] = time.time()
        self._write_job(job)
        try:
            if self._job_suite(job) == SWE_MINI_SUITE:
                returncode = self.launcher(job["command"], env, Path(job["log_path"]))
            else:
                returncode = self._run_lm_eval_job(job, env)
            job["returncode"] = returncode
            if self._job_suite(job) == SWE_MINI_SUITE:
                output_path = self._persist_swe_mini_results(job)
                job["result_files"] = [
                    str(path) for path in find_swe_mini_result_files(output_path)
                ]
            else:
                job["result_files"] = [
                    str(path) for path in find_result_files(job["output_path"])
                ]
            job["telemetry"] = self._collect_telemetry(job, returncode)
            job["model_metadata"] = self._collect_model_metadata(job, returncode)
            self._apply_model_metadata(job)
            job["status"] = "succeeded" if returncode == 0 else "failed"
        except Exception as exc:  # pragma: no cover
            job["status"] = "failed"
            job["error"] = str(exc)
        finally:
            if self._job_suite(job) == SWE_MINI_SUITE:
                raw_options = job.get("swe_options")
                options = raw_options if isinstance(raw_options, dict) else {}
                cleanup_swe_mini_task_target(
                    options.get("pi_bench_dir", self.pi_bench_dir),
                    options.get("task_target"),
                )
            job["updated_at"] = time.time()
            self._write_job(job)

    def _run_lm_eval_job(self, job: dict[str, Any], env: dict[str, str]) -> int:
        batch_size = self._task_batch_size_for_job(job)
        tasks = [str(task) for task in job.get("tasks") or []]
        if batch_size is None or len(tasks) <= batch_size:
            return self.launcher(job["command"], env, Path(job["log_path"]))
        return self._run_lm_eval_task_batches(job, tasks, batch_size)

    def _run_lm_eval_task_batches(
        self, job: dict[str, Any], tasks: list[str], batch_size: int
    ) -> int:
        batches = [
            tasks[index : index + batch_size]
            for index in range(0, len(tasks), batch_size)
        ]
        total = len(batches)
        log_path = Path(job["log_path"])
        job["batch_progress"] = {
            "task_batch_size": batch_size,
            "total": total,
            "completed": 0,
            "current": None,
            "failed": None,
        }
        self._write_job(job)
        for index, batch in enumerate(batches, start=1):
            batch_output_path = (
                Path(job["output_path"]) / f"batch_{index:03d}_of_{total:03d}"
            )
            request = self._eval_request_from_job(
                job, tasks=batch, output_path=str(batch_output_path)
            )
            command, batch_env = build_eval_command(request, self.project_root)
            job["batch_progress"] = {
                "task_batch_size": batch_size,
                "total": total,
                "completed": index - 1,
                "current": index,
                "failed": None,
            }
            self._write_job(job)
            self._append_log(
                log_path,
                "\n"
                f"=== lm-eval task batch {index}/{total} "
                f"({len(batch)} task{'s' if len(batch) != 1 else ''}) ===\n"
                f"$ {shlex.join(command)}\n",
            )
            returncode = self.launcher(command, batch_env, log_path)
            if returncode != 0:
                job["batch_progress"] = {
                    "task_batch_size": batch_size,
                    "total": total,
                    "completed": index - 1,
                    "current": None,
                    "failed": index,
                }
                self._write_job(job)
                return returncode
            job["batch_progress"] = {
                "task_batch_size": batch_size,
                "total": total,
                "completed": index,
                "current": None,
                "failed": None,
            }
            self._write_job(job)
        return 0

    @staticmethod
    def _append_log(log_path: Path, message: str) -> None:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message)

    def _read_job(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise exc
        return data if isinstance(data, dict) else {}

    def _launch_env_for_job(self, job: dict[str, Any]) -> dict[str, str]:
        if self._job_suite(job) == SWE_MINI_SUITE:
            request = self._swe_request_from_job(job)
            return build_swe_mini_command(request)[1]
        return build_eval_command(self._eval_request_from_job(job), self.project_root)[
            1
        ]

    def _eval_request_from_job(
        self,
        job: dict[str, Any],
        tasks: list[str] | None = None,
        output_path: str | None = None,
    ) -> EvalRequest:
        raw_options = job.get("eval_options")
        options = raw_options if isinstance(raw_options, dict) else {}
        return EvalRequest(
            model_id=str(job.get("model_id") or ""),
            tasks=list(tasks if tasks is not None else job.get("tasks") or []),
            output_path=str(output_path or job.get("output_path") or ""),
            lm_eval_python=options.get("lm_eval_python") or self.lm_eval_python,
            openai_base_url=job.get("openai_base_url") or job.get("lemonade_base_url"),
            backend=job.get("backend", "openai-compatible-chat-completions"),
            limit=options.get("limit"),
            num_fewshot=self._optional_int(options.get("num_fewshot")),
            batch_size=str(options.get("batch_size", "1")),
            max_gen_toks=self._int_or_default(options.get("max_gen_toks"), 256),
            num_concurrent=self._int_or_default(options.get("num_concurrent"), 1),
            timeout=self._int_or_default(options.get("timeout"), 300),
            apply_chat_template=self._optional_bool(
                options.get("apply_chat_template"), True
            ),
            fewshot_as_multiturn=self._optional_bool(
                options.get("fewshot_as_multiturn"), False
            ),
            log_samples=self._optional_bool(options.get("log_samples"), False),
            predict_only=self._optional_bool(options.get("predict_only"), False),
            telemetry_path=job.get("telemetry_path"),
            llamacpp_backend=job.get("requested_llamacpp_backend")
            or job.get("llamacpp_backend"),
        )

    def _swe_request_from_job(self, job: dict[str, Any]) -> SweMiniRequest:
        raw_options = job.get("swe_options")
        options = raw_options if isinstance(raw_options, dict) else {}
        return SweMiniRequest(
            model_id=str(job.get("model_id") or ""),
            task_target=str(options.get("task_target") or ""),
            output_path=str(job.get("output_path") or ""),
            pi_bench_dir=options.get("pi_bench_dir") or self.pi_bench_dir,
            project_root=self.project_root,
            openai_base_url=str(
                options.get("openai_base_url")
                or job.get("openai_base_url")
                or self.openai_base_url
            ),
            provider=str(options.get("provider") or "lemonade"),
            judge_model=normalize_swe_mini_judge_model(
                str(options.get("judge_model") or DEFAULT_SWE_MINI_JUDGE_MODEL)
            ),
            platform=str(options.get("platform") or DEFAULT_SWE_MINI_PLATFORM),
            model_tag=str(options.get("model_tag") or job.get("id") or ""),
            timeout_minutes=self._int_or_default(options.get("timeout_minutes"), 30),
            pass_count=self._int_or_default(options.get("pass_count"), 1),
            context_window=self._optional_int(options.get("context_window")),
        )

    @staticmethod
    def _payload_suite(payload: dict[str, Any]) -> str:
        return str(payload.get("suite") or "lm_eval")

    @staticmethod
    def _job_suite(job: dict[str, Any]) -> str:
        return str(job.get("suite") or "lm_eval")

    def _collect_telemetry(
        self, job: dict[str, Any], returncode: int | None
    ) -> dict[str, Any]:
        telemetry = aggregate_telemetry_file(job.get("telemetry_path"))
        if (
            returncode == 0
            and self.telemetry_probe is not None
            and "ttft_s" not in telemetry
        ):
            try:
                probe = self.telemetry_probe(
                    job.get("openai_base_url")
                    or job.get("lemonade_base_url", self.openai_base_url),
                    job["model_id"],
                )
            except Exception as exc:  # pragma: no cover
                telemetry["error"] = str(exc)
            else:
                for key, value in (probe or {}).items():
                    if key == "timings":
                        continue
                    target_key = key if key.startswith("probe_") else f"probe_{key}"
                    telemetry[target_key] = value
                    if key == "ttft_s" and "ttft_s" not in telemetry:
                        telemetry["ttft_s"] = value
        return telemetry

    def _collect_model_metadata(
        self, job: dict[str, Any], returncode: int | None
    ) -> dict[str, Any]:
        if returncode != 0 or self.model_metadata_probe is None:
            existing = job.get("model_metadata")
            return existing if isinstance(existing, dict) else {}
        try:
            metadata = self.model_metadata_probe(
                job.get("openai_base_url")
                or job.get("lemonade_base_url", self.openai_base_url),
                job["model_id"],
            )
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}
        return {
            str(key): value
            for key, value in (metadata or {}).items()
            if value not in (None, "")
        }

    @staticmethod
    def _apply_model_metadata(job: dict[str, Any]) -> None:
        metadata = job.get("model_metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        for key in (
            "runtime_backend",
            "llamacpp_backend",
            "recipe",
            "context_window",
            "device",
            "checkpoint",
        ):
            if metadata.get(key) not in (None, ""):
                job[key] = metadata[key]
        provider_backend = JobManager._provider_backend(job, metadata)
        if provider_backend:
            job["provider_backend"] = provider_backend
            job["lemonade_backend"] = provider_backend
            job.setdefault("runtime_backend", provider_backend)

    @staticmethod
    def _provider_backend(
        job: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> str | None:
        metadata = metadata or {}
        for value in (
            metadata.get("llamacpp_backend"),
            metadata.get("runtime_backend"),
            job.get("llamacpp_backend"),
            job.get("requested_llamacpp_backend"),
            job.get("runtime_backend"),
            job.get("provider_backend"),
            job.get("lemonade_backend"),
        ):
            backend = JobManager._concrete_backend(value)
            if backend:
                return backend
        for recipe in (metadata.get("recipe"), job.get("recipe")):
            backend = JobManager._recipe_backend(recipe)
            if backend:
                return backend
        return JobManager._concrete_backend(job.get("backend"))

    @staticmethod
    def _concrete_backend(value: Any) -> str | None:
        if value in (None, ""):
            return None
        backend = str(value)
        return None if backend == "llamacpp" else backend

    @staticmethod
    def _recipe_backend(recipe: Any) -> str | None:
        if recipe in (None, ""):
            return None
        backend = str(recipe)
        return "system" if backend == "llamacpp" else backend

    @staticmethod
    def _eval_options(
        request: EvalRequest, task_batch_size: int | None = None
    ) -> dict[str, Any]:
        return {
            "lm_eval_python": request.lm_eval_python,
            "limit": request.limit,
            "num_fewshot": request.num_fewshot,
            "batch_size": request.batch_size,
            "max_gen_toks": request.max_gen_toks,
            "num_concurrent": request.num_concurrent,
            "timeout": request.timeout,
            "apply_chat_template": request.apply_chat_template,
            "fewshot_as_multiturn": request.fewshot_as_multiturn,
            "log_samples": request.log_samples,
            "predict_only": request.predict_only,
            "task_batch_size": task_batch_size,
        }

    @staticmethod
    def _rerun_payload(job: dict[str, Any]) -> dict[str, Any]:
        model_id = str(job.get("model_id") or "").strip()
        payload: dict[str, Any] = {
            "suite": JobManager._job_suite(job),
            "model_ids": [model_id] if model_id else [],
            "tasks": list(job.get("tasks") or []),
            "openai_base_url": job.get("openai_base_url")
            or job.get("lemonade_base_url"),
            "rerun_of": job.get("id"),
        }
        if JobManager._job_suite(job) == SWE_MINI_SUITE:
            raw_options = job.get("swe_options")
            options = raw_options if isinstance(raw_options, dict) else {}
            option_map = {
                "judge_model": "judge_model",
                "platform": "platform",
                "pass_count": "pass_count",
                "timeout_minutes": "swe_timeout",
                "context_window": "context_window",
                "provider": "swe_provider",
                "openai_base_url": "openai_base_url",
            }
            for source_key, payload_key in option_map.items():
                if source_key in options:
                    payload[payload_key] = options[source_key]
        else:
            options = job.get("eval_options")
            if not isinstance(options, dict):
                options = {}
            payload["backend"] = job.get(
                "backend", "openai-compatible-chat-completions"
            )
            for key in (
                "lm_eval_python",
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
            ):
                if key in options:
                    payload[key] = options[key]
                elif key in job:
                    payload[key] = job[key]
        llamacpp_backend = job.get("requested_llamacpp_backend") or job.get(
            "llamacpp_backend"
        )
        if llamacpp_backend:
            payload["llamacpp_backend"] = llamacpp_backend
        return payload

    def _swe_mini_result_files(self, job: dict[str, Any]) -> list[Path]:
        result_files = [
            Path(str(path))
            for path in job.get("result_files", [])
            if Path(str(path)).exists()
        ]
        if result_files:
            return result_files
        output_path = self._persist_swe_mini_results(job)
        result_files = find_swe_mini_result_files(output_path)
        if result_files:
            job["result_files"] = [str(path) for path in result_files]
            self._write_job(job)
        return result_files

    def _persist_swe_mini_results(self, job: dict[str, Any]) -> Path:
        """Copy SWE Mini result artifacts from workspace storage into /data/runs."""

        raw_source = str(job.get("output_path") or "").strip()
        persistent_path = self.runs_dir / str(job["id"])
        if not raw_source:
            return persistent_path
        source = Path(raw_source)
        if source.exists() and source not in {Path("."), persistent_path}:
            persistent_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, persistent_path, dirs_exist_ok=True)
            job["pi_bench_output_path"] = str(source)
            job["output_path"] = str(persistent_path)
            raw_options = job.get("swe_options")
            if isinstance(raw_options, dict):
                raw_options["pi_bench_output_path"] = str(source)
        return persistent_path if persistent_path.exists() else source

    def _remove_job_artifacts(self, job: dict[str, Any]) -> None:
        for key in (
            "log_path",
            "output_path",
            "pi_bench_output_path",
            "telemetry_path",
        ):
            raw_path = job.get(key)
            if not raw_path:
                continue
            path = Path(raw_path)
            if path in {Path("."), Path("")}:
                continue
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError:
                continue
        job_path = self.jobs_dir / f"{job['id']}.json"
        if job_path.exists():
            job_path.unlink()

    def _write_job(self, job: dict[str, Any]) -> None:
        job_path = self.jobs_dir / f"{job['id']}.json"
        with self._lock, job_path.open("w", encoding="utf-8") as handle:
            json.dump(self._public_job(job), handle, indent=2, sort_keys=True)

    @staticmethod
    def _public_job(job: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in job.items() if not key.startswith("_")}

    def _task_batch_size_for_job(self, job: dict[str, Any]) -> int | None:
        raw_options = job.get("eval_options")
        options = raw_options if isinstance(raw_options, dict) else {}
        return self._positive_optional_int(
            options.get("task_batch_size", job.get("task_batch_size"))
        )

    @staticmethod
    def _positive_optional_int(value: Any) -> int | None:
        parsed = JobManager._optional_int(value)
        return parsed if parsed is not None and parsed > 0 else None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _int_or_default(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            parsed = default
        return max(1, parsed)

    @staticmethod
    def _optional_bool(value: Any, default: bool) -> bool:
        if value in (None, ""):
            return default
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _optional_llamacpp_backend(value: Any) -> str | None:
        if value in (None, ""):
            return None
        normalized = str(value).strip().lower()
        if normalized in {"auto", "default"}:
            return None
        if normalized not in LLAMACPP_BACKENDS:
            allowed = ", ".join(sorted(LLAMACPP_BACKENDS))
            raise ValueError(f"llama.cpp backend must be one of: {allowed}")
        return normalized
