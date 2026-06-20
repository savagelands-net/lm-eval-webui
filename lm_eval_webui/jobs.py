"""Persistent benchmark job management."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .results import (
    extract_leaderboard_entry,
    extract_result_rows,
    find_result_files,
    load_result_file,
)
from .runner import EvalRequest, build_eval_command
from .telemetry import aggregate_telemetry_file

Launcher = Callable[[list[str], dict[str, str], Path], int]
TelemetryProbe = Callable[[str, str], dict[str, Any]]


def default_launcher(command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
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
        openai_base_url: str = "https://llm.savagelands.net",
        lemonade_base_url: str | None = None,
        telemetry_probe: TelemetryProbe | None = None,
        max_concurrent_jobs: int = 1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.launcher = launcher
        self.run_async = run_async
        self.lm_eval_python = lm_eval_python
        self.openai_base_url = (lemonade_base_url or openai_base_url).rstrip("/")
        self.lemonade_base_url = self.openai_base_url
        self.telemetry_probe = telemetry_probe
        self.max_concurrent_jobs = max(1, int(max_concurrent_jobs))
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

    def set_max_concurrent_jobs(self, value: int) -> None:
        with self._scheduler:
            self.max_concurrent_jobs = max(1, int(value))
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
            for result_file in job.get("result_files", []):
                try:
                    rows.extend(
                        extract_result_rows(job["id"], load_result_file(result_file))
                    )
                except (OSError, json.JSONDecodeError):
                    continue
        return rows

    def leaderboard_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for job in self.list_jobs():
            for result_file in job.get("result_files", []):
                try:
                    entries.append(
                        extract_leaderboard_entry(job, load_result_file(result_file))
                    )
                except (OSError, json.JSONDecodeError):
                    continue
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
        job_id = uuid.uuid4().hex[:12]
        output_path = self.runs_dir / job_id
        log_path = self.logs_dir / f"{job_id}.log"
        telemetry_path = self.telemetry_dir / f"{job_id}.jsonl"
        backend = payload.get("backend", "openai-compatible-chat-completions")
        openai_base_url = payload.get(
            "openai_base_url", payload.get("lemonade_base_url", self.openai_base_url)
        )
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
            max_gen_toks=int(payload.get("max_gen_toks", 256)),
            num_concurrent=int(payload.get("num_concurrent", 1)),
            timeout=int(payload.get("timeout", 300)),
            apply_chat_template=bool(payload.get("apply_chat_template", True)),
            fewshot_as_multiturn=bool(payload.get("fewshot_as_multiturn", False)),
            log_samples=bool(payload.get("log_samples", False)),
            predict_only=bool(payload.get("predict_only", False)),
            telemetry_path=str(telemetry_path),
        )
        command, env = build_eval_command(request, self.project_root)
        now = time.time()
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
            "telemetry": {},
            "result_files": [],
            "returncode": None,
            "error": None,
            "_env": env,
        }
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
        env = (
            job.pop("_env", None)
            or build_eval_command(
                EvalRequest(
                    job["model_id"],
                    job["tasks"],
                    job["output_path"],
                    self.lm_eval_python,
                ),
                self.project_root,
            )[1]
        )
        job["status"] = "running"
        job["updated_at"] = time.time()
        self._write_job(job)
        try:
            returncode = self.launcher(job["command"], env, Path(job["log_path"]))
            job["returncode"] = returncode
            job["result_files"] = [
                str(path) for path in find_result_files(job["output_path"])
            ]
            job["telemetry"] = self._collect_telemetry(job, returncode)
            job["status"] = "succeeded" if returncode == 0 else "failed"
        except Exception as exc:  # pragma: no cover
            job["status"] = "failed"
            job["error"] = str(exc)
        finally:
            job["updated_at"] = time.time()
            self._write_job(job)

    def _read_job(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

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

    def _remove_job_artifacts(self, job: dict[str, Any]) -> None:
        for key in ("log_path", "output_path", "telemetry_path"):
            raw_path = job.get(key)
            if not raw_path:
                continue
            path = Path(raw_path)
            if path in {Path("."), Path("")}:
                continue
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
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

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        return int(value)
