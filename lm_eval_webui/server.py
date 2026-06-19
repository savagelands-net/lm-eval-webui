"""Stdlib HTTP server for the lm-eval WebUI."""

from __future__ import annotations

import importlib.util
import json
import mimetypes
import re
import subprocess
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .jobs import JobManager
from .lemonade import DEFAULT_LEMONADE_BASE_URL, fetch_models
from .runner import find_lm_eval_python
from .telemetry import probe_lemonade_chat_telemetry

COMMON_TASKS = [
    {
        "name": "gsm8k",
        "description": "Grade-school math, generate_until",
        "compatibility": "compatible",
    },
    {
        "name": "ifeval",
        "description": "Instruction following, generate_until",
        "compatibility": "compatible",
    },
    {
        "name": "truthfulqa_gen",
        "description": "TruthfulQA generation",
        "compatibility": "compatible",
    },
    {
        "name": "bbh_cot_zeroshot",
        "description": "BIG-Bench Hard CoT generation group",
        "compatibility": "compatible",
    },
]
_OUTPUT_TYPE_RE = re.compile(
    r"^\s*output_type\s*:\s*['\"]?([A-Za-z0-9_-]+)", re.MULTILINE
)


def write_response(
    handler: BaseHTTPRequestHandler,
    status: int | HTTPStatus,
    content_type: str,
    body: bytes,
) -> None:
    """Write an HTTP response, ignoring client disconnects during any phase."""

    try:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except BrokenPipeError:
        return


def safe_static_path(static_root: str | Path, request_path: str) -> Path | None:
    relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
    root = Path(static_root).resolve()
    file_path = (root / relative).resolve()
    try:
        file_path.relative_to(root)
    except ValueError:
        return None
    if not file_path.exists() or not file_path.is_file():
        return None
    return file_path


def _merge_tasks(
    preferred: list[dict[str, str]], discovered: list[dict[str, str]]
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for task in [*preferred, *discovered]:
        name = task.get("name", "").strip()
        if not name or name in seen:
            continue
        merged.append(
            {
                "name": name,
                "description": task.get("description", ""),
                "compatibility": task.get("compatibility", "unknown"),
            }
        )
        seen.add(name)
    return merged


def load_available_tasks(
    lm_eval_python: str | None = None,
    run_command: Callable[..., Any] = subprocess.run,
    config_reader: Callable[[str], str | None] | None = None,
) -> list[dict[str, str]]:
    python = find_lm_eval_python(lm_eval_python)
    package_root = find_lm_eval_package_root(python)
    read_config = config_reader or (
        lambda config_path: read_lm_eval_config(config_path, package_root)
    )
    try:
        completed = run_command(
            [python, "-m", "lm_eval", "ls", "tasks"],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return COMMON_TASKS
    if getattr(completed, "returncode", 0) != 0:
        return COMMON_TASKS
    discovered = parse_lm_eval_task_table(getattr(completed, "stdout", ""))
    discovered = [annotate_task_compatibility(task, read_config) for task in discovered]
    return _merge_tasks(COMMON_TASKS, discovered) if discovered else COMMON_TASKS


def parse_lm_eval_task_table(output: str) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= {"|", "-"}:
            continue
        columns = [column.strip() for column in stripped.strip("|").split("|")]
        if not columns or columns[0] in {"Group", ""}:
            continue
        config_path = columns[1] if len(columns) > 1 else ""
        tasks.append(
            {"name": columns[0], "description": config_path, "config_path": config_path}
        )
    return tasks


def annotate_task_compatibility(
    task: dict[str, str], config_reader: Callable[[str], str | None]
) -> dict[str, str]:
    config_path = task.get("config_path") or task.get("description", "")
    config_text = config_reader(config_path) if config_path else None
    output_type = task_output_type(config_text or "")
    if output_type == "generate_until":
        compatibility = "compatible"
    elif output_type:
        compatibility = "incompatible"
    else:
        compatibility = "unknown"
    return {**task, "compatibility": compatibility}


def task_output_type(config_text: str) -> str | None:
    match = _OUTPUT_TYPE_RE.search(config_text)
    return match.group(1) if match else None


def find_lm_eval_package_root(lm_eval_python: str) -> Path | None:
    spec = importlib.util.find_spec("lm_eval")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin).parent
    try:
        completed = subprocess.run(
            [
                lm_eval_python,
                "-c",
                "import lm_eval, pathlib; print(pathlib.Path(lm_eval.__file__).parent)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    package_root = Path(completed.stdout.strip())
    return package_root if package_root.exists() else None


def read_lm_eval_config(
    config_path: str, package_root: str | Path | None = None
) -> str | None:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.is_absolute():
        root = (
            Path(package_root)
            if package_root is not None
            else find_lm_eval_package_root("python")
        )
        if root is None:
            return None
        parts = path.parts
        path = (
            root.joinpath(*parts[1:])
            if parts and parts[0] == "lm_eval"
            else root / path
        )
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def make_handler(
    manager: JobManager,
    static_dir: str | Path,
    lemonade_base_url: str = DEFAULT_LEMONADE_BASE_URL,
):
    static_root = Path(static_dir)

    class WebUIHandler(BaseHTTPRequestHandler):
        server_version = "lm-eval-webui/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._json(
                    {
                        "ok": True,
                        "lm_eval_python": find_lm_eval_python(manager.lm_eval_python),
                    }
                )
            elif parsed.path == "/api/models":
                self._handle_models(parsed.query)
            elif parsed.path == "/api/tasks":
                self._json({"tasks": load_available_tasks(manager.lm_eval_python)})
            elif parsed.path == "/api/jobs":
                self._json({"jobs": manager.list_jobs()})
            elif parsed.path.startswith("/api/jobs/"):
                self._handle_job_get(parsed.path)
            elif parsed.path == "/api/results":
                self._json(
                    {
                        "rows": manager.result_rows(),
                        "leaderboard": manager.leaderboard_entries(),
                    }
                )
            else:
                self._serve_static(parsed.path)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/jobs":
                try:
                    jobs = manager.create_jobs(self._read_json())
                except ValueError as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._json({"jobs": jobs}, HTTPStatus.CREATED)
            elif parsed.path == "/api/jobs/clear-failed":
                cleared = manager.clear_failed_jobs()
                self._json({"cleared": cleared, "jobs": manager.list_jobs()})
            elif parsed.path == "/api/jobs/clear":
                payload = self._read_json()
                job_ids = payload.get("job_ids") or []
                if isinstance(job_ids, str):
                    job_ids = [job_ids]
                cleared = manager.clear_jobs([str(job_id) for job_id in job_ids])
                self._json({"cleared": cleared, "jobs": manager.list_jobs()})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _handle_models(self, query: str) -> None:
            params = parse_qs(query)
            base_url = params.get("base_url", [lemonade_base_url])[0]
            try:
                models = fetch_models(base_url=base_url)
            except Exception as exc:  # pragma: no cover
                self._json({"models": [], "error": str(exc)}, HTTPStatus.BAD_GATEWAY)
                return
            self._json({"models": models})

        def _handle_job_get(self, path: str) -> None:
            parts = path.strip("/").split("/")
            if len(parts) == 3:
                try:
                    job = manager.get_job(parts[2])
                except FileNotFoundError:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                job["log_tail"] = manager.get_log(parts[2])
                self._json({"job": job})
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _json(
            self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            write_response(self, status, "application/json; charset=utf-8", body)

        def _serve_static(self, path: str) -> None:
            file_path = safe_static_path(static_root, path)
            if file_path is None:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            content = file_path.read_bytes()
            write_response(
                self,
                HTTPStatus.OK,
                mimetypes.guess_type(str(file_path))[0] or "application/octet-stream",
                content,
            )

    return WebUIHandler


def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    data_dir: str | Path = "data",
    static_dir: str | Path = "static",
    lemonade_base_url: str = DEFAULT_LEMONADE_BASE_URL,
    lm_eval_python: str | None = None,
    max_concurrent_jobs: int = 1,
) -> None:
    manager = JobManager(
        data_dir=data_dir,
        project_root=Path.cwd(),
        lm_eval_python=lm_eval_python,
        lemonade_base_url=lemonade_base_url,
        telemetry_probe=probe_lemonade_chat_telemetry,
        max_concurrent_jobs=max_concurrent_jobs,
    )
    handler = make_handler(manager, static_dir, lemonade_base_url)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Serving lm-eval WebUI at http://{host}:{port}")
    httpd.serve_forever()
