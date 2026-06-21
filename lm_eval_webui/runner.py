"""Command construction for lm-eval benchmark jobs."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .lemonade import DEFAULT_OPENAI_BASE_URL, openai_api_url

DEFAULT_LEMONADE_BASE_URL = DEFAULT_OPENAI_BASE_URL
DEFAULT_LM_EVAL_PYTHON_CANDIDATES = (
    "/home/iain/.venv/lm-eval/bin/python",
    "/home/iain/.virtualenvs/lm-eval/bin/python",
    "/home/iain/venvs/lm-eval/bin/python",
)


@dataclass(slots=True)
class EvalRequest:
    model_id: str
    tasks: list[str]
    output_path: str
    lm_eval_python: str | None = None
    openai_base_url: str | None = None
    lemonade_base_url: str | None = None
    backend: str = "openai-compatible-chat-completions"
    limit: str | None = None
    num_fewshot: int | None = None
    batch_size: str = "1"
    max_gen_toks: int = 256
    num_concurrent: int = 1
    timeout: int = 300
    apply_chat_template: bool = True
    fewshot_as_multiturn: bool = False
    log_samples: bool = False
    predict_only: bool = False
    telemetry_path: str | None = None
    llamacpp_backend: str | None = None
    extra_args: list[str] | None = None


def find_lm_eval_python(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if env_python := os.environ.get("LMEVAL_PYTHON"):
        return env_python
    for candidate in DEFAULT_LM_EVAL_PYTHON_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return shutil.which("python") or "python"


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def build_eval_command(
    request: EvalRequest, project_root: str | Path | None = None
) -> tuple[list[str], dict[str, str]]:
    if not request.tasks:
        raise ValueError("At least one task is required")

    root = Path(project_root) if project_root is not None else Path.cwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(root)
        if not env.get("PYTHONPATH")
        else f"{root}{os.pathsep}{env['PYTHONPATH']}"
    )

    base_url = (
        request.openai_base_url or request.lemonade_base_url or DEFAULT_OPENAI_BASE_URL
    )
    model_args = [
        f"model={request.model_id}",
        f"base_url={openai_api_url(base_url, '/chat/completions')}",
        "tokenizer_backend=None",
        "tokenized_requests=False",
        f"num_concurrent={request.num_concurrent}",
        f"timeout={request.timeout}",
        f"max_gen_toks={request.max_gen_toks}",
        "stream_responses=True",
    ]
    if request.telemetry_path:
        model_args.append(f"telemetry_path={request.telemetry_path}")
    if request.llamacpp_backend:
        model_args.append(f"llamacpp_backend={request.llamacpp_backend}")

    command = [
        find_lm_eval_python(request.lm_eval_python),
        "-m",
        "lm_eval_webui.lm_eval_runner",
        "run",
        "--model",
        request.backend,
        "--model_args",
        *model_args,
        "--tasks",
        *request.tasks,
        "--output_path",
        request.output_path,
        "--batch_size",
        request.batch_size,
    ]
    if request.limit not in (None, ""):
        command.extend(["--limit", str(request.limit)])
    if request.num_fewshot is not None:
        command.extend(["--num_fewshot", str(request.num_fewshot)])
    if request.apply_chat_template:
        command.append("--apply_chat_template")
        command.extend(
            ["--fewshot_as_multiturn", _bool_arg(request.fewshot_as_multiturn)]
        )
    if request.log_samples:
        command.append("--log_samples")
    if request.predict_only:
        command.append("--predict_only")
    if request.extra_args:
        command.extend(request.extra_args)
    return command, env
