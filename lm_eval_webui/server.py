"""Stdlib HTTP server for the lm-eval WebUI."""

from __future__ import annotations

import importlib.util
import json
import math
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
from .lemonade import DEFAULT_OPENAI_BASE_URL, fetch_loaded_model_metadata, fetch_models
from .runner import find_lm_eval_python
from .telemetry import probe_lemonade_chat_telemetry

COMMON_TASKS = [
    {
        "name": "gsm8k",
        "description": "Grade-school math, generate_until",
        "compatibility": "compatible",
        "category": "Math",
    },
    {
        "name": "ifeval",
        "description": "Instruction following, generate_until",
        "compatibility": "compatible",
        "category": "Instruction Following",
    },
    {
        "name": "truthfulqa_gen",
        "description": "TruthfulQA generation",
        "compatibility": "compatible",
        "category": "Reasoning",
    },
    {
        "name": "bbh_cot_zeroshot",
        "description": "BIG-Bench Hard CoT generation group",
        "compatibility": "compatible",
        "category": "Reasoning",
    },
]
TASK_CATEGORY_PATTERNS = [
    ("Math", ("gsm8k", "math", "aime", "amc", "minerva")),
    (
        "Coding / Structured Output",
        ("json", "schema", "code", "humaneval", "mbpp", "repobench", "longbench_lcc"),
    ),
    ("Instruction Following", ("ifeval", "instruction")),
    ("Reasoning", ("arc", "bbh", "truthful", "mmlu", "hellaswag", "winogrande")),
]
_OUTPUT_TYPE_RE = re.compile(
    r"^\s*output_type\s*:\s*['\"]?([A-Za-z0-9_-]+)", re.MULTILINE
)
_GROUP_RE = re.compile(r"^\s*group\s*:", re.MULTILINE)
_TOP_LEVEL_TASK_LIST_RE = re.compile(r"^task\s*:\s*$")
_DATASET_PATH_RE = re.compile(
    r"^\s*dataset_path\s*:\s*['\"]?([^'\"\n#]+)", re.MULTILINE
)
_INCLUDE_RE = re.compile(
    r"^\s*['\"]?include['\"]?\s*:\s*['\"]?([^'\"\n#\[{]+)", re.MULTILINE
)
_BLEURT_METRIC_RE = re.compile(r"^\s*-?\s*metric\s*:\s*['\"]?bleurt\b", re.MULTILINE)
_UNAVAILABLE_METRIC_RE = re.compile(
    r"^\s*-?\s*metric\s*:\s*['\"]?(?:wer)\b", re.MULTILINE
)
_CODE_EVAL_METRIC_RE = re.compile(
    r"^\s*-?\s*metric\s*:\s*!function\s+utils\.pass_at", re.MULTILINE
)
UNSUPPORTED_DATASET_SCRIPT_PATHS = {
    "EleutherAI/unscramble",
    "kumapo/JAQKET",
    "orange_sum",
    "baber/logiqa2",
    "allenai/qasper",
    "csebuetnlp/xlsum",
}
UNAVAILABLE_DATASET_PATHS = {
    "Rakuten/JGLUE",
    "fixie-ai/endpointing-audio",
    "proxectonos/summarization_gl",
}
GATED_DATASET_PATHS = {"gplsi/cocoteros_va", "gplsi/truthfulqa_va"}
COMPATIBLE_TASK_NAMES = {
    "jsonschema_bench",
    "bigbench_bbq_lite_json_generate_until",
    "bigbench_code_line_description_generate_until",
    "bigbench_codenames_generate_until",
    "bigbench_simple_arithmetic_json_generate_until",
    "bigbench_simple_arithmetic_json_subtasks_generate_until",
    "code2text_go",
    "code2text_java",
    "code2text_javascript",
    "code2text_php",
    "code2text_python",
    "code2text_ruby",
    "code2text",
    "graphwalks_128k",
    "jfinqa",
    "jsonschema_bench_hard",
    "jsonschema_bench_medium",
}
INCOMPATIBLE_TASK_NAMES = {
    "ifeval_ca",
    "ifeval_es",
    "niah_single_1",
    "bigbench_bbq_lite_json_multiple_choice",
    "bigbench_code_line_description_multiple_choice",
    "bigbench_simple_arithmetic_json_multiple_choice_generate_until",
    "bigbench_simple_arithmetic_multiple_targets_json_generate_until",
    "humaneval_64_instruct",
    "humaneval_instruct",
    "humaneval_plus",
    "humaneval_random_span_infilling",
    "humaneval_single_line_infilling",
    "humaneval_single_line_infilling_light",
    "infinitebench_code_debug",
    "infinitebench_code_run",
    "longbench_code_tasks",
    "longbench_code_tasks_e",
    "longbench_lcc",
    "longbench_lcc_e",
    "longbench_repobench-p",
    "longbench_repobench-p_e",
    "longbench2_code",
    "mbpp_plus",
    "mbpp_plus_instruct",
    "toksuite_chinese_code_language_script_switching",
    "toksuite_farsi_code_language_script_switching",
    "toksuite_italian_code_language_script_switching",
    "toksuite_stem_unicode_formatting",
    "toksuite_turkish_code_language_script_switching",
    "graphwalks",
    "graphwalks_1M",
    "meddialog_qsumm",
    "tinyGSM8k",
}
COMPATIBLE_TASK_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^bigbench_.*_generate_until$",
        r"^bbh(?:_|$)",
        r"^truthfulqa-multi_gen_",
        r"^nortruthfulqa_gen_",
        r"^hendrycks_math",
        r"^leaderboard_(?:instruction_following|math_hard)$",
        r"^minerva_math(?:_|$)",
        r"^score_(?:prompt_robustness_math|non_greedy_robustness_math|robustness_math)$",
        r"^score_robustness_mmlu_pro$",
        r"^mmlu_cot_llama(?:_|$)",
        r"^mmlu_prox(?:_lite)?_[a-z]{2}(?:_|$)",
        r"^mmlu_(?:.*_generative(?:_spanish)?|redux_.*_generative)$",
        r"^mmlu_flan_cot_(?:fewshot|zeroshot)(?:_|$)",
        r"^mmlu_flan_n_shot_generative(?:_|$)",
        r"^mmlu_(?:de|es|fr|hi|it|pt|th)_llama(?:_|$)",
        r"^mmlu_llama(?:_|$)",
        r"^mmlu_pro(?:_plus)?(?:_|$)",
        r"^metabench_gsm8k_subset$",
        r"^adr(?:_|$)",
        r"^ntrex(?:_|$|-)",
    )
)
INCOMPATIBLE_TASK_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^bigbench_.*multiple_choice.*",
        r"^truthfulqa.*_(?:mc1|mc2)(?:_|$)",
        r"^truthfulqa(?:$|[-_]multi$|_multilingual$|_gl$)",
        r"^truthfulqa_mc2$",
        r"^nortruthfulqa_mc_",
        r"^global_mmlu_",
        r"^mmmlu(?:_|$)",
        r"^m_mmlu(?:_|$)",
        r"^cmmlu(?:_|$)",
        r"^kmmlu(?:_|$)",
        r"^mmlusr(?:_|$)",
        r"^openai_mmlu(?:_|$)",
        r"^AraDiCE_ArabicMMLU",
        r"^arabicmmlu(?:_|$)",
        r"^darijammlu(?:_|$)",
        r"^egymmlu(?:_|$)",
        r"^afrimmlu",
        r"^uhura[-_]arc[-_]easy(?:_|$)",
        r"^naijarc(?:_|$)",
        r"^ceval-valid_",
        r"^noor_",
        r"^turkishmmlu(?:_|$)",
        r"^tmmluplus(?:_|$)",
        r"^toksuite_math(?:_|$)",
        r"^agieval_.*math",
        r"^arabic_leaderboard_.*(?:arc|hellaswag|mmlu)",
        r"^afrobench_mmlu_tasks$",
        r"^ai2_arc$",
        r"^arc_(?:multilingual|challenge_mt)$",
        r"^hellaswag_multilingual$",
        r"^leaderboard_bbh$",
        r"^libra_complex_reasoning_and_mathematical_problems$",
        r"^metabench_(?:arc|hellaswag|mmlu|truthfulqa|winogrande)_subset$",
        r"^mmlu$",
        r"^mmlu_(?!.*(?:generative|flan_cot|flan_n_shot_generative|llama|pro|prox|cot)).*",
        r"^math_word_problems$",
        r"^pile_dm-mathematics$",
        r"^20_newsgroups$",
        r"^ag_news$",
        r"^agieval(?:_|$)",
        r"^afriqa(?:_|$)",
        r"^african_flores(?:_|$)",
        r"^afrimgsm",
        r"^afrisenti(?:_|$)",
        r"^afrixnli",
        r"^belebele(?:_|$)",
        r"^blimp$",
        r"^cnn_dailymail$",
        r"^doc_vqa$",
        r"^flores(?:_|$|-)",
        r"^humaneval_infilling$",
        r"^include_base_44_",
        r"^injongointent(?:_|$)",
        r"^japanese_leaderboard$",
        r"^lambada(?:_|$)",
        r"^leaderboard(?:$|_gpqa$|_musr$)",
        r"^longbench(?:_|$|2)",
        r"^mafand(?:_|$|-)",
        r"^masakhaner(?:_|$)",
        r"^masakhanews(?:_|$)",
        r"^masakhapos(?:_|$)",
        r"^med_concepts_qa(?:_|$)",
        r"^multimedqa$",
        r"^nollysenti(?:_|$)",
        r"^openllm$",
        r"^pawsx$",
        r"^pythia$",
        r"^scrolls(?:_|$)",
        r"^sib(?:_|$)",
        r"^stsb$",
        r"^tinyBenchmarks$",
        r"^wmdp$",
        r"^xcopa$",
        r"^xnli(?:_|$)",
        r"^xstorycloze$",
        r"^xwinograd$",
    )
)
UNKNOWN_TASK_NAMES: set[str] = set()
NON_ENGLISH_TASK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:^|[_-])(?:af|am|ar|bn|ca|cs|da|de|el|es|eu|fa|fi|fr|gl|gu|ha|hi|hr|hu|hy|id|it|ja|kn|ko|ml|mr|ne|nl|nno|nob|nso|pt|ro|ru|sk|sr|sv|sw|ta|te|th|tr|uk|ur|vi|wo|yo|zh|zu)(?:[_-]|$)",
        r"^afri",
        r"^afrobench",
        r"^arab",
        r"^aradice",
        r"^basque",
        r"^catalan",
        r"^ceval",
        r"^cmmlu",
        r"^darija",
        r"^egymmlu",
        r"^evalita",
        r"^flores",
        r"^french",
        r"^galician",
        r"^global_mmlu_(?!full_en)",
        r"^haerae",
        r"^hrm8k(?!_en)",
        r"^include_base_44_",
        r"^japanese",
        r"^kbl",
        r"^kmmlu",
        r"^kobest",
        r"^kormedmcqa",
        r"^mafand",
        r"^masakha",
        r"^m_mmlu_(?!en$)",
        r"^mela",
        r"^mgsm",
        r"^mmmlu_(?!en(?:_|$))",
        r"^mmlu_(?:de|es|fr|hi|it|pt|th)_llama",
        r"^mmlu_prox(?:_lite)?_(?!en(?:_|$))[a-z]{2}(?:_|$)",
        r"^naijarc",
        r"^nor",
        r"^ntrex",
        r"^openai_mmlu_(?!eng|en)",
        r"^portuguese",
        r"^spanish",
        r"^tatoeba",
        r"^tmlu",
        r"^tmmlu",
        r"^toksuite_(?:chinese|farsi|italian|turkish)",
        r"^translation",
        r"^trasnlation",
        r"^truthfulqa[-_]multi_(?!(?:gen|mc1|mc2)_en(?:[^a-z0-9]|$))",
        r"^turkish",
        r"^uhura[-_]arc[-_]easy",
        r"^uyghur",
        r"^wmt",
        r"^xlsum",
        r"^xnli",
        r"^xstorycloze",
        r"^xwinograd",
    )
)


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable value without non-standard NaN/Infinity floats."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


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
                "category": task.get("category") or task_category(name),
                "language_scope": task.get("language_scope")
                or task_language_scope(name, task.get("description", "")),
            }
        )
        seen.add(name)
    return merged


def task_category(name: str) -> str:
    normalized = name.lower()
    for category, needles in TASK_CATEGORY_PATTERNS:
        if any(needle in normalized for needle in needles):
            return category
    return "Other"


def task_matches_pattern(name: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def task_language_scope(name: str, *metadata: str) -> str:
    searchable = " ".join([name, *metadata]).lower()
    return (
        "non_english"
        if task_matches_pattern(searchable, NON_ENGLISH_TASK_PATTERNS)
        else "english"
    )


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
    config_text = config_text or ""
    output_type = task_output_type(config_text)
    task_name = task.get("name", "")
    if task_name in INCOMPATIBLE_TASK_NAMES or task_matches_pattern(
        task_name, INCOMPATIBLE_TASK_PATTERNS
    ):
        compatibility = "incompatible"
    elif uses_gated_dataset(config_text):
        compatibility = "gated"
    elif task_name in UNKNOWN_TASK_NAMES:
        compatibility = "unknown"
    elif task_name in COMPATIBLE_TASK_NAMES or task_matches_pattern(
        task_name, COMPATIBLE_TASK_PATTERNS
    ):
        compatibility = "compatible"
    elif (
        has_malformed_group_task_entries(config_text)
        or uses_unsupported_dataset_script(config_text)
        or uses_unavailable_dataset(config_text)
        or uses_unavailable_bleurt_metric(config_text)
        or uses_unavailable_metric(config_text)
        or uses_unsafe_code_eval_metric(config_text)
    ):
        compatibility = "incompatible"
    elif output_type == "generate_until":
        compatibility = "compatible"
    elif output_type:
        compatibility = "incompatible"
    else:
        compatibility = "incompatible"
    return {
        **task,
        "compatibility": compatibility,
        "language_scope": task_language_scope(task_name, config_path, config_text),
    }


def task_output_type(config_text: str) -> str | None:
    match = _OUTPUT_TYPE_RE.search(config_text)
    return match.group(1) if match else None


def dataset_paths(config_text: str) -> set[str]:
    return {match.group(1).strip() for match in _DATASET_PATH_RE.finditer(config_text)}


def uses_gated_dataset(config_text: str) -> bool:
    return bool(dataset_paths(config_text) & GATED_DATASET_PATHS)


def uses_unsupported_dataset_script(config_text: str) -> bool:
    return bool(dataset_paths(config_text) & UNSUPPORTED_DATASET_SCRIPT_PATHS)


def uses_unavailable_dataset(config_text: str) -> bool:
    return bool(dataset_paths(config_text) & UNAVAILABLE_DATASET_PATHS)


def uses_unavailable_bleurt_metric(config_text: str) -> bool:
    return bool(_BLEURT_METRIC_RE.search(config_text))


def uses_unavailable_metric(config_text: str) -> bool:
    return bool(_UNAVAILABLE_METRIC_RE.search(config_text))


def uses_unsafe_code_eval_metric(config_text: str) -> bool:
    return bool(_CODE_EVAL_METRIC_RE.search(config_text))


def has_malformed_group_task_entries(config_text: str) -> bool:
    if not _GROUP_RE.search(config_text):
        return False
    in_task_list = False
    task_indent = 0
    for line in config_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if not in_task_list and indent == 0 and _TOP_LEVEL_TASK_LIST_RE.match(stripped):
            in_task_list = True
            task_indent = indent
            continue
        if not in_task_list:
            continue
        if indent <= task_indent:
            break
        if not stripped.startswith("- "):
            continue
        entry = stripped[2:].strip()
        if ":" not in entry:
            continue
        first_key = entry.split(":", 1)[0].strip()
        if first_key and first_key not in {"task", "group"}:
            return True
    return False


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
    root = (
        Path(package_root)
        if package_root is not None
        else find_lm_eval_package_root("python")
    )
    path = _lm_eval_config_path(config_path, root)
    if path is None:
        return None
    return _read_lm_eval_config(path, root, set())


def _lm_eval_config_path(config_path: str, package_root: Path | None) -> Path | None:
    path = Path(config_path)
    if path.is_absolute():
        return path
    if package_root is None:
        return None
    parts = path.parts
    path = (
        package_root.joinpath(*parts[1:])
        if parts and parts[0] == "lm_eval"
        else package_root / path
    )
    return path


def _read_lm_eval_config(
    path: Path, package_root: Path | None, seen: set[Path]
) -> str | None:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved in seen:
        return ""
    seen.add(resolved)
    try:
        config_text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    included_texts: list[str] = []
    for match in _INCLUDE_RE.finditer(config_text):
        include_path = _resolve_lm_eval_include(match.group(1), path, package_root)
        if include_path is None:
            continue
        include_text = _read_lm_eval_config(include_path, package_root, seen)
        if include_text:
            included_texts.append(include_text)
    return "\n".join([*included_texts, config_text])


def _resolve_lm_eval_include(
    include_path: str, source_path: Path, package_root: Path | None
) -> Path | None:
    raw = include_path.strip().strip("'\"")
    if not raw:
        return None
    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(source_path.parent / path)
        if package_root is not None:
            parts = path.parts
            candidates.append(
                package_root.joinpath(*parts[1:])
                if parts and parts[0] == "lm_eval"
                else package_root / path
            )
    for candidate in _include_path_candidates(candidates):
        if candidate.exists():
            return candidate
    return None


def _include_path_candidates(paths: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for path in paths:
        candidates.append(path)
        if path.suffix != ".yaml":
            candidates.append(path.with_suffix(".yaml"))
            candidates.append(Path(f"{path}.yaml"))
        path_text = str(path)
        if path_text.endswith("_yaml"):
            candidates.append(Path(f"{path_text[:-5]}.yaml"))
    return candidates


def make_handler(
    manager: JobManager,
    static_dir: str | Path,
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL,
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
            base_url = params.get("base_url", [openai_base_url])[0]
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
            body = json.dumps(json_safe(payload), indent=2, allow_nan=False).encode(
                "utf-8"
            )
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
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL,
    lm_eval_python: str | None = None,
    max_concurrent_jobs: int = 1,
) -> None:
    manager = JobManager(
        data_dir=data_dir,
        project_root=Path.cwd(),
        lm_eval_python=lm_eval_python,
        openai_base_url=openai_base_url,
        telemetry_probe=probe_lemonade_chat_telemetry,
        model_metadata_probe=fetch_loaded_model_metadata,
        max_concurrent_jobs=max_concurrent_jobs,
    )
    handler = make_handler(manager, static_dir, openai_base_url)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Serving lm-eval WebUI at http://{host}:{port}")
    httpd.serve_forever()
