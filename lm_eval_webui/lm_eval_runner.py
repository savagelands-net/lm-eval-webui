"""Entrypoint that registers local lm-eval plugins before delegating to lm-eval."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

TRANSIENT_HF_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_HF_RETRIES = 5
DEFAULT_HF_RETRY_DELAY = 10.0
DEFAULT_HF_RETRY_MAX_DELAY = 120.0
DATASET_INFO_FILENAME = "dataset_info.json"


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, value)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _status_code(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _mentions_huggingface(value: Any) -> bool:
    return "huggingface.co" in str(value).lower()


def is_transient_huggingface_error(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        response = getattr(item, "response", None)
        status_code = _status_code(getattr(response, "status_code", None))
        if status_code not in TRANSIENT_HF_STATUS_CODES:
            continue
        module_name = type(item).__module__
        url = getattr(response, "url", "")
        if (
            _mentions_huggingface(url)
            or _mentions_huggingface(item)
            or "huggingface_hub" in module_name
        ):
            return True
    return False


def is_huggingface_dataset_cache_miss(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        message = str(item).lower()
        if (
            "couldn't find cache for" in message
            and "available configs in the cache" in message
        ):
            return True
    return False


def _should_retry_hf_error(exc: BaseException, attempt: int, retry_count: int) -> bool:
    if attempt >= retry_count:
        return False
    return is_transient_huggingface_error(exc) or is_huggingface_dataset_cache_miss(exc)


def _retry_delay(attempt: int, base_delay: float, delay_cap: float) -> float:
    return min(delay_cap, base_delay * (2 ** (attempt - 1)))


def _default_hf_dataset_cache_roots() -> list[Path]:
    roots: list[Path] = []
    if hf_datasets_cache := os.environ.get("HF_DATASETS_CACHE"):
        roots.append(Path(hf_datasets_cache))
    if hf_home := os.environ.get("HF_HOME"):
        roots.append(Path(hf_home) / "datasets")
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
    return _deduplicate_paths(roots)


def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        deduplicated.append(path.expanduser())
        seen.add(key)
    return deduplicated


def _hf_dataset_cache_roots(
    cache_roots: Iterable[str | Path] | None = None,
) -> list[Path]:
    if cache_roots is None:
        return _default_hf_dataset_cache_roots()
    return _deduplicate_paths(Path(root) for root in cache_roots)


def find_corrupt_hf_dataset_cache_dirs(
    cache_roots: Iterable[str | Path] | None = None,
) -> list[Path]:
    corrupt_dirs: list[Path] = []
    seen: set[str] = set()
    for root in _hf_dataset_cache_roots(cache_roots):
        if not root.exists():
            continue
        try:
            info_paths = root.rglob(DATASET_INFO_FILENAME)
            for info_path in info_paths:
                try:
                    json.loads(info_path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    cache_dir = info_path.parent
                    key = str(cache_dir)
                    if key not in seen:
                        corrupt_dirs.append(cache_dir)
                        seen.add(key)
                except OSError:
                    continue
        except OSError:
            continue
    return corrupt_dirs


def repair_corrupt_hf_dataset_cache(
    cache_roots: Iterable[str | Path] | None = None,
    stderr: Any | None = None,
) -> int:
    output = stderr or sys.stderr
    repaired = 0
    for cache_dir in find_corrupt_hf_dataset_cache_dirs(cache_roots):
        print(
            f"Removing corrupt Hugging Face dataset cache directory: {cache_dir}",
            file=output,
            flush=True,
        )
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except OSError as exc:
            print(
                f"Could not remove corrupt Hugging Face cache {cache_dir}: {exc}",
                file=output,
                flush=True,
            )
            continue
        repaired += 1
    return repaired


def run_cli_with_hf_retries(
    cli_evaluate: Callable[[], Any],
    retries: int | None = None,
    initial_delay: float | None = None,
    max_delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    stderr: Any | None = None,
    cache_roots: Iterable[str | Path] | None = None,
) -> int:
    retry_count = (
        _env_int("LMEVAL_WEBUI_HF_RETRIES", DEFAULT_HF_RETRIES)
        if retries is None
        else max(0, retries)
    )
    base_delay = (
        _env_float("LMEVAL_WEBUI_HF_RETRY_DELAY", DEFAULT_HF_RETRY_DELAY)
        if initial_delay is None
        else max(0.0, initial_delay)
    )
    delay_cap = (
        _env_float("LMEVAL_WEBUI_HF_RETRY_MAX_DELAY", DEFAULT_HF_RETRY_MAX_DELAY)
        if max_delay is None
        else max(0.0, max_delay)
    )
    output = stderr or sys.stderr
    attempt = 0
    while True:
        try:
            return int(cli_evaluate() or 0)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            repaired = (
                repair_corrupt_hf_dataset_cache(cache_roots, output)
                if attempt < retry_count
                else 0
            )
            if repaired <= 0:
                raise
            attempt += 1
            delay = _retry_delay(attempt, base_delay, delay_cap)
            print(
                "Repaired corrupt Hugging Face dataset cache; "
                f"retrying lm-eval in {delay:g}s "
                f"({attempt}/{retry_count}): {exc}",
                file=output,
                flush=True,
            )
            sleep(delay)
            continue
        except (OSError, ValueError) as exc:
            if _should_retry_hf_error(exc, attempt, retry_count):
                attempt += 1
                delay = _retry_delay(attempt, base_delay, delay_cap)
                print(
                    "Transient Hugging Face dataset error; "
                    f"retrying lm-eval in {delay:g}s "
                    f"({attempt}/{retry_count}): {exc}",
                    file=output,
                    flush=True,
                )
                sleep(delay)
                continue
            raise


def _is_duplicate_acp_filter_error(name: str, exc: ValueError) -> bool:
    if name != "ACP_grammar_filter":
        return False
    return "already registered" in str(exc)


def allow_duplicate_acp_filter_registration(registry_module: Any | None = None) -> None:
    """Allow lm-eval's duplicate ACP grammar filter registration bug.

    lm-eval 0.4.12's ACPBench gen and gen-with-PDDL task families both register
    ``ACP_grammar_filter``. Loading both families in one run raises before any
    benchmark request is made. The filter implementations are equivalent for the
    affected task configs, so keep the first registration and ignore only that
    known duplicate alias.
    """

    registry = cast(
        Any,
        registry_module
        if registry_module is not None
        else importlib.import_module("lm_eval.api.registry"),
    )
    original_register_filter = registry.register_filter
    if getattr(original_register_filter, "_webui_acp_duplicate_guard", False):
        return

    def register_filter(name: str):
        decorate = original_register_filter(name)

        def guarded_decorate(cls: type[Any]) -> type[Any]:
            try:
                return decorate(cls)
            except ValueError as exc:
                if _is_duplicate_acp_filter_error(name, exc):
                    return cls
                raise

        return guarded_decorate

    register_filter._webui_acp_duplicate_guard = True  # type: ignore[attr-defined]
    registry.register_filter = register_filter


def main() -> int:
    importlib.import_module("lm_eval_webui.lemonade_model")
    allow_duplicate_acp_filter_registration()
    lm_eval_main = importlib.import_module("lm_eval.__main__")
    cli_evaluate = cast(Callable[[], Any], lm_eval_main.__dict__["cli_evaluate"])
    return run_cli_with_hf_retries(cli_evaluate)


if __name__ == "__main__":
    sys.exit(main())
