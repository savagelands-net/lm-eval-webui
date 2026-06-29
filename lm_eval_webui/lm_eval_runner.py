"""Entrypoint that registers local lm-eval plugins before delegating to lm-eval."""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Callable
from typing import Any, cast

TRANSIENT_HF_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_HF_RETRIES = 5
DEFAULT_HF_RETRY_DELAY = 10.0
DEFAULT_HF_RETRY_MAX_DELAY = 120.0


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


def _should_retry_hf_error(
    exc: OSError, attempt: int, retry_count: int
) -> bool:
    if attempt >= retry_count:
        return False
    return is_transient_huggingface_error(exc)


def run_cli_with_hf_retries(
    cli_evaluate: Callable[[], Any],
    retries: int | None = None,
    initial_delay: float | None = None,
    max_delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    stderr: Any | None = None,
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
        except OSError as exc:
            if _should_retry_hf_error(exc, attempt, retry_count):
                attempt += 1
                delay = min(delay_cap, base_delay * (2 ** (attempt - 1)))
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
