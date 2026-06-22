"""Entrypoint that registers local lm-eval plugins before delegating to lm-eval."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any, cast


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
                if name == "ACP_grammar_filter" and "already registered" in str(exc):
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
    return int(cli_evaluate() or 0)


if __name__ == "__main__":
    sys.exit(main())
