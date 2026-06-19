"""Entrypoint that registers local lm-eval plugins before delegating to lm-eval."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any, cast


def main() -> int:
    importlib.import_module("lm_eval_webui.lemonade_model")
    lm_eval_main = importlib.import_module("lm_eval.__main__")
    cli_evaluate = cast(Callable[[], Any], lm_eval_main.__dict__["cli_evaluate"])
    return int(cli_evaluate() or 0)


if __name__ == "__main__":
    sys.exit(main())
