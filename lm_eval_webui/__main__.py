"""Command-line entrypoint for the Lemonade lm-eval Benchmark WebUI."""

from __future__ import annotations

import argparse
from typing import Any

from lm_eval_webui.lemonade import DEFAULT_OPENAI_BASE_URL
from lm_eval_webui.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Lemonade lm-eval Benchmark WebUI"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--static-dir", default="static")
    parser.add_argument("--openai-base-url", default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--lemonade-base-url", default=None)
    parser.add_argument("--lm-eval-python", default=None)
    parser.add_argument("--max-concurrent-jobs", type=int, default=1)
    parser.add_argument("--max-request-workers", type=int, default=16)
    parser.add_argument("--pi-bench-dir", default=None)
    args = parser.parse_args()
    serve_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": args.port,
        "data_dir": args.data_dir,
        "static_dir": args.static_dir,
        "openai_base_url": args.lemonade_base_url or args.openai_base_url,
        "lm_eval_python": args.lm_eval_python,
        "max_concurrent_jobs": args.max_concurrent_jobs,
        "max_request_workers": args.max_request_workers,
        "pi_bench_dir": args.pi_bench_dir,
    }
    serve(**serve_kwargs)


if __name__ == "__main__":
    main()
