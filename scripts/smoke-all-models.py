#!/usr/bin/env python3
"""Queue a small GSM8K, MMLU, and IFEval preflight for every chat model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SMOKE_TASKS = ("gsm8k", "mmlu_abstract_algebra_generative", "ifeval")
DEFAULT_WEBUI_URL = "http://127.0.0.1:8080"
DEFAULT_MAX_GEN_TOKS = 32_768
DEFAULT_TIMEOUT = 7_200
TERMINAL_STATUSES = {"cancelled", "failed", "succeeded"}
NON_CHAT_LABELS = {
    "3d",
    "audio-generation",
    "embeddings",
    "image-generation",
    "reranking",
    "transcription",
    "tts",
}


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path} returned HTTP {exc.code}: {body}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{method} {path} failed: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{method} {path} returned a non-object response")
    return result


def is_chat_model(model: dict[str, Any]) -> bool:
    labels = {str(label).lower() for label in model.get("labels") or []}
    return not labels.intersection(NON_CHAT_LABELS)


def discover_models(
    webui_url: str, openai_base_url: str, selected: list[str]
) -> list[str]:
    query = urllib.parse.urlencode({"base_url": openai_base_url})
    response = request_json(webui_url, f"/api/models?{query}")
    available = [
        str(model.get("id"))
        for model in response.get("models") or []
        if isinstance(model, dict) and model.get("id") and is_chat_model(model)
    ]
    if not selected:
        return available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise RuntimeError(f"Unknown or unavailable model(s): {', '.join(missing)}")
    return selected


def smoke_payload(
    model_ids: list[str], openai_base_url: str, samples: int
) -> dict[str, Any]:
    return {
        "suite": "lm_eval",
        "model_ids": model_ids,
        "tasks": list(SMOKE_TASKS),
        "openai_base_url": openai_base_url,
        "limit": str(samples),
        "num_fewshot": None,
        "max_gen_toks": DEFAULT_MAX_GEN_TOKS,
        "timeout": DEFAULT_TIMEOUT,
        "num_concurrent": 1,
        "max_concurrent_jobs": 1,
        "batch_size": "1",
        "task_batch_size": 1,
        "apply_chat_template": True,
        "fewshot_as_multiturn": False,
        "log_samples": True,
    }


def wait_for_jobs(webui_url: str, job_ids: set[str], poll_seconds: float) -> bool:
    while True:
        response = request_json(webui_url, "/api/jobs")
        jobs = {
            str(job.get("id")): job
            for job in response.get("jobs") or []
            if isinstance(job, dict) and job.get("id") in job_ids
        }
        status_line = " | ".join(
            f"{job.get('model_id')}: {job.get('status')}" for job in jobs.values()
        )
        print(status_line or "Waiting for queued jobs to appear...", flush=True)
        if len(jobs) == len(job_ids) and all(
            job.get("status") in TERMINAL_STATUSES for job in jobs.values()
        ):
            break
        time.sleep(poll_seconds)

    passed = True
    print("\nModel smoke-test summary")
    print("model\tstatus\tresponses\tfinal-content\tempty\tlimited\tresult")
    for job_id in sorted(job_ids):
        job = request_json(webui_url, f"/api/jobs/{job_id}").get("job") or {}
        telemetry = job.get("telemetry") or {}
        response_count = telemetry.get("response_metadata_count", 0)
        final_count = telemetry.get("final_content_response_count", 0)
        empty_count = telemetry.get("empty_response_count", 0)
        limited_count = telemetry.get("generation_limited_response_count", 0)
        ok = (
            job.get("status") == "succeeded"
            and response_count > 0
            and final_count == response_count
            and empty_count == 0
            and limited_count == 0
        )
        passed = passed and ok
        print(
            f"{job.get('model_id')}\t{job.get('status')}\t{response_count}"
            f"\t{final_count}\t{empty_count}\t{limited_count}"
            f"\t{'PASS' if ok else 'FAIL'}"
        )
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--webui-url",
        default=os.environ.get("LMEVAL_WEBUI_URL", DEFAULT_WEBUI_URL),
    )
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--model", action="append", default=[], help="Model ID; repeatable"
    )
    parser.add_argument("--samples", type=int, default=3, help="Samples per smoke task")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually enqueue jobs; without this flag the script is read-only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise RuntimeError("--samples must be at least 1")
    openai_base_url = args.openai_base_url
    if not openai_base_url:
        openai_base_url = str(
            request_json(args.webui_url, "/api/config").get("openai_base_url") or ""
        )
    if not openai_base_url:
        raise RuntimeError("No OpenAI-compatible base URL is configured")

    model_ids = discover_models(args.webui_url, openai_base_url, args.model)
    if not model_ids:
        raise RuntimeError("No downloaded models were returned by the endpoint")
    payload = smoke_payload(model_ids, openai_base_url, args.samples)
    print(json.dumps(payload, indent=2))
    if not args.run:
        print("\nDry run only. Add --run to enqueue these serial smoke jobs.")
        return 0

    response = request_json(
        args.webui_url,
        "/api/jobs",
        method="POST",
        payload=payload,
    )
    job_ids = {
        str(job.get("id"))
        for job in response.get("jobs") or []
        if isinstance(job, dict) and job.get("id")
    }
    if len(job_ids) != len(model_ids):
        raise RuntimeError(
            f"Expected {len(model_ids)} jobs but the WebUI created {len(job_ids)}"
        )
    return 0 if wait_for_jobs(args.webui_url, job_ids, args.poll_seconds) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
