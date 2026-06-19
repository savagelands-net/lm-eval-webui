"""Helpers for talking to a Lemonade/OpenAI-compatible model host."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

DEFAULT_LEMONADE_BASE_URL = os.environ.get(
    "LEMONADE_BASE_URL", "https://llm.savagelands.net"
).rstrip("/")


def normalize_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("data", [])
    normalized: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("downloaded") is False:
            continue
        model_id = str(item.get("id") or item.get("name") or "").strip()
        if not model_id:
            continue
        labels = item.get("labels") or []
        normalized.append(
            {
                "id": model_id,
                "name": model_id,
                "labels": [str(label) for label in labels if label is not None],
                "size_gb": item.get("size"),
                "context_window": item.get("max_context_window"),
                "recipe": item.get("recipe", ""),
                "checkpoint": item.get("checkpoint", ""),
                "suggested": bool(item.get("suggested", False)),
            }
        )
    return sorted(normalized, key=lambda model: model["id"].lower())


def fetch_models(
    base_url: str = DEFAULT_LEMONADE_BASE_URL, timeout: int = 15
) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/models", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return normalize_models(payload)
