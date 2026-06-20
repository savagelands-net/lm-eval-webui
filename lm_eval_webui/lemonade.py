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
        recipe = str(item.get("recipe", ""))
        recipe_options = item.get("recipe_options") or {}
        llamacpp_backend = recipe_options.get("llamacpp_backend")
        runtime_backend = str(llamacpp_backend or recipe or "")
        normalized.append(
            {
                "id": model_id,
                "name": model_id,
                "labels": [str(label) for label in labels if label is not None],
                "size_gb": item.get("size"),
                "context_window": item.get("max_context_window"),
                "recipe": recipe,
                "llamacpp_backend": llamacpp_backend,
                "runtime_backend": runtime_backend,
                "checkpoint": item.get("checkpoint", ""),
                "suggested": bool(item.get("suggested", False)),
            }
        )
    return sorted(normalized, key=lambda model: model["id"].lower())


def enrich_models_from_health(
    models: list[dict[str, Any]], health_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    loaded_backends: dict[str, str] = {}
    for loaded in health_payload.get("all_models_loaded") or []:
        if not isinstance(loaded, dict):
            continue
        model_name = str(loaded.get("model_name") or "")
        recipe_options = loaded.get("recipe_options") or {}
        backend = recipe_options.get("llamacpp_backend")
        if model_name and backend:
            loaded_backends[model_name] = str(backend)
    enriched: list[dict[str, Any]] = []
    for model in models:
        backend = loaded_backends.get(str(model.get("id")))
        if backend:
            model = {**model, "llamacpp_backend": backend, "runtime_backend": backend}
        enriched.append(model)
    return enriched


def fetch_models(
    base_url: str = DEFAULT_LEMONADE_BASE_URL, timeout: int = 15
) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/models", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    models = normalize_models(payload)
    try:
        health_request = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/health", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(health_request, timeout=timeout) as response:  # noqa: S310
            health = json.loads(response.read().decode("utf-8"))
    except Exception:
        return models
    return enrich_models_from_health(models, health)
