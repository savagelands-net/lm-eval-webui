"""Helpers for talking to OpenAI-compatible model hosts."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

DEFAULT_OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    os.environ.get("LEMONADE_BASE_URL", "https://llm.savagelands.net"),
).rstrip("/")
DEFAULT_LEMONADE_BASE_URL = DEFAULT_OPENAI_BASE_URL


def normalize_openai_base_url(base_url: str) -> str:
    normalized = str(base_url or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized[: -len("/chat/completions")]
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def openai_api_url(base_url: str, path: str) -> str:
    suffix = path if path.startswith("/") else f"/{path}"
    if suffix.startswith("/v1/"):
        suffix = suffix[len("/v1") :]
    return f"{normalize_openai_base_url(base_url)}{suffix}"


def _runtime_backend(recipe: str, recipe_options: dict[str, Any]) -> str:
    explicit = recipe_options.get("llamacpp_backend")
    if explicit:
        return str(explicit)
    return "" if recipe == "llamacpp" else recipe


def loaded_model_metadata_from_health(
    health_payload: dict[str, Any], model_id: str
) -> dict[str, Any]:
    requested = str(model_id)
    for loaded in health_payload.get("all_models_loaded") or []:
        if not isinstance(loaded, dict):
            continue
        model_name = str(loaded.get("model_name") or "")
        if model_name != requested:
            continue
        recipe = str(loaded.get("recipe") or "")
        recipe_options = loaded.get("recipe_options") or {}
        llamacpp_backend = recipe_options.get("llamacpp_backend")
        runtime_backend = _runtime_backend(recipe, recipe_options)
        return {
            "model_name": model_name,
            "recipe": recipe,
            "llamacpp_backend": str(llamacpp_backend)
            if llamacpp_backend
            else None,
            "runtime_backend": runtime_backend,
            "device": loaded.get("device"),
            "checkpoint": loaded.get("checkpoint", ""),
            "backend_url": loaded.get("backend_url", ""),
        }
    return {}


def fetch_loaded_model_metadata(
    base_url: str = DEFAULT_LEMONADE_BASE_URL,
    model_id: str = "",
    timeout: int = 15,
) -> dict[str, Any]:
    health_request = urllib.request.Request(
        openai_api_url(base_url, "/health"), headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(health_request, timeout=timeout) as response:  # noqa: S310
        health = json.loads(response.read().decode("utf-8"))
    return loaded_model_metadata_from_health(health, model_id)


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
        runtime_backend = _runtime_backend(recipe, recipe_options)
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
    loaded_backends: dict[str, dict[str, Any]] = {}
    for loaded in health_payload.get("all_models_loaded") or []:
        if not isinstance(loaded, dict):
            continue
        model_name = str(loaded.get("model_name") or "")
        if not model_name:
            continue
        metadata = loaded_model_metadata_from_health(health_payload, model_name)
        if metadata.get("runtime_backend"):
            loaded_backends[model_name] = metadata
    enriched: list[dict[str, Any]] = []
    for model in models:
        backend = loaded_backends.get(str(model.get("id")))
        if backend:
            model = {
                **model,
                "llamacpp_backend": backend.get("llamacpp_backend"),
                "runtime_backend": backend.get("runtime_backend"),
            }
        enriched.append(model)
    return enriched


def fetch_models(
    base_url: str = DEFAULT_LEMONADE_BASE_URL, timeout: int = 15
) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        openai_api_url(base_url, "/models"), headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    models = normalize_models(payload)
    try:
        health_request = urllib.request.Request(
            openai_api_url(base_url, "/health"), headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(health_request, timeout=timeout) as response:  # noqa: S310
            health = json.loads(response.read().decode("utf-8"))
    except Exception:
        return models
    return enrich_models_from_health(models, health)
