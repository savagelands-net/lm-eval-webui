"""Helpers for talking to OpenAI-compatible model hosts."""

from __future__ import annotations

import importlib
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

DEFAULT_LOCAL_OPENAI_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    os.environ.get("LEMONADE_BASE_URL", DEFAULT_LOCAL_OPENAI_BASE_URL),
).rstrip("/")
DEFAULT_LEMONADE_BASE_URL = DEFAULT_OPENAI_BASE_URL
MODEL_LOAD_TIMEOUT = 7_200
MODEL_PIN_TIMEOUT = 120


class ModelLifecycleUnsupported(RuntimeError):
    """Raised when an OpenAI-compatible host has no Lemonade lifecycle API."""


def normalize_openai_base_url(base_url: str) -> str:
    normalized = str(base_url or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "OpenAI-compatible base URL must start with http:// or https://"
        )
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


def lemonade_management_base_url(base_url: str) -> str:
    """Return the server root corresponding to an OpenAI-compatible /v1 URL."""

    parsed = urlsplit(normalize_openai_base_url(base_url))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def lemonade_management_url(base_url: str, path: str) -> str:
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{lemonade_management_base_url(base_url)}{suffix}"


def _runtime_backend(recipe: str, recipe_options: dict[str, Any]) -> str:
    explicit = recipe_options.get("llamacpp_backend")
    if explicit:
        return str(explicit)
    return "system" if recipe == "llamacpp" else recipe


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _context_window(
    metadata: dict[str, Any], recipe_options: dict[str, Any]
) -> int | None:
    for value in (
        metadata.get("max_context_window"),
        metadata.get("context_window"),
        metadata.get("max_model_len"),
        recipe_options.get("ctx_size"),
        recipe_options.get("max_model_len"),
    ):
        context_window = _positive_int(value)
        if context_window is not None:
            return context_window
    return None


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
        raw_recipe_options = loaded.get("recipe_options") or {}
        recipe_options = (
            raw_recipe_options if isinstance(raw_recipe_options, dict) else {}
        )
        runtime_backend = _runtime_backend(recipe, recipe_options)
        return {
            "model_name": model_name,
            "recipe": recipe,
            "llamacpp_backend": runtime_backend if recipe == "llamacpp" else None,
            "runtime_backend": runtime_backend,
            "context_window": _context_window(loaded, recipe_options),
            "device": loaded.get("device"),
            "checkpoint": loaded.get("checkpoint", ""),
            "backend_url": loaded.get("backend_url", ""),
        }
    return {}


def _read_json_response(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("Invalid JSON response from model endpoint") from exc
    return payload if isinstance(payload, dict) else {}


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    requests_module = importlib.import_module("requests")
    response = requests_module.get(
        url,
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return _read_json_response(response)


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    requests_module = importlib.import_module("requests")
    response = requests_module.post(
        url,
        json=payload,
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    if response.status_code in {404, 405}:
        raise ModelLifecycleUnsupported(
            f"Model host does not expose the Lemonade lifecycle API ({response.status_code})"
        )
    response.raise_for_status()
    if not response.content:
        return {}
    return _read_json_response(response)


def load_and_pin_model(
    base_url: str,
    model_id: str,
    timeout: int = MODEL_LOAD_TIMEOUT,
) -> dict[str, Any]:
    """Load a Lemonade model and protect it from automatic eviction."""

    return _post_json(
        lemonade_management_url(base_url, "/api/v1/load"),
        {"model_name": str(model_id), "pinned": True},
        timeout,
    )


def unpin_model(
    base_url: str,
    model_id: str,
    timeout: int = MODEL_PIN_TIMEOUT,
) -> dict[str, Any]:
    """Release benchmark-owned eviction protection for a Lemonade model."""

    return _post_json(
        lemonade_management_url(base_url, "/internal/pin"),
        {"model": str(model_id), "pinned": False},
        timeout,
    )


def fetch_loaded_model_metadata(
    base_url: str = DEFAULT_LEMONADE_BASE_URL,
    model_id: str = "",
    timeout: int = 15,
) -> dict[str, Any]:
    health = _get_json(openai_api_url(base_url, "/health"), timeout)
    return loaded_model_metadata_from_health(health, model_id)


def normalize_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("data", [])
    normalized: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        downloaded = item.get("downloaded")
        if isinstance(downloaded, bool) and not downloaded:
            continue
        model_id = str(item.get("id") or item.get("name") or "").strip()
        if not model_id:
            continue
        labels = item.get("labels") or []
        recipe = str(item.get("recipe", ""))
        raw_recipe_options = item.get("recipe_options") or {}
        recipe_options = (
            raw_recipe_options if isinstance(raw_recipe_options, dict) else {}
        )
        runtime_backend = _runtime_backend(recipe, recipe_options)
        normalized.append(
            {
                "id": model_id,
                "name": model_id,
                "labels": [str(label) for label in labels if label is not None],
                "size_gb": item.get("size"),
                "context_window": _context_window(item, recipe_options),
                "recipe": recipe,
                "llamacpp_backend": runtime_backend if recipe == "llamacpp" else None,
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
                "context_window": backend.get("context_window")
                or model.get("context_window"),
            }
        enriched.append(model)
    return enriched


def fetch_models(
    base_url: str = DEFAULT_LEMONADE_BASE_URL, timeout: int = 15
) -> list[dict[str, Any]]:
    payload = _get_json(openai_api_url(base_url, "/models"), timeout)
    models = normalize_models(payload)
    try:
        health = _get_json(openai_api_url(base_url, "/health"), timeout)
    except (OSError, ValueError):
        return models
    return enrich_models_from_health(models, health)
