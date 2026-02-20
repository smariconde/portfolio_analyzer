from __future__ import annotations

import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


def _candidate_models() -> list[str]:
    models: list[str] = []

    raw_list = os.environ.get("GOOGLE_GENAI_MODELS", "")
    if raw_list.strip():
        models.extend([item.strip() for item in raw_list.split(",") if item.strip()])

    primary = os.environ.get("GOOGLE_GENAI_MODEL") or os.environ.get("GEMINI_MODEL")
    if primary:
        models.append(primary.strip())

    # Prefer newer Flash models first, keep older ones only as fallback.
    models.extend(
        [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
    )

    # Preserve order and remove duplicates.
    deduped: list[str] = []
    seen = set()
    for model in models:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def invoke_gemini(
    prompt: Any,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_retries: int = 4,
    stop: list[str] | None = None,
):
    last_error: Exception | None = None
    tried: list[str] = []

    for model_name in _candidate_models():
        tried.append(model_name)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            stop=stop,
        )
        try:
            return llm.invoke(prompt), model_name
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            model_not_found = (
                "not_found" in message
                or "is not found" in message
                or "supported for generatecontent" in message
            )
            if model_not_found:
                last_error = exc
                continue
            raise

    tried_str = ", ".join(tried)
    raise RuntimeError(
        "No compatible Gemini model found. "
        f"Tried: {tried_str}. "
        "Set GOOGLE_GENAI_MODEL or GOOGLE_GENAI_MODELS in .env. "
        f"Last error: {last_error}"
    )
