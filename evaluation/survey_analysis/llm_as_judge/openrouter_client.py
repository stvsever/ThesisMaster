"""
Thin wrapper around OpenRouter for the LLM-as-judge.

Uses the OpenAI Python SDK if available (preferred) and falls back to direct
HTTP via :mod:`urllib` otherwise. Either way the call is wrapped in an
exponential-backoff retry loop with a JSON-validation re-prompt step.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "google/gemini-2.5-flash"
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS: int = 2048
DEFAULT_TIMEOUT_S: float = 90.0

# Optional headers OpenRouter accepts for attribution / project leaderboards.
DEFAULT_REFERER: str = "https://github.com/PHOENIX-eval/masterproef"
DEFAULT_TITLE: str = "PHOENIX Evaluation"

JSON_RETRY_INSTRUCTION = (
    "Your previous response failed JSON parsing. "
    "Return strict JSON only — no prose, no markdown fences. "
    "Match the schema described in the previous user turn exactly."
)


@dataclass
class ChatCompletion:
    """Outcome of one chat completion call."""

    text: str
    raw: Dict[str, Any]
    model: str
    elapsed_s: float
    attempt: int


class OpenRouterError(RuntimeError):
    """Raised when all retries fail."""


class OpenRouterClient:
    """
    Minimal client that targets OpenRouter's OpenAI-compatible endpoint.

    Parameters
    ----------
    api_key : str | None
        Resolved from ``OPENROUTER_API_KEY`` if not supplied.
    model : str
        OpenRouter model identifier.
    temperature : float
        Sampling temperature.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: str = DEFAULT_BASE_URL,
        referer: str = DEFAULT_REFERER,
        title: str = DEFAULT_TITLE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is missing. Either set the environment "
                "variable or use --mode pseudo to run without API calls."
            )
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.title = title
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

        # Lazy import: try OpenAI SDK first, fall back to urllib.
        try:
            from openai import OpenAI  # type: ignore
            self._sdk = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": self.referer,
                    "X-Title": self.title,
                },
                timeout=self.timeout_s,
            )
            self._mode = "openai"
        except Exception as exc:  # pragma: no cover - dep not installed
            logger.info("openai SDK not available, falling back to urllib (%s)", exc)
            self._sdk = None
            self._mode = "urllib"

    # ── Public API ──────────────────────────────────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        seed: Optional[int] = None,
    ) -> ChatCompletion:
        """One non-retried chat completion call (used internally)."""
        t0 = time.monotonic()
        if self._mode == "openai":
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if seed is not None:
                # Some OpenRouter routes propagate `seed`; harmless if ignored.
                kwargs["seed"] = int(seed)
            resp = self._sdk.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            raw = resp.model_dump() if hasattr(resp, "model_dump") else {"raw": str(resp)}
        else:
            text, raw = self._urllib_call(messages, seed=seed)
        return ChatCompletion(
            text=text,
            raw=raw,
            model=self.model,
            elapsed_s=time.monotonic() - t0,
            attempt=1,
        )

    def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        n_retries: int = 3,
        seed: Optional[int] = None,
    ) -> ChatCompletion:
        """
        Chat completion with exponential backoff on transient errors.
        Does NOT do JSON validation; that's the caller's responsibility.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, n_retries + 1):
            try:
                completion = self.chat_completion(messages, seed=seed)
                completion.attempt = attempt
                return completion
            except Exception as exc:
                last_exc = exc
                wait = min(30.0, 1.5 ** attempt + random.random() * 0.5)
                logger.warning(
                    "OpenRouter call failed on attempt %d/%d: %s; sleeping %.1fs",
                    attempt, n_retries, exc, wait,
                )
                time.sleep(wait)
        raise OpenRouterError(f"All {n_retries} retries failed: {last_exc!r}")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _urllib_call(
        self,
        messages: List[Dict[str, str]],
        seed: Optional[int],
    ) -> tuple[str, Dict[str, Any]]:
        import urllib.request
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            body["seed"] = int(seed)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.referer,
                "X-Title": self.title,
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = (payload["choices"][0]["message"].get("content") or "").strip()
        return text, payload


__all__ = [
    "OpenRouterClient",
    "OpenRouterError",
    "ChatCompletion",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "JSON_RETRY_INSTRUCTION",
]
