"""Thin wrapper around the Anthropic SDK: one place for model choice, timeouts, and errors.

Callers get plain text back or an LLMError with a message that is safe to show
in the UI. The API key is read by the SDK from ANTHROPIC_API_KEY and is never
logged or echoed.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import anthropic

from scout.config import LLM_MAX_RETRIES, LLM_TIMEOUT_S, SCOUT_MODEL, has_api_key

log = logging.getLogger(__name__)

# Models that accept server-side refusal fallbacks in the "default" form.
# On a safety decline the API re-runs the request on a suitable model inside
# the same call, so a rare false-positive refusal doesn't break the demo.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Raised for any LLM failure. The message is user-safe."""


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Shared client with conservative timeout and automatic retries."""
    if not has_api_key():
        raise LLMError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(timeout=LLM_TIMEOUT_S, max_retries=LLM_MAX_RETRIES)


def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = 4000,
    json_schema: dict[str, Any] | None = None,
    effort: str = "low",
    model: str = SCOUT_MODEL,
) -> str:
    """Send one system+user turn and return the response text.

    json_schema: if given, the response is constrained to JSON matching it
    (structured outputs). Callers should still validate, e.g. with Pydantic.
    effort: thinking depth; "low" is plenty for parsing, use "medium" for prose.
    """
    output_config: dict[str, Any] = {"effort": effort}
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    kwargs: dict[str, Any] = dict(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}], output_config=output_config,
    )
    try:
        client = get_client()
        if model in _FALLBACK_MODELS:
            response = client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = client.messages.create(**kwargs)
    except LLMError:
        raise
    except anthropic.AuthenticationError as e:
        raise LLMError("Anthropic rejected the API key (check ANTHROPIC_API_KEY)") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Rate limited by the Anthropic API; try again in a minute") from e
    except anthropic.APITimeoutError as e:
        raise LLMError(f"Anthropic API timed out after {LLM_TIMEOUT_S:.0f}s") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Anthropic API (network issue)") from e
    except anthropic.BadRequestError as e:
        raise LLMError(f"Anthropic API rejected the request: {e.message}") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}") from e

    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request")
    if response.stop_reason == "max_tokens":
        raise LLMError("The model ran out of output tokens")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise LLMError("The model returned an empty response")
    log.info("LLM call ok: model=%s request_id=%s", response.model, response._request_id)
    return text
