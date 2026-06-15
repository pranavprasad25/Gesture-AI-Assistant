"""
ChatGPT service — thin wrapper around the OpenAI Chat Completions API.

Maintains a rolling conversation history so follow-up questions work
naturally, and trims it automatically to stay within token limits.

Usage::

    svc = ChatGPTService()
    reply = svc.ask("What is the capital of France?")
    # reply → "The capital of France is Paris."
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI, APIError, AuthenticationError, RateLimitError

from src.config import (
    OPENAI_API_KEY,
    CHATGPT_MODEL,
    CHATGPT_MAX_TOKENS,
    CHATGPT_TEMPERATURE,
    CHATGPT_MAX_HISTORY,
    CHATGPT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class ChatGPTService:
    """
    Stateful ChatGPT client with rolling conversation memory.

    * **ask()** — send a user message, receive the assistant reply.
    * **reset()** — clear conversation history (keeps system prompt).
    * **is_available** — ``False`` when no API key is configured.

    The history is capped at ``CHATGPT_MAX_HISTORY`` *user+assistant* turn
    pairs so the context window stays manageable.
    """

    def __init__(self) -> None:
        self._available = bool(OPENAI_API_KEY)

        if not self._available:
            logger.warning(
                "OPENAI_API_KEY is not set — ChatGPT features disabled. "
                "Add it to your .env file."
            )
            self._client = None
        else:
            self._client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(
                "ChatGPTService ready (model=%s, max_tokens=%d, temp=%.1f).",
                CHATGPT_MODEL, CHATGPT_MAX_TOKENS, CHATGPT_TEMPERATURE,
            )

        # Conversation history: list of {"role": ..., "content": ...} dicts.
        # The system prompt is prepended at request time, not stored here.
        self._history: list[dict[str, str]] = []

    # ── Public API ───────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """``True`` if an API key is configured and the client is ready."""
        return self._available

    def ask(self, user_message: str) -> Optional[str]:
        """
        Send *user_message* to ChatGPT and return the assistant's reply.

        Conversation history is automatically maintained between calls.
        Returns ``None`` on error or when the service is unavailable.

        Args:
            user_message: The text the user wants to ask.

        Returns:
            The assistant's reply string, or ``None``.
        """
        if not self._available or self._client is None:
            logger.debug("ChatGPT unavailable — skipping ask().")
            return None

        # Trim *before* appending so that the history never grows past the
        # limit between a user message and its paired assistant reply.
        self._trim_history()
        self._history.append({"role": "user", "content": user_message})

        messages = [
            {"role": "system", "content": CHATGPT_SYSTEM_PROMPT},
            *self._history,
        ]

        try:
            response = self._client.chat.completions.create(
                model=CHATGPT_MODEL,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=CHATGPT_MAX_TOKENS,
                temperature=CHATGPT_TEMPERATURE,
            )
            reply = (response.choices[0].message.content or "").strip()

            # Store assistant turn — no need to trim again; we trimmed above.
            self._history.append({"role": "assistant", "content": reply})

            logger.info("ChatGPT reply: '%s'", reply[:120])
            return reply

        except AuthenticationError:
            logger.error("OpenAI authentication failed — check OPENAI_API_KEY.")
            self._available = False
        except RateLimitError:
            logger.warning("OpenAI rate limit reached — try again shortly.")
        except APIError as exc:
            logger.error("OpenAI API error: %s", exc)
        except Exception:
            logger.exception("Unexpected error while calling OpenAI API.")

        # Remove the user message we added since the API call failed.
        # After trim-before-append the last item is always the user message.
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()
        return None

    def reset(self) -> None:
        """Clear conversation history while keeping the service alive."""
        self._history.clear()
        logger.info("ChatGPT conversation history reset.")

    # ── Internals ────────────────────────────────────────────────

    def _trim_history(self) -> None:
        """
        Trim history to at most ``CHATGPT_MAX_HISTORY * 2`` messages.

        Called *before* appending each user message so that the window
        always has room for one new user+assistant pair.  Removes the
        oldest messages first.
        """
        # Each turn = 1 user + 1 assistant message; keep N-1 pairs to leave
        # room for the incoming user message and its forthcoming reply.
        max_messages = max(0, (CHATGPT_MAX_HISTORY - 1) * 2)
        if len(self._history) > max_messages:
            self._history = self._history[len(self._history) - max_messages:]
