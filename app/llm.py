"""One function between the app and the model, so the provider is easy to swap.

Every call is logged to SQLite (purpose, model, tokens, duration), which is
where the speed and cost figures on the Evaluation page come from.
"""
import contextvars
import functools
import time
from contextlib import contextmanager

import config
import persistence as db

_tags = contextvars.ContextVar("llm_tags", default={})


@contextmanager
def tagged(**tags):
    """Attach labels (source, doc_id, ref) to every call made inside the block."""
    token = _tags.set({**_tags.get(), **tags})
    try:
        yield
    finally:
        _tags.reset(token)


@functools.lru_cache(maxsize=1)
def _client():
    if config.LLM_PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI(max_retries=5)  # reads OPENAI_API_KEY and OPENAI_BASE_URL
    from anthropic import Anthropic
    return Anthropic(max_retries=5)  # reads ANTHROPIC_API_KEY


def cost(model, input_tokens, output_tokens):
    """USD for one call, from the prices in .env; 0 if none are set for the model."""
    price_in, price_out = config.PRICES.get(model, (0, 0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def complete(system, messages, fast=False, max_tokens=2000, purpose="", model=None):
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...] -> reply text."""
    model = model or (config.LLM_FAST_MODEL if fast else config.LLM_MODEL)
    started = time.perf_counter()
    if config.LLM_PROVIDER == "openai":
        resp = _client().chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}, *messages]
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tokens = (getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)
    else:
        resp = _client().messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        tokens = (resp.usage.input_tokens, resp.usage.output_tokens)
    record(purpose, model, *tokens, int((time.perf_counter() - started) * 1000))
    return text


def record(purpose, model, input_tokens, output_tokens, duration_ms):
    tags = _tags.get()
    try:
        db.log_llm_call(tags.get("source", "live"), tags.get("doc_id"), tags.get("ref"),
                        purpose, model, input_tokens, output_tokens, duration_ms)
    except Exception:
        pass  # usage logging must never break an answer
