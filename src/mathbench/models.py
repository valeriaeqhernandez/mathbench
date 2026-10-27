"""Model clients.

`OpenAICompatibleModel` talks to any server that speaks the OpenAI chat API. That one
format covers OpenAI, OpenRouter, Together, vLLM, and Ollama running on your laptop, so
one class lets you compare many models.

The fake models need no network. They exist for tests and for sanity checks:
`memorizer` only knows canonical answers, so a working benchmark MUST show a large gap
for it. If it doesn't, the pipeline is broken.
"""

from __future__ import annotations

import asyncio
import os
from typing import Protocol

import httpx

from mathbench.cache import Cache
from mathbench.problems import Problem

SYSTEM_PROMPT = (
    "Solve the math problem. Reason step by step, then give the final answer "
    "as a single integer inside \\boxed{}."
)


class Model(Protocol):
    name: str

    async def solve(self, problem: Problem, sample: int) -> str: ...


class OracleModel:
    """Always right. Upper bound; checks extraction and grading."""

    def __init__(self, name: str = "oracle"):
        self.name = name

    async def solve(self, problem: Problem, sample: int) -> str:
        return f"Working... so the answer is \\boxed{{{problem.answer}}}."


class MemorizerModel:
    """Right only on canonical problems; guesses wrong otherwise. Simulates contamination."""

    def __init__(self, name: str = "memorizer"):
        self.name = name

    async def solve(self, problem: Problem, sample: int) -> str:
        if problem.kind == "canonical":
            return f"\\boxed{{{problem.answer}}}"
        return f"\\boxed{{{problem.answer + 1}}}"


class OpenAICompatibleModel:
    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key_env: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache: Cache | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
        transport: httpx.AsyncBaseTransport | None = None,  # lets tests fake the server
    ):
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if api_key_env and not self.api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set (needed by {name})")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)
        self.retry_base_delay = 1.0

    def _payload(self, problem: Problem) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem.text},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    async def solve(self, problem: Problem, sample: int) -> str:
        payload = self._payload(problem)
        key = Cache.key(self.base_url, payload, sample)
        if self.cache and (hit := self.cache.get(key)) is not None:
            return hit

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                break
            except (httpx.HTTPStatusError, httpx.TransportError):
                if attempt == self.max_retries:
                    raise
                # exponential backoff: wait 1s, 2s, 4s, ... before retrying
                await asyncio.sleep(self.retry_base_delay * 2**attempt)

        if self.cache:
            self.cache.set(key, text)
        return text

    async def aclose(self) -> None:
        await self._client.aclose()


def build_model(spec: dict, cache: Cache | None) -> Model:
    kind = spec.get("type", "openai")
    if kind == "oracle":
        return OracleModel(spec.get("name", "oracle"))
    if kind == "memorizer":
        return MemorizerModel(spec.get("name", "memorizer"))
    if kind == "openai":
        return OpenAICompatibleModel(
            name=spec["name"],
            model=spec["model"],
            base_url=spec["base_url"],
            api_key_env=spec.get("api_key_env"),
            temperature=spec.get("temperature", 0.0),
            max_tokens=spec.get("max_tokens", 2048),
            cache=cache,
        )
    raise ValueError(f"Unknown model type: {kind}")
