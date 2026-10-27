"""Test the real HTTP client against a fake server (no network, no API key)."""

import asyncio

import httpx

from mathbench.cache import Cache
from mathbench.models import OpenAICompatibleModel
from mathbench.problems import generate
from mathbench.templates import TEMPLATES

PROBLEM = generate(TEMPLATES[:1], n_numeric=1, seed=0)[0]


def _reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def _model(handler, cache=None):
    m = OpenAICompatibleModel(
        name="fake",
        model="fake-1",
        base_url="http://fake/v1",
        cache=cache,
        transport=httpx.MockTransport(handler),
    )
    m.retry_base_delay = 0.0
    return m


def test_retries_on_rate_limit_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429) if len(calls) < 3 else _reply("\\boxed{5}")

    assert asyncio.run(_model(handler).solve(PROBLEM, 0)) == "\\boxed{5}"
    assert len(calls) == 3


def test_cache_prevents_second_call(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return _reply("\\boxed{9}")

    cache = Cache(tmp_path / "c.sqlite")
    asyncio.run(_model(handler, cache).solve(PROBLEM, 0))
    asyncio.run(_model(handler, cache).solve(PROBLEM, 0))
    assert len(calls) == 1
    # a different sample index is a different request, so it is not cached
    asyncio.run(_model(handler, cache).solve(PROBLEM, 1))
    assert len(calls) == 2
