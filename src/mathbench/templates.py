"""Parametric problem templates.

Each template is a problem *family*. It has:
  - one or more phrasings (text with {placeholders})
  - canonical parameters (the "original" version of the problem)
  - a sampler that draws new valid parameters (the perturbed versions)
  - a solver that computes the exact answer from any parameters

Because answers are computed by code, every variant is guaranteed correct.
That is the core trick that makes the benchmark trustworthy at scale.

TODO(Valeria): these seed templates are original and only illustrate the format.
For a real contamination signal, the canonical version of each template should be a
*well-known public problem* (e.g. a past competition problem a model has likely seen
in training). Add those in new files and register them in TEMPLATES.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd

Params = dict[str, int]


@dataclass(frozen=True)
class Template:
    id: str
    topic: str
    phrasings: tuple[str, ...]
    canonical: Params
    sample: Callable[[random.Random], Params]
    solve: Callable[[Params], int]

    def render(self, params: Params, phrasing: int = 0) -> str:
        return self.phrasings[phrasing].format(**params)


# ---------------------------------------------------------------- helpers


def _distinct_primes(rng: random.Random, k: int, pool=(2, 3, 5, 7, 11, 13)) -> list[int]:
    return rng.sample(list(pool), k)


def _digit_sum_count(n: int, s: int) -> int:
    return sum(1 for x in range(1, n + 1) if sum(map(int, str(x))) == s)


def _crt_smallest(a: int, p: int, b: int, q: int) -> int:
    n = 1
    while not (n % p == a and n % q == b):
        n += 1
    return n


def _squares_in_grid(m: int, n: int) -> int:
    return sum((m - k + 1) * (n - k + 1) for k in range(1, min(m, n) + 1))


def _same_color_m_plus_n(r: int, b: int) -> int:
    total = r + b
    p = Fraction(r * (r - 1) + b * (b - 1), total * (total - 1))
    return p.numerator + p.denominator


def _paths_avoiding(m: int, n: int, a: int, b: int) -> int:
    return comb(m + n, m) - comb(a + b, a) * comb(m - a + n - b, m - a)


# ---------------------------------------------------------------- samplers


def _sample_divisors(rng: random.Random) -> Params:
    p, q = _distinct_primes(rng, 2)
    return {"p": p, "q": q, "a": rng.randint(2, 9), "b": rng.randint(2, 9)}


def _sample_crt(rng: random.Random) -> Params:
    while True:
        p, q = rng.randint(5, 23), rng.randint(5, 23)
        if p != q and gcd(p, q) == 1:
            return {"a": rng.randint(1, p - 1), "p": p, "b": rng.randint(1, q - 1), "q": q}


def _sample_paths(rng: random.Random) -> Params:
    m, n = rng.randint(4, 8), rng.randint(4, 8)
    return {"m": m, "n": n, "a": rng.randint(1, m - 1), "b": rng.randint(1, n - 1)}


def _sample_arith(rng: random.Random) -> Params:
    d = rng.randint(3, 13)
    return {"n": rng.randint(100, 900), "r": rng.randint(1, d - 1), "d": d}


def _sample_balls(rng: random.Random) -> Params:
    return {"r": rng.randint(3, 15), "b": rng.randint(3, 15)}


# ---------------------------------------------------------------- templates

TEMPLATES: list[Template] = [
    Template(
        id="nt-divisors",
        topic="number theory",
        phrasings=(
            "How many positive divisors does {p}^{a} * {q}^{b} have?",
            "Let N = {p}^{a} * {q}^{b}. Find the number of positive integers that divide N.",
        ),
        canonical={"p": 2, "q": 3, "a": 4, "b": 6},
        sample=_sample_divisors,
        solve=lambda x: (x["a"] + 1) * (x["b"] + 1),
    ),
    Template(
        id="nt-power-mod",
        topic="number theory",
        phrasings=(
            "Find the remainder when {base}^{k} is divided by {m}.",
            "What is {base}^{k} mod {m}?",
        ),
        canonical={"base": 7, "k": 2026, "m": 100},
        sample=lambda r: {
            "base": r.randint(3, 19),
            "k": r.randint(500, 5000),
            "m": r.randint(50, 999),
        },
        solve=lambda x: pow(x["base"], x["k"], x["m"]),
    ),
    Template(
        id="nt-crt",
        topic="number theory",
        phrasings=(
            "Find the smallest positive integer n such that n leaves remainder {a} when divided "
            "by {p} and remainder {b} when divided by {q}.",
            "What is the least positive integer congruent to {a} mod {p} and to {b} mod {q}?",
        ),
        canonical={"a": 3, "p": 7, "b": 5, "q": 11},
        sample=_sample_crt,
        solve=lambda x: _crt_smallest(x["a"], x["p"], x["b"], x["q"]),
    ),
    Template(
        id="nt-digit-sum",
        topic="number theory",
        phrasings=(
            "How many integers from 1 to {n} inclusive have digits that sum to {s}?",
            "Count the positive integers not exceeding {n} whose digit sum equals {s}.",
        ),
        canonical={"n": 1000, "s": 10},
        sample=lambda r: {"n": r.choice([500, 999, 1500, 2000, 2500]), "s": r.randint(5, 20)},
        solve=lambda x: _digit_sum_count(x["n"], x["s"]),
    ),
    Template(
        id="comb-lattice-avoid",
        topic="combinatorics",
        phrasings=(
            "A path from (0,0) to ({m},{n}) moves one unit right or one unit up at each step. "
            "How many such paths do not pass through the point ({a},{b})?",
            "Count the up/right lattice paths from the origin to ({m},{n}) that avoid ({a},{b}).",
        ),
        canonical={"m": 6, "n": 6, "a": 3, "b": 3},
        sample=_sample_paths,
        solve=lambda x: _paths_avoiding(x["m"], x["n"], x["a"], x["b"]),
    ),
    Template(
        id="comb-grid-squares",
        topic="combinatorics",
        phrasings=(
            "A rectangle is divided into a {m} by {n} grid of unit squares. "
            "How many squares of any size appear in the grid?",
            "In an {m} x {n} grid of unit squares, count all squares (of every size) "
            "whose sides lie on grid lines.",
        ),
        canonical={"m": 8, "n": 8},
        sample=lambda r: {"m": r.randint(4, 12), "n": r.randint(4, 12)},
        solve=lambda x: _squares_in_grid(x["m"], x["n"]),
    ),
    Template(
        id="prob-same-color",
        topic="probability",
        phrasings=(
            "A bag contains {r} red balls and {b} blue balls. Two balls are drawn at random "
            "without replacement. The probability that both have the same color is m/n, where "
            "m and n are relatively prime positive integers. Find m + n.",
            "Two balls are removed at random from a bag holding {r} red and {b} blue balls. "
            "If the chance they match in color is m/n in lowest terms, what is m + n?",
        ),
        canonical={"r": 5, "b": 7},
        sample=_sample_balls,
        solve=lambda x: _same_color_m_plus_n(x["r"], x["b"]),
    ),
    Template(
        id="alg-arith-sum",
        topic="algebra",
        phrasings=(
            "Find the sum of all positive integers less than {n} that leave remainder {r} "
            "when divided by {d}.",
            "What is the sum of every positive integer below {n} that is congruent to {r} mod {d}?",
        ),
        canonical={"n": 200, "r": 3, "d": 7},
        sample=_sample_arith,
        solve=lambda x: sum(k for k in range(1, x["n"]) if k % x["d"] == x["r"]),
    ),
]


def get_templates(ids: list[str] | None = None) -> list[Template]:
    if not ids:
        return list(TEMPLATES)
    by_id = {t.id: t for t in TEMPLATES}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"Unknown template ids: {missing}")
    return [by_id[i] for i in ids]
