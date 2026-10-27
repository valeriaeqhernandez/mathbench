"""Check every solver against an independent brute-force computation.

A wrong solver silently corrupts every variant of a template, so this file matters most.
"""

import itertools
import random

import pytest

from mathbench.templates import TEMPLATES, get_templates


def brute_paths(m, n, a, b):
    # dynamic programming over the grid, with the forbidden point set to 0
    ways = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if (i, j) == (a, b):
                ways[i][j] = 0
            elif i == 0 and j == 0:
                ways[i][j] = 1
            else:
                ways[i][j] = (ways[i - 1][j] if i else 0) + (ways[i][j - 1] if j else 0)
    return ways[m][n]


def brute_squares(m, n):
    count = 0
    for x, y in itertools.product(range(m + 1), range(n + 1)):
        for k in range(1, max(m, n) + 1):
            if x + k <= m and y + k <= n:
                count += 1
    return count


def brute_same_color(r, b):
    from fractions import Fraction

    balls = ["R"] * r + ["B"] * b
    pairs = list(itertools.combinations(range(len(balls)), 2))
    same = sum(balls[i] == balls[j] for i, j in pairs)
    f = Fraction(same, len(pairs))
    return f.numerator + f.denominator


def brute_divisors(p, q, a, b):
    # Shrink exponents so brute force stays fast; the formula's correctness doesn't
    # depend on size. We check the solver on the same shrunken params (see test below).
    n = p ** min(a, 3) * q ** min(b, 3)
    return sum(1 for d in range(1, n + 1) if n % d == 0)


BRUTE = {
    "comb-lattice-avoid": lambda x: brute_paths(x["m"], x["n"], x["a"], x["b"]),
    "comb-grid-squares": lambda x: brute_squares(x["m"], x["n"]),
    "prob-same-color": lambda x: brute_same_color(x["r"], x["b"]),
    "nt-divisors": lambda x: brute_divisors(x["p"], x["q"], x["a"], x["b"]),
}


@pytest.mark.parametrize("tid", sorted(BRUTE))
def test_solver_matches_brute_force(tid):
    (t,) = get_templates([tid])
    rng = random.Random(123)
    for params in [t.canonical] + [t.sample(rng) for _ in range(15)]:
        if tid == "nt-divisors":
            params = {**params, "a": min(params["a"], 3), "b": min(params["b"], 3)}
        assert t.solve(params) == BRUTE[tid](params), params


@pytest.mark.parametrize("t", TEMPLATES, ids=lambda t: t.id)
def test_every_phrasing_renders(t):
    for i in range(len(t.phrasings)):
        text = t.render(t.canonical, i)
        assert "{" not in text, "unfilled placeholder"


@pytest.mark.parametrize("t", TEMPLATES, ids=lambda t: t.id)
def test_answers_are_nonnegative_integers(t):
    rng = random.Random(7)
    for params in [t.canonical] + [t.sample(rng) for _ in range(50)]:
        ans = t.solve(params)
        assert isinstance(ans, int) and ans >= 0, (params, ans)


def test_crt_answer_satisfies_congruences():
    (t,) = get_templates(["nt-crt"])
    rng = random.Random(0)
    for _ in range(20):
        x = t.sample(rng)
        n = t.solve(x)
        assert n % x["p"] == x["a"] and n % x["q"] == x["b"]
        assert all(not (k % x["p"] == x["a"] and k % x["q"] == x["b"]) for k in range(1, n))


def test_template_ids_unique():
    ids = [t.id for t in TEMPLATES]
    assert len(ids) == len(set(ids))
