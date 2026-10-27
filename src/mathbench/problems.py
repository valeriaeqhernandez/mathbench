"""Turn templates into concrete problems: one canonical version plus perturbed variants.

Variant kinds (this split is what makes the analysis interesting):
  canonical - original numbers, original wording
  rephrase  - original numbers, different wording   -> tests surface-text memorization
  numeric   - new numbers, original wording         -> tests answer memorization
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from mathbench.templates import Template

KINDS = ("canonical", "rephrase", "numeric")


@dataclass(frozen=True)
class Problem:
    id: str
    template_id: str
    topic: str
    kind: str
    text: str
    answer: int
    params: dict


def generate(templates: list[Template], n_numeric: int, seed: int) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    for t in templates:
        out.append(_make(t, t.canonical, 0, "canonical", 0))
        for i in range(1, len(t.phrasings)):
            out.append(_make(t, t.canonical, i, "rephrase", i - 1))

        seen = {tuple(sorted(t.canonical.items()))}
        attempts = 0
        while sum(p.template_id == t.id and p.kind == "numeric" for p in out) < n_numeric:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(f"{t.id}: could not sample {n_numeric} distinct variants")
            params = t.sample(rng)
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            idx = sum(p.template_id == t.id and p.kind == "numeric" for p in out)
            out.append(_make(t, params, 0, "numeric", idx))
    return out


def _make(t: Template, params: dict, phrasing: int, kind: str, idx: int) -> Problem:
    return Problem(
        id=f"{t.id}/{kind}-{idx}",
        template_id=t.id,
        topic=t.topic,
        kind=kind,
        text=t.render(params, phrasing),
        answer=t.solve(params),
        params=dict(params),
    )


def save(problems: list[Problem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in problems:
            f.write(json.dumps(asdict(p)) + "\n")


def load(path: Path) -> list[Problem]:
    with path.open() as f:
        return [Problem(**json.loads(line)) for line in f if line.strip()]
