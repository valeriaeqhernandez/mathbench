"""Statistics: accuracy with confidence intervals, and the memorization gap.

Why a *cluster* bootstrap over templates? Variants from the same template are not
independent: if a model can't do lattice paths, it fails every lattice-path variant.
Treating them as independent would make the confidence intervals falsely narrow.
So we resample whole templates, not individual problems.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def accuracy_ci(
    correct: list[bool], clusters: list[str], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean accuracy and a 95% cluster-bootstrap CI."""
    if not correct:
        return float("nan"), float("nan"), float("nan")
    by_cluster: dict[str, list[bool]] = defaultdict(list)
    for c, k in zip(correct, clusters, strict=True):
        by_cluster[k].append(c)
    groups = [np.array(v, dtype=float) for v in by_cluster.values()]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[i] = np.concatenate([groups[j] for j in pick]).mean()
    return (
        float(np.mean(correct)),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def memorization_gap(
    records: list[dict], variant_kind: str = "numeric", n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Accuracy(canonical) - Accuracy(variant_kind), paired by template, with a 95% CI.

    Positive gap = the model does better on the original than on perturbed versions,
    which is evidence of memorization. A CI that excludes 0 means the gap is real.
    """
    canon: dict[str, list[float]] = defaultdict(list)
    var: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["kind"] == "canonical":
            canon[r["template_id"]].append(float(r["correct"]))
        elif r["kind"] == variant_kind:
            var[r["template_id"]].append(float(r["correct"]))
    tids = sorted(set(canon) & set(var))
    if not tids:
        return float("nan"), float("nan"), float("nan")
    # Per-template gap, then average over templates (each template weighs the same).
    diffs = np.array([np.mean(canon[t]) - np.mean(var[t]) for t in tids])
    rng = np.random.default_rng(seed)
    boots = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)])
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
