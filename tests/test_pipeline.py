"""End-to-end tests with the offline fake models."""

import asyncio
import math

from mathbench.models import MemorizerModel, OracleModel
from mathbench.problems import generate, load, save
from mathbench.report import build_report
from mathbench.runner import run
from mathbench.stats import accuracy_ci, memorization_gap
from mathbench.templates import TEMPLATES


def test_generation_is_deterministic_and_distinct():
    a = generate(TEMPLATES, n_numeric=5, seed=1)
    b = generate(TEMPLATES, n_numeric=5, seed=1)
    assert a == b
    for t in TEMPLATES:
        numeric = [p for p in a if p.template_id == t.id and p.kind == "numeric"]
        assert len(numeric) == 5
        keys = {tuple(sorted(p.params.items())) for p in numeric}
        assert len(keys) == 5 and tuple(sorted(t.canonical.items())) not in keys
    assert len({p.id for p in a}) == len(a)


def test_save_load_roundtrip(tmp_path):
    probs = generate(TEMPLATES, n_numeric=2, seed=0)
    save(probs, tmp_path / "p.jsonl")
    assert load(tmp_path / "p.jsonl") == probs


def test_pipeline_detects_memorizer(tmp_path):
    probs = generate(TEMPLATES, n_numeric=4, seed=0)
    records = asyncio.run(run([OracleModel(), MemorizerModel()], probs, 1, 4, tmp_path / "r.jsonl"))
    oracle = [r for r in records if r["model"] == "oracle"]
    mem = [r for r in records if r["model"] == "memorizer"]

    assert all(r["correct"] for r in oracle)
    assert memorization_gap(oracle)[0] == 0.0

    gap, lo, hi = memorization_gap(mem)
    assert gap == 1.0 and lo == 1.0  # perfect on originals, zero on variants

    report = build_report(records)
    assert "memorizer" in report and "⚠️" in report


def test_accuracy_ci_brackets_mean():
    correct = [True, False, True, True, False, True] * 5
    clusters = [f"t{i % 6}" for i in range(len(correct))]
    m, lo, hi = accuracy_ci(correct, clusters)
    assert lo <= m <= hi
    assert math.isclose(m, 4 / 6)
