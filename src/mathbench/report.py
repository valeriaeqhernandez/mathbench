"""Turn graded records into a Markdown report."""

from __future__ import annotations

from collections import Counter, defaultdict

from mathbench.stats import accuracy_ci, memorization_gap


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def build_report(records: list[dict]) -> str:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    lines = ["# mathbench results", ""]
    lines += [
        "## Accuracy by variant kind (95% cluster-bootstrap CI)",
        "",
        "| Model | Canonical | Rephrase | Numeric | Gap: canonical − numeric |",
        "|---|---|---|---|---|",
    ]
    for name, recs in sorted(by_model.items()):
        cells = []
        for kind in ("canonical", "rephrase", "numeric"):
            sub = [r for r in recs if r["kind"] == kind]
            if not sub:
                cells.append("–")
                continue
            m, lo, hi = accuracy_ci([r["correct"] for r in sub], [r["template_id"] for r in sub])
            cells.append(f"{_pct(m)} [{_pct(lo)}, {_pct(hi)}]")
        g, glo, ghi = memorization_gap(recs, "numeric")
        flag = " ⚠️" if glo > 0 else ""
        cells.append(f"{_pct(g)} [{_pct(glo)}, {_pct(ghi)}]{flag}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "⚠️ = the gap's CI excludes zero: evidence the model does better on original "
        "problems than on perturbed ones.",
        "",
        "## Accuracy by topic",
        "",
    ]
    topics = sorted({r["topic"] for r in records})
    lines += ["| Model | " + " | ".join(topics) + " |", "|---|" + "---|" * len(topics)]
    for name, recs in sorted(by_model.items()):
        row = []
        for t in topics:
            sub = [r["correct"] for r in recs if r["topic"] == t]
            row.append(_pct(sum(sub) / len(sub)) if sub else "–")
        lines.append(f"| {name} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Sanity checks (read these before trusting the numbers above)",
        "",
        "| Model | Extracted via \\boxed | via 'answer is' | Not extracted | API errors |",
        "|---|---|---|---|---|",
    ]
    for name, recs in sorted(by_model.items()):
        c = Counter(r["extract_method"] for r in recs)
        errs = sum(1 for r in recs if r["error"])
        lines.append(f"| {name} | {c['boxed']} | {c['answer_is']} | {c['none']} | {errs} |")
    lines.append("")
    return "\n".join(lines)
