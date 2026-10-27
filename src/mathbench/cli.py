"""Command-line entry point.

mathbench run configs/smoke.yaml       # generate problems, query models, write report
mathbench report results/<run>/records.jsonl   # rebuild a report from saved records
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from mathbench.cache import Cache
from mathbench.models import build_model
from mathbench.problems import generate, save
from mathbench.report import build_report
from mathbench.templates import get_templates


def cmd_run(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    run_dir = Path(cfg.get("output_dir", "results")) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    templates = get_templates(cfg.get("templates"))
    problems = generate(templates, n_numeric=cfg["n_numeric_variants"], seed=cfg["seed"])
    save(problems, run_dir / "problems.jsonl")
    print(f"Generated {len(problems)} problems from {len(templates)} templates")

    cache = Cache(Path(cfg.get("cache_path", ".cache/responses.sqlite")))
    models = [build_model(spec, cache) for spec in cfg["models"]]
    records = asyncio.run(
        _run(models, problems, cfg.get("n_samples", 1), cfg.get("concurrency", 8), run_dir)
    )

    (run_dir / "config.yaml").write_text(config_path.read_text())  # snapshot for reproducibility
    report = build_report(records)
    (run_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved to {run_dir}/")


async def _run(models, problems, n_samples, concurrency, run_dir):
    from mathbench.runner import run

    return await run(models, problems, n_samples, concurrency, run_dir / "records.jsonl")


def cmd_report(records_path: Path) -> None:
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    print(build_report(records))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mathbench")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Run a benchmark from a YAML config")
    p_run.add_argument("config", type=Path)
    p_rep = sub.add_parser("report", help="Rebuild a report from records.jsonl")
    p_rep.add_argument("records", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        cmd_run(args.config)
    else:
        cmd_report(args.records)


if __name__ == "__main__":
    main()
