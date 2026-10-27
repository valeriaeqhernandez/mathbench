"""Run every (model, problem, sample) concurrently and record graded results."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from mathbench.grading import extract_answer, is_correct
from mathbench.models import Model
from mathbench.problems import Problem


async def _one(model: Model, problem: Problem, sample: int, sem: asyncio.Semaphore) -> dict:
    async with sem:  # the semaphore caps how many requests are in flight at once
        t0 = time.perf_counter()
        try:
            response = await model.solve(problem, sample)
            error = None
        except Exception as e:  # record failures instead of crashing the whole run
            response, error = "", f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
    pred, method = extract_answer(response)
    return {
        "model": model.name,
        "problem_id": problem.id,
        "template_id": problem.template_id,
        "topic": problem.topic,
        "kind": problem.kind,
        "sample": sample,
        "gold": problem.answer,
        "pred": pred,
        "extract_method": method,
        "correct": is_correct(pred, problem.answer),
        "latency_s": round(latency, 3),
        "error": error,
        "response": response,
    }


async def run(
    models: list[Model],
    problems: list[Problem],
    n_samples: int,
    concurrency: int,
    out_path: Path,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    jobs = [_one(m, p, s, sem) for m in models for p in problems for s in range(n_samples)]
    records: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, fut in enumerate(asyncio.as_completed(jobs), 1):
            rec = await fut
            records.append(rec)
            f.write(json.dumps(rec) + "\n")  # stream to disk so a crash loses nothing
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)
    for m in models:
        if hasattr(m, "aclose"):
            await m.aclose()
    return records
