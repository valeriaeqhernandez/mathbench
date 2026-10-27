# mathbench

**Do language models reason about math, or remember it?**

Benchmark scores can be inflated when test problems leak into training data. `mathbench`
measures this directly. It takes each problem and generates controlled *variants* that
keep the same math but change the surface:

| Variant | What changes | What a drop in accuracy suggests |
|---|---|---|
| `canonical` | nothing (the original problem) | baseline |
| `rephrase` | wording only | memorized the problem's text |
| `numeric` | numbers only (answer recomputed exactly) | memorized the answer |

Every problem is a **parametric template** with a solver, so every variant's answer is
computed by code and verified against an independent brute-force implementation in the
test suite. The headline metric is the **memorization gap**, accuracy(canonical) minus
accuracy(numeric), reported with a 95% **cluster bootstrap** confidence interval.

> Status: early development. Results below come from the offline sanity check.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                               # 40 tests, under a second
mathbench run configs/smoke.yaml     # offline sanity check, no API key needed
```

The smoke run uses two fake models. `oracle` is always right, and `memorizer` knows only
the original problems. A working pipeline must show a 100-point gap for `memorizer` and
zero for `oracle`:

| Model | Canonical | Rephrase | Numeric | Gap |
|---|---|---|---|---|
| memorizer | 100.0 | 0.0 | 0.0 | 100.0 ⚠️ |
| oracle | 100.0 | 100.0 | 100.0 | 0.0 |

## Running real models

Every model in `configs/real.yaml` uses the OpenAI-compatible chat API, which covers
OpenAI, OpenRouter, Together, vLLM, and local models via [Ollama](https://ollama.com).

```bash
export OPENROUTER_API_KEY=...        # or put it in .env (git-ignored)
mathbench run configs/real.yaml
```

Responses are cached in SQLite (`.cache/`), so re-running after a grading fix costs nothing.
Each run saves `problems.jsonl`, `records.jsonl` (every response), `config.yaml`, and
`report.md` to `results/<run_name>/`.

## Design choices

- **Integer answers only.** This removes most grading ambiguity, which is a major source of
  silent eval errors.
- **No "last number in the response" fallback** when extracting answers. That fallback
  inflates accuracy by matching numbers from the model's working. The report shows how
  every answer was extracted.
- **Cluster bootstrap over templates.** Variants of one template are correlated, so treating
  them as independent would make confidence intervals falsely narrow.

## Project layout

```
src/mathbench/
  templates.py   problem families: phrasings, canonical params, sampler, exact solver
  problems.py    generate canonical / rephrase / numeric variants
  models.py      OpenAI-compatible async client (retries, cache) + offline fake models
  runner.py      concurrent evaluation, streamed to disk
  grading.py     answer extraction and grading
  stats.py       bootstrap CIs and the memorization gap
  report.py      Markdown report
tests/           solver brute-force checks, extraction cases, end-to-end pipeline
```

## Roadmap

- [ ] Add templates whose canonical version is a well-known public competition problem
- [ ] Run 4–5 open and API models; publish results
- [ ] Failure analysis: where in the reasoning chain do errors begin on variants?
- [ ] Does longer reasoning reduce the gap?
- [ ] Use as the evaluation set for RL post-training experiments
