"""Answer extraction and grading.

Extraction bugs are one of the most common ways LLM evals silently go wrong, so we
record *how* each answer was extracted. Always check the 'extract_method' breakdown
in a report before trusting an accuracy number.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANSWER_IS = re.compile(r"(?:final answer|answer)\s*(?:is|:|=)\s*\**\s*\$?(-?\d[\d,]*)", re.I)


def _last_boxed(text: str) -> str | None:
    """Return the contents of the last \\boxed{...}, handling nested braces."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def _to_int(s: str) -> int | None:
    s = s.replace(",", "").replace("$", "").replace("\\!", "").strip().rstrip(".")
    m = _NUM.fullmatch(s) or _NUM.search(s)
    if not m:
        return None
    val = m.group(0).replace(",", "")
    try:
        f = float(val)
    except ValueError:
        return None
    return int(f) if f.is_integer() else None


def extract_answer(response: str) -> tuple[int | None, str]:
    """Return (answer, method). method is 'boxed', 'answer_is', or 'none'."""
    boxed = _last_boxed(response)
    if boxed is not None:
        val = _to_int(boxed)
        if val is not None:
            return val, "boxed"
    matches = _ANSWER_IS.findall(response)
    if matches:
        val = _to_int(matches[-1])
        if val is not None:
            return val, "answer_is"
    # Deliberately no "last number in the text" fallback: it inflates accuracy
    # by accidentally matching numbers from the model's working.
    return None, "none"


def is_correct(predicted: int | None, gold: int) -> bool:
    return predicted is not None and predicted == gold
