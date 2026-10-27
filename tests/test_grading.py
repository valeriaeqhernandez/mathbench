import pytest

from mathbench.grading import extract_answer, is_correct


@pytest.mark.parametrize(
    "text, expected, method",
    [
        ("so we get \\boxed{42}.", 42, "boxed"),
        ("\\boxed{1,234}", 1234, "boxed"),
        ("\\boxed{\\$17}", 17, "boxed"),
        ("first \\boxed{3}, actually \\boxed{5}", 5, "boxed"),  # last box wins
        ("\\boxed{\\text{12}}", 12, "boxed"),  # nested braces
        ("\\boxed{7.0}", 7, "boxed"),
        ("The final answer is 99.", 99, "answer_is"),
        ("Answer: **250**", 250, "answer_is"),
        ("I think it's around 12 or 13", None, "none"),  # no guessing from stray numbers
        ("\\boxed{2.5}", None, "none"),  # non-integer is not a valid answer here
        ("", None, "none"),
    ],
)
def test_extract(text, expected, method):
    assert extract_answer(text) == (expected, method)


def test_is_correct():
    assert is_correct(5, 5)
    assert not is_correct(None, 5)
    assert not is_correct(4, 5)
