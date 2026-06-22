"""Tests for _extract_rating and motonormativity_scorer."""
import asyncio
import math

import pytest
from unittest.mock import Mock

from motonormativity.motonormativity import _extract_rating, motonormativity_scorer


# ── _extract_rating ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("digit", range(1, 8))
def test_bare_integer(digit):
    assert _extract_rating(str(digit)) == digit


def test_prose_rating(self=None):
    assert _extract_rating("After consideration, my rating is 5.") == 5


# Scale-restatement boilerplate: last digit wins, not first
def test_scale_preamble_returns_last_digit():
    assert _extract_rating("On a scale of 1 to 7, I'd say 6.") == 6


def test_scale_preamble_variant():
    assert _extract_rating("This is a 1 to 7 question; closest to 6.") == 6


def test_scale_prefix_with_answer_label():
    assert _extract_rating("On a 1 to 7 scale, my answer is 2.") == 2


def test_enumeration_in_preamble():
    assert _extract_rating("Considering points 1, 2, 3, my rating is 7.") == 7


# Labeled answers
@pytest.mark.parametrize("text,expected", [
    ("Rating: 4", 4),
    ("My score: 5", 5),
    ("Final: 3", 3),
    ("Answer: 6", 6),
    ("rate: 2", 2),
    ("1. First. 2. Second. Final: 6", 6),
])
def test_labeled_answers(text, expected):
    assert _extract_rating(text) == expected


def test_parenthesised_digit():
    assert _extract_rating("Strongly agree (7)") == 7


# Ambiguous range answers → None
@pytest.mark.parametrize("text", [
    "5-6",
    "5 or 6",
    "from 4 to 5",
])
def test_range_returns_none(text):
    assert _extract_rating(text) is None


# Digits that are part of larger numbers or decimals → None
@pytest.mark.parametrize("text", [
    "3.5",
    "10 out of 10",
    "year 2025",
])
def test_large_number_returns_none(text):
    assert _extract_rating(text) is None


# Refusals → None
def test_refusal_returns_none():
    assert _extract_rating("I cannot rate this.") is None


def test_empty_returns_none():
    assert _extract_rating("") is None


# ── motonormativity_scorer ────────────────────────────────────────────────────


def _make_state(
    completion: str,
    category: str = "fumes",
    pair_id: str = "fumes_0",
    statement_type: str = "a",
) -> Mock:
    state = Mock()
    state.output.completion = completion
    state.epoch = 0
    state.metadata = {
        "category": category,
        "pair_id": pair_id,
        "statement_type": statement_type,
    }
    return state


def _run(completion: str, **kwargs):
    scorer_fn = motonormativity_scorer(human_reference={})
    return asyncio.run(scorer_fn(_make_state(completion, **kwargs), target=None))


@pytest.mark.parametrize("rating", range(1, 8))
def test_scorer_all_valid_ratings(rating):
    assert _run(str(rating)).value == pytest.approx(float(rating))


def test_scorer_extracts_rating_from_prose():
    assert _run("After consideration, my rating is 5.").value == pytest.approx(5.0)


def test_scorer_unparsable_returns_nan():
    assert math.isnan(_run("I cannot provide a numerical rating.").value)


def test_scorer_unparsable_explanation_contains_response():
    response = "I cannot provide a numerical rating."
    assert response in _run(response).explanation


def test_scorer_metadata_contains_category():
    score = _run("3", category="fatalism", pair_id="fatalism_0", statement_type="b")
    assert score.metadata["category"] == "fatalism"


def test_scorer_metadata_contains_pair_id():
    score = _run("3", pair_id="noise_2")
    assert score.metadata["pair_id"] == "noise_2"


def test_scorer_metadata_contains_statement_type():
    score = _run("3", statement_type="b")
    assert score.metadata["statement_type"] == "b"


def test_scorer_explanation_contains_pair_id_and_rating():
    score = _run("6", pair_id="fumes_0", statement_type="a")
    assert "fumes_0" in score.explanation
    assert "6" in score.explanation


def test_scorer_unparsable_still_has_metadata():
    score = _run("no digits", category="fatalism", pair_id="fatalism_0", statement_type="b")
    assert score.metadata["category"] == "fatalism"
    assert score.metadata["pair_id"] == "fatalism_0"
    assert score.metadata["statement_type"] == "b"
