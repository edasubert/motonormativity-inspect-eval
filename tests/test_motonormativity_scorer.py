import asyncio
import math

import pytest
from unittest.mock import Mock

from motonormativity.motonormativity import motonormativity_scorer


def _make_state(completion: str, pair_id: str = "fumes", statement_type: str = "a") -> Mock:
    state = Mock()
    state.output.completion = completion
    state.metadata = {"pair_id": pair_id, "statement_type": statement_type}
    return state


def _run(completion: str, pair_id: str = "fumes", statement_type: str = "a"):
    scorer_fn = motonormativity_scorer()
    return asyncio.run(scorer_fn(_make_state(completion, pair_id, statement_type), target=None))


@pytest.mark.parametrize("rating", range(1, 8))
def test_scorer_all_valid_ratings(rating):
    assert _run(str(rating)).value == pytest.approx(float(rating))


def test_scorer_extracts_rating_from_prose():
    assert _run("After consideration, my rating is 5.").value == pytest.approx(5.0)


def test_scorer_propagates_pair_id():
    assert _run("3", pair_id="noise").metadata["pair_id"] == "noise"


def test_scorer_propagates_statement_type():
    assert _run("3", statement_type="b").metadata["statement_type"] == "b"


def test_scorer_explanation_contains_pair_id_and_rating():
    score = _run("6", pair_id="fumes", statement_type="a")
    assert "fumes" in score.explanation
    assert "6" in score.explanation


def test_scorer_unparsable_returns_invalid_marker():
    score = _run("I cannot provide a numerical rating.")
    assert math.isnan(score.value)


def test_scorer_unparsable_explanation_contains_response():
    response = "I cannot provide a numerical rating."
    score = _run(response)
    assert response in score.explanation


def test_scorer_unparsable_still_has_metadata():
    score = _run("no digits", pair_id="fatalism", statement_type="b")
    assert score.metadata["pair_id"] == "fatalism"
    assert score.metadata["statement_type"] == "b"
