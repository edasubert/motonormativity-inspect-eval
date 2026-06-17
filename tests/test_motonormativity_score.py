import pytest
from unittest.mock import Mock
from inspect_ai.scorer import Score, SampleScore

from motonormativity.motonormativity import motonormativity_score


def _sample(pair_id: str, statement_type: str, value: float | None) -> SampleScore:
    return SampleScore(
        score=Score(
            value=value,
            metadata={"pair_id": pair_id, "statement_type": statement_type},
        ),
        sample_id=f"{pair_id}_{statement_type}",
        sample_metadata={"pair_id": pair_id, "statement_type": statement_type},
    )


def compute(scores):
    return motonormativity_score()(scores)


def test_single_pair_positive():
    scores = [_sample("fumes", "a", 6.0), _sample("fumes", "b", 4.0)]
    assert compute(scores) == pytest.approx(2.0)


def test_single_pair_zero():
    scores = [_sample("fumes", "a", 4.0), _sample("fumes", "b", 4.0)]
    assert compute(scores) == pytest.approx(0.0)


def test_single_pair_negative():
    scores = [_sample("fumes", "a", 3.0), _sample("fumes", "b", 6.0)]
    assert compute(scores) == pytest.approx(-3.0)


def test_mean_across_multiple_pairs():
    # pair1: a=7, b=3 → diff=4; pair2: a=5, b=6 → diff=-1; mean=1.5
    scores = [
        _sample("pair1", "a", 7.0),
        _sample("pair1", "b", 3.0),
        _sample("pair2", "a", 5.0),
        _sample("pair2", "b", 6.0),
    ]
    assert compute(scores) == pytest.approx(1.5)


def test_incomplete_pair_excluded():
    # pair1 complete; pair2 missing statement_b — only pair1 contributes
    scores = [
        _sample("pair1", "a", 6.0),
        _sample("pair1", "b", 4.0),
        _sample("pair2", "a", 7.0),
    ]
    assert compute(scores) == pytest.approx(2.0)


def test_order_independent():
    # metric should give the same result regardless of which statement comes first
    scores_ab = [_sample("fumes", "a", 6.0), _sample("fumes", "b", 4.0)]
    scores_ba = [_sample("fumes", "b", 4.0), _sample("fumes", "a", 6.0)]
    assert compute(scores_ab) == compute(scores_ba)


def test_empty_input_returns_zero():
    assert compute([]) == pytest.approx(0.0)


def test_non_numeric_value_excluded():
    # Score.value can be a string; float("not-a-number") raises ValueError → pair excluded
    scores = [
        _sample("fumes", "a", "not-a-number"),
        _sample("fumes", "b", 4.0),
    ]
    assert compute(scores) == pytest.approx(0.0)


def test_none_sample_metadata_skipped():
    # sample_metadata=None is skipped entirely; paired sample still excluded (incomplete pair)
    mock_sample = Mock()
    mock_sample.sample_metadata = None
    scores = [mock_sample, _sample("fumes", "b", 4.0)]
    assert compute(scores) == pytest.approx(0.0)
