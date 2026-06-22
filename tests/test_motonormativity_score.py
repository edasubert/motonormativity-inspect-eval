"""Tests for motonormativity_by_construct metric."""
import math

import pytest
from inspect_ai.scorer import Score, SampleScore

from motonormativity.motonormativity import motonormativity_by_construct

# Known human reference values (Walker & te Brömmelstroet 2025, Table 2).
# Used here to avoid HF network calls in unit tests; production loads from the dataset.
_REF = {
    "fumes":          {"mean":  2.15, "sd": 1.87},
    "noise":          {"mean":  0.39, "sd": 1.60},
    "fatalism":       {"mean": -0.43, "sd": 1.53},
    "responsibility": {"mean": -0.29, "sd": 1.30},
    "subsidy":        {"mean": -0.91, "sd": 1.94},
}


def _ss(
    category: str,
    pair_id: str,
    statement_type: str,
    value: float,
) -> SampleScore:
    return SampleScore(
        score=Score(
            value=value,
            metadata={
                "category": category,
                "pair_id": pair_id,
                "statement_type": statement_type,
            },
        ),
        sample_id=f"{pair_id}_{statement_type}",
        sample_metadata={},
    )


def compute(scores):
    return motonormativity_by_construct(human_reference=_REF)(scores)


# ── Empty / missing data ──────────────────────────────────────────────────────


def test_empty_input_no_construct_keys():
    result = compute([])
    assert not any(k != "parse_failure_rate" for k in result)
    assert math.isnan(result["parse_failure_rate"])


def test_no_complete_pairs_no_construct_keys():
    # Only "a" side present — no complete pair
    result = compute([_ss("fumes", "fumes_0", "a", 5.0)])
    assert not any(k.startswith("fumes_") for k in result)


def test_nan_value_excluded():
    scores = [
        _ss("fumes", "fumes_0", "a", float("nan")),
        _ss("fumes", "fumes_0", "b", 4.0),
    ]
    # pair is incomplete → no construct data
    assert not any(k.startswith("fumes_") for k in compute(scores))


# ── Per-construct model_mean ──────────────────────────────────────────────────


def test_single_pair_model_mean():
    scores = [_ss("fumes", "fumes_0", "a", 6.0), _ss("fumes", "fumes_0", "b", 4.0)]
    result = compute(scores)
    assert result["fumes_model_mean"] == pytest.approx(2.0)


def test_single_pair_sd_and_se_nan_with_one_pair():
    result = compute([_ss("fumes", "fumes_0", "a", 6.0), _ss("fumes", "fumes_0", "b", 4.0)])
    assert math.isnan(result["fumes_sd_between_paraphrase"])
    assert math.isnan(result["fumes_se"])


def test_multiple_pairs_model_mean_sd_and_se():
    # fumes: para means [2, 4] → mean 3, sd_between = sqrt(2), se = sqrt(2)/sqrt(2) = 1
    scores = [
        _ss("fumes", "fumes_0", "a", 6.0),
        _ss("fumes", "fumes_0", "b", 4.0),
        _ss("fumes", "fumes_1", "a", 7.0),
        _ss("fumes", "fumes_1", "b", 3.0),
    ]
    result = compute(scores)
    assert result["fumes_model_mean"] == pytest.approx(3.0)
    assert result["fumes_sd_between_paraphrase"] > 0
    assert result["fumes_se"] == pytest.approx(result["fumes_sd_between_paraphrase"] / 2 ** 0.5)


def test_n_paraphrases_count():
    scores = [
        _ss("fumes", "fumes_0", "a", 6.0),
        _ss("fumes", "fumes_0", "b", 4.0),
        _ss("fumes", "fumes_1", "a", 7.0),
        _ss("fumes", "fumes_1", "b", 3.0),
    ]
    assert compute(scores)["fumes_n_paraphrases"] == pytest.approx(2.0)


# ── human reference values ────────────────────────────────────────────────────


def test_human_mean_matches_reference():
    scores = [_ss("fumes", "fumes_0", "a", 5.0), _ss("fumes", "fumes_0", "b", 3.0)]
    assert compute(scores)["fumes_human_mean"] == pytest.approx(_REF["fumes"]["mean"])


def test_human_sd_matches_reference():
    scores = [_ss("fumes", "fumes_0", "a", 5.0), _ss("fumes", "fumes_0", "b", 3.0)]
    assert compute(scores)["fumes_human_sd"] == pytest.approx(_REF["fumes"]["sd"])


# ── Construct-level independence ─────────────────────────────────────────────


def test_per_construct_means_not_distorted_by_unequal_pair_counts():
    # fumes: 2 pairs with diffs [2, 4] → construct mean 3.0
    # fatalism: 1 pair with diff [-2] → construct mean -2.0
    # Each construct is reported independently; pair count within a construct
    # only affects that construct's own mean/sd, not the others.
    scores = [
        _ss("fumes", "fumes_0", "a", 6.0),
        _ss("fumes", "fumes_0", "b", 4.0),
        _ss("fumes", "fumes_1", "a", 7.0),
        _ss("fumes", "fumes_1", "b", 3.0),
        _ss("fatalism", "fatalism_0", "a", 3.0),
        _ss("fatalism", "fatalism_0", "b", 5.0),
    ]
    result = compute(scores)
    assert result["fumes_model_mean"] == pytest.approx(3.0)
    assert result["fatalism_model_mean"] == pytest.approx(-2.0)


def test_incomplete_pair_excluded_from_construct():
    # fumes_1 only has "a" side → excluded; only fumes_0 contributes
    scores = [
        _ss("fumes", "fumes_0", "a", 6.0),
        _ss("fumes", "fumes_0", "b", 4.0),
        _ss("fumes", "fumes_1", "a", 7.0),  # no "b" partner
    ]
    result = compute(scores)
    assert result["fumes_model_mean"] == pytest.approx(2.0)
    assert result["fumes_n_paraphrases"] == pytest.approx(1.0)


def test_unknown_construct_has_no_human_keys():
    # A construct not in the reference should still report model stats
    # but no *_human_mean / *_human_sd keys
    scores = [_ss("unknown", "unknown_0", "a", 5.0), _ss("unknown", "unknown_0", "b", 3.0)]
    result = compute(scores)
    assert "unknown_model_mean" in result
    assert "unknown_human_mean" not in result
    assert "unknown_human_sd" not in result


# ── sd_within_paraphrase, parse_failure_rate, n_obs ──────────────────────────


def test_sd_within_paraphrase_nan_single_epoch():
    # Without epochs (single draw per paraphrase) within-paraphrase SD is NaN.
    scores = [_ss("fumes", "fumes_0", "a", 6.0), _ss("fumes", "fumes_0", "b", 4.0)]
    assert math.isnan(compute(scores)["fumes_sd_within_paraphrase"])


def test_n_obs_counts_complete_pairs():
    scores = [
        _ss("fumes", "fumes_0", "a", 6.0),
        _ss("fumes", "fumes_0", "b", 4.0),
        _ss("fumes", "fumes_1", "a", 7.0),
        _ss("fumes", "fumes_1", "b", 3.0),
    ]
    assert compute(scores)["fumes_n_obs"] == pytest.approx(2.0)


def test_parse_failure_rate_all_valid():
    scores = [_ss("fumes", "fumes_0", "a", 5.0), _ss("fumes", "fumes_0", "b", 3.0)]
    assert compute(scores)["parse_failure_rate"] == pytest.approx(0.0)


def test_parse_failure_rate_one_nan():
    scores = [
        _ss("fumes", "fumes_0", "a", float("nan")),
        _ss("fumes", "fumes_0", "b", 4.0),
    ]
    assert compute(scores)["parse_failure_rate"] == pytest.approx(0.5)
