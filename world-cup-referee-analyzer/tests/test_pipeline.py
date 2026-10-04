import numpy as np
import pandas as pd
import pytest

from src.analysis import _bh_qvalues, prior_strength, run_pipeline
from src.data import build_match_table, load_bookings


@pytest.fixture(scope="module")
def matches():
    return build_match_table()


def test_every_match_has_one_referee(matches):
    assert matches["match_id"].is_unique
    assert matches["referee_id"].notna().all()


def test_card_totals_match_raw_bookings(matches):
    assert matches["cards"].sum() == len(load_bookings())


def test_zero_card_matches_are_kept(matches):
    assert (matches["cards"] == 0).any()


def test_no_pre_card_era_matches(matches):
    assert matches.loc[matches["competition"] == "Men's", "year"].min() >= 1970


def test_baseline_reproduces_total_cards(matches):
    m, ref, meta = run_pipeline(matches)
    # A Poisson model with an intercept preserves the total count.
    assert m["expected_cards"].sum() == pytest.approx(m["cards"].sum(), rel=1e-3)
    assert meta["phi_cards"] >= 1.0


def test_referee_totals_are_consistent(matches):
    m, ref, _ = run_pipeline(matches)
    assert ref["cards"].sum() == m["cards"].sum()
    assert ref["q_value"].dropna().between(0, 1).all()


def test_bh_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    q = _bh_qvalues(p)
    assert (q >= p).all() and (q <= 1).all()
    assert list(q) == sorted(q)


def test_prior_strength_no_extra_variance():
    exp = np.array([10.0, 20.0, 30.0])
    obs = exp.copy()  # zero spread -> no evidence referees differ
    a, tau = prior_strength(obs, exp, phi=1.0)
    assert np.isinf(a) and tau == 0.0
