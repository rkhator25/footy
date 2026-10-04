"""Referee card analysis.

Idea: raw cards-per-match is a poor way to compare referees, because referees
work different tournaments (card rates changed a lot over the decades) and
different stages. So we:

1. Fit a Poisson baseline that predicts cards for every match from context only
   (tournament, stage, extra time) -- the referee is deliberately NOT in the model.
2. For each referee, compare cards actually shown (O) with cards the baseline
   expected for the matches they took (E).
3. Test whether O - E is bigger than chance allows (over-dispersion-adjusted z,
   Benjamini-Hochberg corrected), shrink noisy ratios toward 1 (empirical Bayes),
   and look for unusual multi-feature "styles" with an Isolation Forest.
4. Permutation test: shuffle referees across matches within each tournament to
   check whether referees differ from each other by more than luck.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

MIN_MATCHES_DEFAULT = 5


# --------------------------------------------------------------------------- #
# 1. Context-only baseline
# --------------------------------------------------------------------------- #
def _design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[["tournament_id", "stage"]].astype(str)).astype(float)
    X["extra_time"] = df["extra_time"].astype(float).values
    return X


def fit_baseline(df: pd.DataFrame, target: str) -> tuple[np.ndarray, float]:
    """Return expected counts per match and the Pearson dispersion (phi >= 1)."""
    X, y = _design(df), df[target].to_numpy(dtype=float)
    model = PoissonRegressor(alpha=1e-4, max_iter=2000).fit(X, y)
    mu = np.clip(model.predict(X), 1e-6, None)
    dof = max(len(y) - (X.shape[1] + 1), 1)
    phi = max(1.0, float(((y - mu) ** 2 / mu).sum() / dof))
    return mu, phi


def add_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    phis = {}
    for target in ("cards", "dismissals"):
        mu, phi = fit_baseline(out, target)
        out[f"expected_{target}"] = mu
        phis[target] = phi
    return out, phis


# --------------------------------------------------------------------------- #
# 2. Referee-level deviation from the baseline
# --------------------------------------------------------------------------- #
def _bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def prior_strength(obs: np.ndarray, exp: np.ndarray, phi: float) -> tuple[float, float]:
    """Method-of-moments estimate of how much true referee rates vary.

    Model: O_i ~ overdispersed Poisson(theta_i * E_i), theta_i ~ Gamma(mean 1, var tau2).
    E[(O-E)^2] = phi*E + tau2*E^2  =>  tau2 = sum((O-E)^2 - phi*E) / sum(E^2).
    Returns (a, tau) where a = 1/tau2 is the prior strength (inf if tau2 <= 0).
    """
    tau2 = float((((obs - exp) ** 2) - phi * exp).sum() / (exp**2).sum())
    if tau2 <= 1e-6:
        return float("inf"), 0.0
    return 1.0 / tau2, float(np.sqrt(tau2))


def referee_table(
    df: pd.DataFrame, phis: dict[str, float], min_matches: int = MIN_MATCHES_DEFAULT
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-referee stats, deviation tests and shrunk ratios.

    `df` must already contain expected_cards / expected_dismissals.
    """
    g = df.groupby(
        ["referee_id", "referee_name", "referee_country", "confederation_code"],
        as_index=False,
    ).agg(
        matches=("match_id", "count"),
        tournaments=("tournament_id", "nunique"),
        cards=("cards", "sum"),
        yellows=("yellows", "sum"),
        dismissals=("dismissals", "sum"),
        late_cards=("late_cards", "sum"),
        first_half_cards=("first_half_cards", "sum"),
        expected_cards=("expected_cards", "sum"),
        expected_dismissals=("expected_dismissals", "sum"),
    )
    g["cards_per_match"] = g["cards"] / g["matches"]
    g["dismissals_per_match"] = g["dismissals"] / g["matches"]
    g["late_share"] = np.where(g["cards"] > 0, g["late_cards"] / g["cards"].clip(lower=1), np.nan)
    g["first_half_share"] = np.where(
        g["cards"] > 0, g["first_half_cards"] / g["cards"].clip(lower=1), np.nan
    )

    g["ratio"] = g["cards"] / g["expected_cards"]
    g["z"] = (g["cards"] - g["expected_cards"]) / np.sqrt(phis["cards"] * g["expected_cards"])
    g["p_value"] = 2 * stats.norm.sf(np.abs(g["z"]))

    eligible = g["matches"] >= min_matches
    g["eligible"] = eligible
    g["q_value"] = np.nan
    g.loc[eligible, "q_value"] = _bh_qvalues(g.loc[eligible, "p_value"].to_numpy())

    a, tau = prior_strength(
        g.loc[eligible, "cards"].to_numpy(float),
        g.loc[eligible, "expected_cards"].to_numpy(float),
        phis["cards"],
    )
    g["shrunk_ratio"] = 1.0 if not np.isfinite(a) else (g["cards"] + a) / (g["expected_cards"] + a)

    g["direction"] = np.select(
        [g["z"] >= 2, g["z"] <= -2], ["stricter than expected", "more lenient than expected"], "in line"
    )
    g.loc[~eligible, "direction"] = "too few matches"

    meta = {"tau": tau, "prior_strength": a, "phi_cards": phis["cards"], "n_eligible": int(eligible.sum())}
    return g, meta


# --------------------------------------------------------------------------- #
# 3. Multi-feature style outliers (Isolation Forest)
# --------------------------------------------------------------------------- #
STYLE_FEATURES = ["ratio", "dismissals_per_match", "late_share", "first_half_share"]


def add_isolation_forest(
    ref: pd.DataFrame, contamination: float = 0.05, seed: int = 42
) -> pd.DataFrame:
    out = ref.copy()
    out["style_score"] = np.nan
    out["style_outlier"] = False
    ok = out["eligible"] & out[STYLE_FEATURES].notna().all(axis=1)
    if ok.sum() < 20:
        return out
    X = StandardScaler().fit_transform(out.loc[ok, STYLE_FEATURES])
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    out.loc[ok, "style_score"] = -iso.score_samples(X)  # higher = more unusual
    out.loc[ok, "style_outlier"] = iso.predict(X) == -1
    return out


# --------------------------------------------------------------------------- #
# 4. Do referees differ by more than luck?
# --------------------------------------------------------------------------- #
def _sum_sq_z(df: pd.DataFrame, ref_ids: np.ndarray, phi: float, min_matches: int) -> float:
    tmp = pd.DataFrame({"r": ref_ids, "o": df["cards"].to_numpy(), "e": df["expected_cards"].to_numpy()})
    agg = tmp.groupby("r").agg(n=("o", "size"), o=("o", "sum"), e=("e", "sum"))
    agg = agg[agg["n"] >= min_matches]
    return float((((agg["o"] - agg["e"]) ** 2) / (phi * agg["e"])).sum())


def permutation_test(
    df: pd.DataFrame, phi: float, min_matches: int = MIN_MATCHES_DEFAULT, n_perm: int = 1000, seed: int = 42
) -> dict:
    """Shuffle referees among matches *within each tournament* and compare sum(z^2)."""
    rng = np.random.default_rng(seed)
    observed = _sum_sq_z(df, df["referee_id"].to_numpy(), phi, min_matches)
    groups = [np.asarray(v) for v in df.groupby("tournament_id").indices.values()]
    base = df["referee_id"].to_numpy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = base.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(base[idx])
        null[i] = _sum_sq_z(df, shuffled, phi, min_matches)
    p = (1 + (null >= observed).sum()) / (1 + n_perm)
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_95": float(np.quantile(null, 0.95)),
        "p_value": float(p),
        "null": null,
    }


# --------------------------------------------------------------------------- #
# Convenience pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(matches: pd.DataFrame, min_matches: int = MIN_MATCHES_DEFAULT):
    """matches -> (matches_with_expected, referee_table, meta)."""
    m, phis = add_baseline(matches)
    ref, meta = referee_table(m, phis, min_matches)
    ref = add_isolation_forest(ref)
    meta["phis"] = phis
    return m, ref, meta
