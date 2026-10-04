"""World Cup Referee Decision Analyzer -- Streamlit dashboard.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    add_baseline,
    add_isolation_forest,
    permutation_test,
    referee_table,
)
from src.data import build_match_table, card_minutes

st.set_page_config(page_title="World Cup Referee Analyzer", page_icon="🟨", layout="wide")


# --------------------------------------------------------------------------- #
# Cached data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data and fitting baseline model...")
def load_all():
    matches, phis = add_baseline(build_match_table())
    return matches, phis, card_minutes()


@st.cache_data(show_spinner=False)
def compute_referees(competition: str, min_matches: int):
    matches, phis, _ = load_all()
    if competition != "Both":
        matches = matches[matches["competition"] == competition]
    ref, meta = referee_table(matches, phis, min_matches)
    return add_isolation_forest(ref), meta


@st.cache_data(show_spinner="Running permutation test (this takes a few seconds)...")
def compute_permutation(competition: str, min_matches: int, n_perm: int = 500):
    matches, phis, _ = load_all()
    if competition != "Both":
        matches = matches[matches["competition"] == competition]
    return permutation_test(matches, phis["cards"], min_matches, n_perm)


matches_all, PHIS, minutes_all = load_all()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Filters")
competition = st.sidebar.radio("Competition", ["Both", "Men's", "Women's"])
min_matches = st.sidebar.slider("Min. matches for a referee to be scored", 3, 12, 5)
st.sidebar.caption(
    "Data: Fjelstul World Cup Database (CC-BY-SA 4.0). Card data exists for men's "
    "World Cups from 1970 and all women's World Cups."
)

matches = matches_all if competition == "Both" else matches_all[matches_all["competition"] == competition]
ref, meta = compute_referees(competition, min_matches)
ref_scored = ref[ref["eligible"]]

st.title("🟨 World Cup Referee Decision Analyzer")
st.markdown(
    "Do some referees show more cards than others **once you account for the "
    "tournament, stage and extra time**? This dashboard compares cards actually shown "
    "with the number a context-only model expected."
)

tab_over, tab_ref, tab_cmp, tab_tour, tab_anom, tab_method = st.tabs(
    ["Overview", "Referee profile", "Compare referees", "Tournament view", "Anomaly board", "Methodology"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(matches):,}")
    c2.metric("Cards shown", f"{int(matches['cards'].sum()):,}")
    c3.metric("Cards per match", f"{matches['cards'].mean():.2f}")
    c4.metric("Referees", f"{matches['referee_id'].nunique():,}")

    by_t = (
        matches.groupby(["tournament_name", "year", "competition"], as_index=False)
        .agg(cards_per_match=("cards", "mean"), dismissals_per_match=("dismissals", "mean"), matches=("match_id", "count"))
        .sort_values("year")
    )
    fig = px.line(
        by_t, x="year", y="cards_per_match", color="competition", markers=True,
        hover_data=["tournament_name", "matches"], title="Cards per match by tournament",
    )
    fig.update_layout(yaxis_title="Cards per match", xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    by_stage = (
        matches.groupby("stage", as_index=False)
        .agg(cards_per_match=("cards", "mean"), matches=("match_id", "count"))
        .sort_values("cards_per_match", ascending=False)
    )
    fig2 = px.bar(by_stage, x="stage", y="cards_per_match", text="matches", title="Cards per match by stage (label = number of matches)")
    fig2.update_layout(xaxis_title=None, yaxis_title="Cards per match")
    st.plotly_chart(fig2, width="stretch")
    st.caption(
        "The rate has changed a lot over time, which is exactly why raw referee averages are misleading: "
        "a referee who worked several high-card tournaments looks strict even if they were typical for their games."
    )

# --------------------------------------------------------------------------- #
# Referee profile
# --------------------------------------------------------------------------- #
with tab_ref:
    options = ref.sort_values(["matches", "referee_name"], ascending=[False, True])
    labels = {
        r.referee_id: f"{r.referee_name} ({r.referee_country}) - {r.matches} matches"
        for r in options.itertuples()
    }
    rid = st.selectbox("Choose a referee", list(labels), format_func=labels.get)
    row = ref[ref["referee_id"] == rid].iloc[0]
    r_matches = matches[matches["referee_id"] == rid].sort_values("date")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Matches", int(row["matches"]))
    k2.metric("Cards shown", int(row["cards"]))
    k3.metric("Cards expected", f"{row['expected_cards']:.1f}")
    k4.metric("Observed / expected", f"{row['ratio']:.2f}")
    k5.metric("Shrunk ratio", f"{row['shrunk_ratio']:.2f}", help="Ratio pulled toward 1 in proportion to how little data there is.")
    if row["eligible"]:
        st.info(
            f"**{row['direction'].capitalize()}** (z = {row['z']:+.2f}, adjusted q = {row['q_value']:.2f}). "
            "A q-value above about 0.1 means the difference is not distinguishable from chance once you "
            "account for testing many referees."
        )
    else:
        st.warning(f"Only {int(row['matches'])} matches - below the minimum of {min_matches}, so no test is run.")

    by_tour = (
        r_matches.groupby("tournament_name", as_index=False)
        .agg(Observed=("cards", "sum"), Expected=("expected_cards", "sum"), Matches=("match_id", "count"))
    )
    long = by_tour.melt(id_vars=["tournament_name", "Matches"], value_vars=["Observed", "Expected"], var_name="Type", value_name="Cards")
    fig = px.bar(long, x="tournament_name", y="Cards", color="Type", barmode="group", title="Cards shown vs expected, per tournament")
    fig.update_layout(xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    mins = minutes_all[minutes_all["referee_id"] == rid]
    if len(mins):
        fig = go.Figure()
        fig.add_histogram(x=mins["minute_regulation"], histnorm="percent", name=row["referee_name"], xbins=dict(start=0, end=120, size=5), opacity=0.7)
        fig.add_histogram(x=minutes_all["minute_regulation"], histnorm="percent", name="All referees", xbins=dict(start=0, end=120, size=5), opacity=0.4)
        fig.update_layout(barmode="overlay", title="When cards are shown (% of cards per 5-minute bin)", xaxis_title="Minute", yaxis_title="% of cards")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Matches")
    st.dataframe(
        r_matches[["date", "match_label", "stage", "cards", "expected_cards", "yellows", "dismissals"]]
        .assign(date=lambda d: d["date"].dt.date, expected_cards=lambda d: d["expected_cards"].round(2))
        .rename(columns={"match_label": "Match", "expected_cards": "expected"}),
        hide_index=True, width="stretch",
    )

# --------------------------------------------------------------------------- #
# Compare referees
# --------------------------------------------------------------------------- #
with tab_cmp:
    pool = ref_scored.sort_values("matches", ascending=False)
    names = dict(zip(pool["referee_id"], pool["referee_name"] + " (" + pool["referee_country"] + ")"))
    default = list(names)[:3]
    picked = st.multiselect("Pick 2-6 referees", list(names), default=default, format_func=names.get, max_selections=6)
    if len(picked) >= 2:
        sel = ref[ref["referee_id"].isin(picked)].copy()
        sel["label"] = sel["referee_name"]
        fig = go.Figure()
        fig.add_bar(x=sel["label"], y=sel["ratio"], name="Observed / expected")
        fig.add_bar(x=sel["label"], y=sel["shrunk_ratio"], name="Shrunk ratio")
        fig.add_hline(y=1, line_dash="dash", annotation_text="expected for context")
        fig.update_layout(barmode="group", title="Cards relative to what the matches predicted", yaxis_title="Ratio")
        st.plotly_chart(fig, width="stretch")
        show = sel[["referee_name", "referee_country", "matches", "cards", "expected_cards", "cards_per_match", "dismissals_per_match", "late_share", "z", "q_value"]].round(2)
        st.dataframe(show.rename(columns={"expected_cards": "expected"}), hide_index=True, width="stretch")
        st.caption("Raw cards per match is in the table; the chart adjusts for which matches each referee actually took.")
    else:
        st.info("Select at least two referees.")

# --------------------------------------------------------------------------- #
# Tournament view
# --------------------------------------------------------------------------- #
with tab_tour:
    tours = matches.drop_duplicates("tournament_id").sort_values("year")
    tid = st.selectbox("Tournament", tours["tournament_id"], index=len(tours) - 1, format_func=lambda t: tours.set_index("tournament_id").loc[t, "tournament_name"])
    tm = matches[matches["tournament_id"] == tid]
    a, b, c = st.columns(3)
    a.metric("Matches", len(tm))
    b.metric("Cards per match", f"{tm['cards'].mean():.2f}")
    c.metric("Dismissals", int(tm["dismissals"].sum()))
    per_ref = (
        tm.groupby("referee_name", as_index=False)
        .agg(matches=("match_id", "count"), cards=("cards", "sum"), expected=("expected_cards", "sum"))
    )
    per_ref["cards_per_match"] = per_ref["cards"] / per_ref["matches"]
    per_ref["difference"] = per_ref["cards"] - per_ref["expected"]
    per_ref = per_ref.sort_values("difference")
    fig = px.bar(per_ref, x="difference", y="referee_name", orientation="h", hover_data=["matches", "cards", "expected"],
                 title="Cards shown minus cards expected, by referee (this tournament only)")
    fig.update_layout(yaxis_title=None, xaxis_title="Cards above / below expectation", height=max(350, 24 * len(per_ref)))
    st.plotly_chart(fig, width="stretch")
    st.caption("Within one tournament most referees have only 1-4 matches, so treat these differences as noise unless they are large.")

# --------------------------------------------------------------------------- #
# Anomaly board
# --------------------------------------------------------------------------- #
with tab_anom:
    st.subheader("Are referees different from each other at all?")
    perm = compute_permutation(competition, min_matches)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Referees scored", meta["n_eligible"])
    m2.metric("Permutation-test p", f"{perm['p_value']:.3f}", help="Chance of seeing this much referee-to-referee spread if referees were assigned to matches at random within each tournament.")
    m3.metric("Est. real spread in card rate", f"±{meta['tau'] * 100:.0f}%", help="Estimated standard deviation of true referee rates around the context-expected rate.")
    m4.metric("Overdispersion", f"{meta['phi_cards']:.2f}", help="1.0 = pure Poisson noise.")

    st.subheader("Statistical outliers (cards vs. context-expected)")
    if len(ref_scored):
        fig = px.scatter(
            ref_scored, x="expected_cards", y="cards", color="direction", size="matches", hover_name="referee_name",
            hover_data={"referee_country": True, "matches": True, "z": ":.2f", "q_value": ":.2f"},
            color_discrete_map={"stricter than expected": "#d62728", "more lenient than expected": "#1f77b4", "in line": "#9e9e9e"},
        )
        top = float(max(ref_scored["expected_cards"].max(), ref_scored["cards"].max())) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dash", color="black"))
        fig.update_layout(xaxis_title="Cards expected for the matches they took", yaxis_title="Cards actually shown")
        st.plotly_chart(fig, width="stretch")

        table = ref_scored.sort_values("z", ascending=False)[
            ["referee_name", "referee_country", "matches", "cards", "expected_cards", "ratio", "shrunk_ratio", "z", "q_value", "direction"]
        ].round(2)
        st.dataframe(table.rename(columns={"expected_cards": "expected"}), hide_index=True, width="stretch")

        n_sig = int((ref_scored["q_value"] < 0.10).sum())
        st.caption(f"{n_sig} referee(s) remain flagged after Benjamini-Hochberg correction at q < 0.10.")

        st.subheader("Unusual style (Isolation Forest)")
        st.caption("Looks at four features together: observed/expected ratio, dismissals per match, share of cards after the 75th minute, share in the first half.")
        style = ref_scored.dropna(subset=["style_score"])
        if len(style):
            fig = px.scatter(
                style, x="ratio", y="late_share", color="style_outlier", size="matches", hover_name="referee_name",
                hover_data={"dismissals_per_match": ":.2f", "first_half_share": ":.2f", "style_score": ":.2f"},
                color_discrete_map={True: "#ff7f0e", False: "#9e9e9e"},
            )
            fig.update_layout(xaxis_title="Cards observed / expected", yaxis_title="Share of cards after minute 75")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                style[style["style_outlier"]].sort_values("style_score", ascending=False)[
                    ["referee_name", "referee_country", "matches", "ratio", "dismissals_per_match", "late_share", "first_half_share", "style_score"]
                ].round(2),
                hide_index=True, width="stretch",
            )
    else:
        st.warning("No referees meet the minimum-match threshold with the current filters.")

# --------------------------------------------------------------------------- #
# Methodology
# --------------------------------------------------------------------------- #
with tab_method:
    st.markdown(
        f"""
### How it works
1. **Match table.** Every match in a tournament with card data (men's 1970-2022, all women's tournaments).
   Matches with zero cards are kept as zeros.
2. **Context-only baseline.** A Poisson regression predicts cards per match from tournament, stage and
   extra time. The referee is *not* an input, so each referee's expected total is what an average
   referee would have shown in the same games.
3. **Deviation test.** For each referee, `z = (observed - expected) / sqrt(phi x expected)`, where
   phi = {meta['phi_cards']:.2f} is the over-dispersion estimated from the data. P-values are corrected across referees
   with Benjamini-Hochberg.
4. **Shrinkage.** Ratios from small samples are noisy, so they are shrunk toward 1 using an
   empirical-Bayes (Gamma-Poisson) estimate of how much real referee rates vary.
5. **Style outliers.** An Isolation Forest (contamination = 5%) on four standardized features.
6. **Permutation test.** Referees are shuffled across matches within each tournament to see whether
   the spread between referees exceeds what luck produces.

### Caveats
- A flag means *unusual relative to context*, not *biased or wrong*. Card counts depend on the teams
  playing, their tactics and the match flow, none of which are in the baseline.
- Referees work few matches (median 2), so most cannot be distinguished from average.
- The database records the **main referee only**; VAR and assistants are not modelled.
- Testing dozens of referees guarantees some look extreme by chance - that is why corrected q-values are shown.

### Data
Joshua C. Fjelstul, Ph.D., *The Fjelstul World Cup Database* v1.2 (c) 2023, CC-BY-SA 4.0.
Source: https://github.com/jfjelstul/worldcup. Modified: filtered to card-era tournaments,
joined bookings to referees, and aggregated to match level.
"""
    )
