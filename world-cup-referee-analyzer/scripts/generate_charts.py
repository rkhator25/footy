"""Generate the static charts and headline numbers used in the README.

Run from the repo root:  python scripts/generate_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import add_baseline, add_isolation_forest, permutation_test, referee_table
from src.data import build_match_table

OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
MIN_MATCHES = 5


def main() -> None:
    matches, phis = add_baseline(build_match_table())
    ref, meta = referee_table(matches, phis, MIN_MATCHES)
    ref = add_isolation_forest(ref)
    scored = ref[ref["eligible"]]

    # 1. Cards per match over time
    by_t = matches.groupby(["year", "competition"], as_index=False)["cards"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for comp, grp in by_t.groupby("competition"):
        ax.plot(grp["year"], grp["cards"], marker="o", label=comp)
    ax.set_ylabel("Cards per match")
    ax.set_title("Card rates changed a lot over time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cards_per_match_by_tournament.png", dpi=150)
    plt.close(fig)

    # 2. Observed vs expected
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = np.where(scored["z"] >= 2, "#d62728", np.where(scored["z"] <= -2, "#1f77b4", "#9e9e9e"))
    ax.scatter(scored["expected_cards"], scored["cards"], s=20 + 8 * scored["matches"], c=colors, alpha=0.8)
    lim = max(scored["expected_cards"].max(), scored["cards"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    for i, r in enumerate(scored[scored["z"].abs() >= 2].itertuples()):
        dx, dy = (-38, 6) if i % 2 else (6, 6)
        ax.annotate(r.referee_name.split()[-1], (r.expected_cards, r.cards), fontsize=8, xytext=(dx, dy), textcoords="offset points")
    ax.set_xlabel("Cards expected for the matches they took")
    ax.set_ylabel("Cards actually shown")
    ax.set_title(f"Referees with >= {MIN_MATCHES} matches (|z| >= 2 labelled)")
    fig.tight_layout()
    fig.savefig(OUT / "observed_vs_expected.png", dpi=150)
    plt.close(fig)

    # 3. Permutation test
    perm = permutation_test(matches, phis["cards"], MIN_MATCHES, n_perm=1000)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm["null"], bins=40, color="#9e9e9e", alpha=0.8, label="Shuffled referees")
    ax.axvline(perm["observed"], color="#d62728", lw=2, label=f"Actual (p = {perm['p_value']:.3f})")
    ax.set_xlabel("Total referee-to-referee spread (sum of z$^2$)")
    ax.set_title("Do referees differ more than luck allows?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "permutation_test.png", dpi=150)
    plt.close(fig)

    # Headline numbers
    print("matches:", len(matches), "| cards:", int(matches["cards"].sum()), "| referees:", matches["referee_id"].nunique())
    print("scored referees:", meta["n_eligible"], "| dispersion phi:", round(meta["phi_cards"], 2), "| tau:", round(meta["tau"], 3))
    print("|z|>=2:", int((scored["z"].abs() >= 2).sum()), "| q<0.10:", int((scored["q_value"] < 0.10).sum()))
    print("permutation:", {k: round(v, 3) for k, v in perm.items() if k != "null"})
    print("style outliers:", scored.loc[scored["style_outlier"], "referee_name"].tolist())
    cols = ["referee_name", "referee_country", "matches", "cards", "expected_cards", "ratio", "shrunk_ratio", "z", "q_value"]
    print(scored.sort_values("z", ascending=False)[cols].head(5).round(2).to_string(index=False))
    print(scored.sort_values("z")[cols].head(3).round(2).to_string(index=False))
    print(matches.groupby("competition")[["cards", "dismissals"]].mean().round(2))
    print(matches.groupby("year")["cards"].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
