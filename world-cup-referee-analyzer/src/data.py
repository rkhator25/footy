"""Load the Fjelstul World Cup Database and build analysis-ready tables.

The raw `bookings` table has no referee column, so cards are joined to referees
through `referee_appearances` (one referee per match in this dataset).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

STAGE_MAP = {
    "group stage": "Group stage",
    "second group stage": "Second group stage",
    "final round": "Final round",
    "round of 16": "Round of 16",
    "quarter-final": "Quarter-final",
    "quarter-finals": "Quarter-final",
    "semi-final": "Semi-final",
    "semi-finals": "Semi-final",
    "third-place match": "Third-place match",
    "final": "Final",
}
KNOCKOUT = {"Round of 16", "Quarter-final", "Semi-final", "Third-place match", "Final"}


def _read(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / f"{name}.csv")


def load_bookings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per card, with the referee of that match attached."""
    bookings = _read("bookings", raw_dir)
    apps = _read("referee_appearances", raw_dir)[["match_id", "referee_id"]]
    bookings = bookings.merge(apps, on="match_id", how="left")
    bookings["late_card"] = (bookings["minute_regulation"] >= 75).astype(int)
    bookings["first_half_card"] = (bookings["minute_regulation"] <= 45).astype(int)
    return bookings


def load_referees(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    ref = _read("referees", raw_dir)
    ref["referee_name"] = ref["given_name"].str.strip() + " " + ref["family_name"].str.strip()
    ref = ref.rename(columns={"country_name": "referee_country"})
    return ref[["referee_id", "referee_name", "referee_country", "confederation_code", "female"]]


def build_match_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per match, restricted to tournaments where card data exists.

    Card data starts in 1970 (men) and covers every women's tournament, so we
    keep only tournaments that appear in `bookings`. Matches with zero cards
    are kept (as zeros) -- dropping them would bias every rate upward.
    """
    matches = _read("matches", raw_dir)
    tournaments = _read("tournaments", raw_dir)[["tournament_id", "year"]]
    apps = _read("referee_appearances", raw_dir)[["match_id", "referee_id"]]
    bookings = load_bookings(raw_dir)
    referees = load_referees(raw_dir)

    card_tournaments = set(bookings["tournament_id"].unique())
    matches = matches[matches["tournament_id"].isin(card_tournaments)].copy()

    per_match = bookings.groupby("match_id").agg(
        cards=("booking_id", "count"),
        yellows=("yellow_card", "sum"),
        dismissals=("sending_off", "sum"),
        late_cards=("late_card", "sum"),
        first_half_cards=("first_half_card", "sum"),
    )

    df = (
        matches.merge(tournaments, on="tournament_id", how="left")
        .merge(apps, on="match_id", how="left")
        .merge(referees, on="referee_id", how="left")
        .merge(per_match, on="match_id", how="left")
    )
    count_cols = ["cards", "yellows", "dismissals", "late_cards", "first_half_cards"]
    df[count_cols] = df[count_cols].fillna(0).astype(int)

    df["stage"] = df["stage_name"].map(STAGE_MAP).fillna(df["stage_name"])
    df["knockout"] = df["stage"].isin(KNOCKOUT).astype(int)
    df["competition"] = df["tournament_name"].str.contains("Women", case=False).map(
        {True: "Women's", False: "Men's"}
    )
    df["goals"] = df["home_team_score"] + df["away_team_score"]
    df["date"] = pd.to_datetime(df["match_date"])
    df["match_label"] = df["match_name"] + " (" + df["year"].astype(str) + ")"
    df = df.dropna(subset=["referee_id"]).reset_index(drop=True)

    keep = [
        "match_id", "tournament_id", "tournament_name", "year", "competition",
        "date", "match_label", "stage", "knockout", "extra_time", "goals",
        "referee_id", "referee_name", "referee_country", "confederation_code",
        "cards", "yellows", "dismissals", "late_cards", "first_half_cards",
    ]
    return df[keep]


def card_minutes(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Every card with its referee, minute and tournament (for timing plots)."""
    b = load_bookings(raw_dir)
    ref = load_referees(raw_dir)[["referee_id", "referee_name"]]
    b = b.merge(ref, on="referee_id", how="left")
    return b[
        ["booking_id", "match_id", "referee_id", "referee_name", "tournament_id",
         "tournament_name", "minute_regulation", "yellow_card", "sending_off"]
    ]
