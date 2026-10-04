"""Re-download the raw CSVs from the Fjelstul World Cup Database.

The repo already ships these files, so you only need this to refresh them:
    python scripts/download_data.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
FILES = ["bookings", "matches", "referees", "referee_appearances", "tournaments"]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"{BASE}/{name}.csv"
        dest = OUT / f"{name}.csv"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    print(f"Saved {len(FILES)} files to {OUT}")


if __name__ == "__main__":
    main()
