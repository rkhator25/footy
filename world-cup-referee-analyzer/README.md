# 🟨 World Cup Referee Decision Analyzer

Do some World Cup referees show more cards than others — **once you account for the tournament, stage and extra time?**

Raw cards-per-match is a misleading way to rank referees: card rates changed enormously over the decades, and referees work different tournaments. This project builds a context-only baseline, measures how far each referee deviates from it, and tests whether those deviations are real or just luck. It ships with an interactive Streamlit dashboard.

**Live demo:** _add your Streamlit Cloud link here_

![Cards per match by tournament](assets/cards_per_match_by_tournament.png)

## Key findings

Data: **1,016 matches, 3,178 cards, 391 referees** (men's World Cups 1970–2022, women's 1991–2019).

- **Context matters more than the referee.** Cards per match ranged from about 1.3 (1978) to 5.2 (2006). Men's matches average 3.43 cards vs 2.20 in women's matches.
- **Four referees look unusual, but none survive correction.** Of the 55 referees with at least 5 matches, four have |z| ≥ 2 (e.g. Antonio Mateu Lahoz: 32 cards vs 18.7 expected). After Benjamini–Hochberg correction for testing 55 referees, none is significant (smallest q ≈ 0.21). With this many referees, a few extremes are expected by chance.
- **Referees do differ a little — as a group.** A permutation test (shuffling referees across matches within each tournament, 1,000 shuffles) gives p ≈ 0.02. Empirical-Bayes estimates put the real spread in card rate at roughly **±13%** around what context predicts.
- **Small samples dominate.** The median referee has only 2 matches, so shrinkage toward the expected rate is applied before ranking.
- **Style outliers (Isolation Forest):** Arturo Brizio Carter, Björn Kuipers and Antonio Mateu Lahoz stand out on a mix of card volume, dismissals and card timing.

![Observed vs expected cards](assets/observed_vs_expected.png)
![Permutation test](assets/permutation_test.png)

> A flag means "unusual relative to context", **not** "biased" or "wrong". Team styles, match flow and VAR are not in the model.

## Method

| Step | What it does |
|---|---|
| 1. Match table | Joins bookings → referee appearances → matches. Keeps tournaments with card data; matches with zero cards are kept as zeros. |
| 2. Baseline | Poisson regression: cards ~ tournament + stage + extra time. The referee is deliberately **not** a feature. |
| 3. Deviation | `z = (observed − expected) / sqrt(φ · expected)` with over-dispersion φ ≈ 1.14 estimated from the data. Benjamini–Hochberg q-values across referees. |
| 4. Shrinkage | Gamma–Poisson empirical Bayes pulls noisy ratios toward 1. |
| 5. Style outliers | Isolation Forest on ratio, dismissals per match, late-card share (after minute 75) and first-half share. |
| 6. Sanity check | Within-tournament permutation test on total referee-to-referee spread. |

## Project structure

```
├── app.py                    # Streamlit dashboard
├── src/
│   ├── data.py               # loading, joins, match table
│   └── analysis.py           # baseline, tests, shrinkage, Isolation Forest, permutation test
├── scripts/
│   ├── generate_charts.py    # rebuilds README charts + prints headline numbers
│   └── download_data.py      # refreshes the raw CSVs
├── tests/test_pipeline.py    # pytest suite
├── data/raw/                 # 5 CSVs from the Fjelstul World Cup Database
├── assets/                   # README images
└── .github/workflows/ci.yml  # runs tests on every push
```

## Run it locally

```bash
git clone https://github.com/<your-username>/world-cup-referee-analyzer.git
cd world-cup-referee-analyzer
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Other commands:

```bash
pytest                            # run tests
python scripts/generate_charts.py # regenerate charts and headline numbers
```

## Limitations and next steps

- Only the main referee is recorded (no VAR/assistants), and referee assignments are not random.
- Adding team effects (e.g. how card-prone the two teams are) would separate referee tendencies from team style.
- A hierarchical Bayesian model could replace the moment-based shrinkage.

## Data and license

Data: Joshua C. Fjelstul, Ph.D., *The Fjelstul World Cup Database* v1.2, © 2023 Joshua C. Fjelstul, Ph.D. Licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode). Source: <https://github.com/jfjelstul/worldcup>.
**Modifications:** used 5 tables (bookings, matches, referees, referee appearances, tournaments), filtered to tournaments with card data, joined bookings to referees, and aggregated to match and referee level.

Code is released under the MIT License (see `LICENSE`). The data files in `data/raw/` remain under CC BY-SA 4.0.
