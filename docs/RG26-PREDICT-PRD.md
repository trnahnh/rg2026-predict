# RG26-Predict — Product Requirements Document

**Project:** Roland Garros 2026 Men's Singles Winner Prediction Model
**Author:** Andrea (Anh) Tran
**Created:** 2026-03-23
**Tournament Date:** May 18 – June 7, 2026
**Status:** Planning

---

## 1. Problem Statement

Roland Garros is a 7-round, 128-player single-elimination tournament on clay. Predicting the tournament winner requires compounding individual match probabilities through a full bracket — a single match prediction model is insufficient. The goal is to build an end-to-end ML pipeline that ingests historical ATP match data, engineers clay-specific features, trains a match outcome classifier, and simulates the RG 2026 draw via Monte Carlo to produce tournament-winner probabilities for every seeded and unseeded player.

---

## 2. Objectives

1. **Individual match prediction accuracy on clay holdout set** — ≥ 65% accuracy (baseline: always pick higher-ranked = ~62%).
2. **Tournament simulation producing calibrated win probabilities** — Top-5 predicted favorites include the actual winner in ≥ 60% of backtested years (2015–2025).
3. **Full pipeline reproducibility** — Single `make run` from raw data to final predictions (alias: `make all`).
4. **Portfolio-grade documentation and visualization** — README with architecture diagram, methodology, and interactive output.

---

## 3. Scope

### In Scope (MVP)

- ATP Men's Singles only
- Historical match data from 2000–2026 (training window configurable)
- Match-level binary classification: player A wins vs player B wins
- Monte Carlo bracket simulation (10,000+ draws)
- Static output: tournament win probability table + bracket visualization
- Backtesting framework against RG 2015–2025 results

### Out of Scope (Post-MVP)

- WTA / Women's Singles
- Live in-tournament updating (round-by-round re-simulation)
- Betting odds integration as a feature or benchmark
- Web dashboard / API serving predictions
- Point-by-point or shot-level data (Match Charting Project)

---

## 4. Data Sources

### 4.1 Historical Base: Jeff Sackmann / Tennis Abstract

- `JeffSackmann/tennis_atp` — Match results, player metadata, weekly rankings — CSV per year (`atp_matches_YYYY.csv`) — 1968–2024 (last updated ~2024 season).
- `JeffSackmann/tennis_atp` — Rankings files — CSV (`atp_rankings_*.csv`) — 1973–2024 (intermittent before 1985).

**License:** CC BY-NC-SA 4.0. Attribution required. Non-commercial use only.

**Limitation:** Sackmann's repo has not been updated past the 2024 season. Match stats for 2025 and early 2026 clay events (Monte Carlo, Madrid, Rome) are missing. This is critical — recent form features and Elo ratings will be stale without 2025–2026 data.

### 4.2 Live Data: TML-Database (TennisMyLife)

- `Tennismylife/TML-Database` — ATP match results with stats, live-updated — CSV per year (`YYYY.csv`) + single `ATP_Database.csv` — 1968–2026 (updated daily/weekly).
- `Tennismylife/TML-Database` — Player database with ATP IDs — `ATP_Database.csv` — Cross-referenceable with ATP website.

**Key advantages over Sackmann:**

- Has 2025 and 2026 match data (critical for current Elo + form features)
- Uses ATP player IDs (easier cross-referencing)
- Claims to fill gaps in Sackmann's data (e.g. Connors' full career wins)
- Indoor/outdoor column added
- Updated daily or more frequently

**License:** Based on Sackmann's CC BY-NC-SA 4.0 work. Non-commercial unless explicitly permitted. Additional data sourced from ATP official website.

**Column schema:** Same structure as Sackmann — `tourney_id`, `tourney_name`, `surface`, `draw_size`, `tourney_level`, `tourney_date`, `match_num`, `winner_id`, `winner_name`, `winner_hand`, `winner_ht`, `winner_ioc`, `winner_age`, `winner_rank`, `winner_rank_points`, `winner_seed`, `loser_id`, `loser_name`, `loser_hand`, `loser_ht`, `loser_ioc`, `loser_age`, `loser_rank`, `loser_rank_points`, `loser_seed`, `score`, `best_of`, `round`, `minutes`, `w_ace`, `w_df`, `w_svpt`, `w_1stIn`, `w_1stWon`, `w_2ndWon`, `w_SvGms`, `w_bpSaved`, `w_bpFaced` (and mirrored `l_*` columns for loser).

### 4.3 Data Strategy: Merge Pipeline

Use TML-Database as the **primary source** for 2025–2026 data and Sackmann as the **validated historical base** for 2000–2024. The merge strategy:

1. Load Sackmann CSVs for 2000–2024 (well-tested, community-vetted)
2. Load TML CSVs for 2025–2026 (live data, needed for current form/Elo)
3. Deduplicate on `(tourney_id, match_num)` or `(tourney_date, winner_id, loser_id)`
4. Reconcile player IDs (Sackmann uses custom IDs, TML uses ATP IDs) — build a mapping table via `(player_name, birth_date)` join
5. Prefer Sackmann stats where both sources overlap (more community vetting); prefer TML where Sackmann has gaps

**ID reconciliation is a Phase 1 blocker.** If the mapping is too noisy, fall back to TML as the sole source for simplicity.

### 4.4 Secondary: Betting Odds (Evaluation Only)

- `tennis-data.co.uk` — Historical match odds from multiple bookmakers.

Used strictly for calibration benchmarking — not as a model feature in MVP.

### 4.5 Derived: Elo Ratings

Compute custom Elo ratings from match history rather than using ATP ranking points directly. ATP points decay on a calendar basis and don't reflect recent form well. Elo captures relative strength more accurately.

---

## 5. Feature Engineering

Features are computed per player at the time of each match (no future leakage).

### 5.1 Core Features

- `elo_overall` — Global Elo rating (K=32 for new, K=24 for established) — Overall player strength.
- `elo_clay` — Surface-specific Elo (clay matches only) — Clay specialists vs all-courters.
- `elo_delta` — `player_elo_clay - opponent_elo_clay` — Relative strength signal.
- `rank` — ATP ranking at match date — Baseline strength.
- `rank_points` — ATP ranking points — Continuous ranking signal.
- `rank_diff` — `opponent_rank - player_rank` — Relative ranking gap.
- `age` — Player age at match date — Age curve effects.
- `height` — Player height in cm — Physical attribute.
- `hand` — Playing hand (R/L/U) encoded — Handedness matchup effects.

### 5.2 Form Features (Rolling Windows)

- `win_rate_clay_12m` — 12 months — Clay win% over last 12 months.
- `win_rate_clay_3m` — 3 months — Recent clay form.
- `matches_played_30d` — 30 days — Fatigue / match fitness signal.
- `matches_played_clay_season` — Current clay season (April–June) — Clay season readiness.
- `titles_clay_12m` — 12 months — Clay title count.

### 5.3 Head-to-Head Features

- `h2h_wins` — Total H2H wins vs this opponent.
- `h2h_clay_wins` — Clay-specific H2H wins.
- `h2h_total_matches` — Total H2H matches played.
- `h2h_win_rate` — H2H win percentage (smoothed with Laplace prior for small samples).

### 5.4 Tournament Context Features

- `round_number` — Encoded round (R128=1, R64=2, ..., F=7).
- `is_seeded` — Binary: player is seeded.
- `seed` — Seed number (0 if unseeded).
- `prev_rg_best` — Best previous RG result (round number).
- `rg_matches_won_career` — Career RG match wins.

### 5.5 Serving Features (Aggregated)

Computed as rolling averages over last N clay matches (default N=20, minimum 5).

- `first_serve_pct` — 1st serve in percentage.
- `first_serve_win_pct` — Points won on 1st serve.
- `second_serve_win_pct` — Points won on 2nd serve.
- `break_points_saved_pct` — Break points saved %.
- `ace_rate` — Aces per service game.
- `df_rate` — Double faults per service game.

---

## 6. Model Architecture

### 6.1 Match Prediction Model

**Primary model:** XGBoost binary classifier.

- Input: feature vector for (player_A, player_B) match context
- Output: P(player_A wins)
- Symmetry enforcement: each match generates two training rows (A vs B, B vs A) with flipped labels

**Hyperparameter search:** Optuna with expanding-window time-series CV within the training set (2000–2022). Folds use train on years < T, validate on year T, for T in {2018, 2019, 2020, 2021, 2022}. The held-out validation set (2023–2024) and test set (2025) are never seen during HPO.

**Baseline models for comparison:**

1. Always pick higher-ranked player
2. Logistic regression on Elo delta only
3. Logistic regression on full feature set

### 6.2 Tournament Simulation

**Method:** Monte Carlo simulation of the full 128-player bracket.

```text
For each simulation (N=10,000):
    Initialize bracket with seeded draw
    For each round (R128 → Final):
        For each match in round:
            Compute P(player_A wins) from model
            Sample outcome from Bernoulli(P)
            Advance winner
    Record tournament winner

Tournament_win_prob[player] = count(wins) / N
```

**Draw handling:**

- Use actual RG 2026 draw once released (late May)
- Before draw release: simulate seeded draw using ATP seedings + random unseeded placement
- Byes and qualifiers handled as placeholder entries

---

## 7. Tech Stack

- `Language` — Python 3.12 — Ecosystem (pandas, scikit-learn, xgboost).
- `Data` — pandas, polars (optional for perf) — Tabular data wrangling.
- `ML` — XGBoost, scikit-learn, Optuna — Gradient boosting + hyperparameter optimization.
- `Visualization` — matplotlib, seaborn, plotly — Static + interactive charts.
- `Elo computation` — Custom module — No good off-the-shelf tennis Elo library.
- `Orchestration` — Makefile — Single-command reproducibility.
- `Environment` — uv / pip + requirements.txt — Dependency management.
- `Version control` — Git + GitHub — `github.com/trnahnh/rg26-predict`.

---

## 8. Project Structure

```text
rg26-predict/
├── Makefile                    # make data, make train, make predict, make backtest
├── README.md
├── SYSTEM_DESIGN.md
├── requirements.txt
├── data/
│   ├── raw/                    # Sackmann CSVs (gitignored, fetched by make data)
│   ├── processed/              # Cleaned + feature-engineered parquet files
│   └── elo/                    # Precomputed Elo rating histories
├── src/
│   ├── data/
│   │   ├── fetch.py            # Clone/pull Sackmann repo
│   │   ├── clean.py            # Parse, filter, type-cast
│   │   └── features.py         # Feature engineering pipeline
│   ├── elo/
│   │   └── engine.py           # Elo rating computation (overall + per-surface)
│   ├── model/
│   │   ├── train.py            # XGBoost training + Optuna HPO
│   │   ├── evaluate.py         # Accuracy, log-loss, calibration plots
│   │   └── baseline.py         # Baseline models (rank, logistic)
│   ├── simulate/
│   │   ├── bracket.py          # Draw structure + seeding logic
│   │   └── montecarlo.py       # Monte Carlo tournament simulation
│   └── viz/
│       ├── bracket_viz.py      # Bracket probability visualization
│       └── feature_importance.py
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── models/                     # Saved model artifacts (.joblib)
├── outputs/                    # Final predictions, plots
└── tests/
    ├── test_elo.py
    ├── test_features.py
    └── test_simulation.py
```

---

## 9. Phases

### Phase 1 — Data Pipeline (Week 1)

- Fetch Sackmann `tennis_atp` repo (2000–2024) and TML-Database repo (2025–2026)
- Build player ID mapping table between Sackmann IDs and ATP IDs (join on name + birth date)
- Merge: Sackmann as base for 2000–2024, TML for 2025–2026
- Deduplicate overlapping records on `(tourney_date, winner_id, loser_id)`
- Parse and clean merged CSVs, filter to completed matches with valid stats
- Build player metadata lookup
- Output: `data/processed/matches.parquet`, `data/processed/player_id_map.parquet`

### Phase 2 — Elo Engine (Week 1–2)

- Implement Elo rating system with configurable K-factor
- Surface-specific Elo (separate ratings for clay/hard/grass)
- Compute historical Elo for every player at every match date
- Validate: Elo correlation with ATP ranking should be > 0.85
- Output: `data/elo/elo_history.parquet`

### Phase 3 — Feature Engineering (Week 2)

- Implement all features from Section 5
- Rolling window computations with strict temporal ordering (no leakage)
- H2H computation with Laplace smoothing
- Serving stat aggregation over last N clay matches
- Output: `data/processed/features.parquet`

### Phase 4 — Model Training + Evaluation (Week 2–3)

- Train/val split: train on 2000–2022, validate on 2023–2024, test on 2025
- Baseline models (rank-based, logistic on Elo delta)
- XGBoost with Optuna HPO (50–100 trials)
- Evaluation: accuracy, log-loss, calibration curve, feature importance
- Target: ≥ 65% accuracy on clay holdout

### Phase 5 — Tournament Simulation (Week 3)

- Implement 128-player bracket with seeding rules
- Monte Carlo simulation (10K iterations)
- Backtest against RG 2015–2025: for each year Y, train on data < Y, simulate RG Y, check if actual winner was in top-5 predicted. Note: the 2025 backtest retrains from scratch on pre-2025 data — it does not reuse the Phase 4 test-set model.
- Output: per-player tournament win probabilities

### Phase 6 — Visualization + Documentation (Week 3–4)

- Bracket probability heatmap
- Player win probability bar chart (top 20)
- Feature importance plot
- Calibration curve
- README with methodology, architecture diagram, results
- SYSTEM_DESIGN.md

### Phase 7 — RG 2026 Live Prediction (Late May)

- Fetch actual draw once released
- Run final simulation with latest Elo + form data
- Publish predictions to README / outputs

---

## 10. Risks + Mitigations

- `Sackmann data stops at 2024 — no 2025/2026 matches` — Impact: Stale Elo / form features, useless for current predictions. Mitigation: Use TML-Database for 2025–2026 data (updated daily).
- `Player ID mismatch between Sackmann (custom IDs) and TML (ATP IDs)` — Impact: Merge pipeline breaks, duplicate/missing players. Mitigation: Build mapping via `(name, birth_date)` join; if too noisy, use TML as sole source.
- `TML data less community-vetted than Sackmann` — Impact: Potential data quality issues in 2025–2026 matches. Mitigation: Sanity checks: stat totals, score parsing, rank consistency.
- `Injuries not captured in data` — Impact: Model predicts withdrawn players. Mitigation: Manual exclusion list before simulation.
- `Clay season form is volatile (small sample)` — Impact: High variance in predictions. Mitigation: Use longer rolling windows + regularization.
- `Draw not released until late May` — Impact: Can't finalize bracket sim. Mitigation: Use projected seedings for preliminary run.
- `Class imbalance in early rounds (favorites dominate)` — Impact: Model overconfident. Mitigation: Calibration via Platt scaling or isotonic regression.
- `Both sources' licenses prohibit commercial use` — Impact: Legal. Mitigation: Project is non-commercial, educational only. Attribution in README for both Sackmann and TML.

---

## 11. Resume Framing

> **RG26-Predict** — Python, XGBoost, Optuna, pandas
> Engineered **35+ temporal features** (surface Elo, rolling form, H2H, serving stats) from **25 years of ATP match data** (180K+ matches) with strict anti-leakage guarantees; trained XGBoost classifier achieving **X% accuracy** on clay holdout (vs 62% rank baseline); built **Monte Carlo bracket simulator** (10K iterations) that placed the actual RG winner in top-5 predicted in **Y/11 backtested years**

Fill X and Y with actual numbers post-training.

---

## 12. References

- Jeff Sackmann, Tennis Abstract: [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) (CC BY-NC-SA 4.0) — historical base, 1968–2024
- TennisMyLife, TML-Database: [Tennismylife/TML-Database](https://github.com/Tennismylife/TML-Database) — live-updated ATP data, 1968–2026
- TML interactive explorer: [stats.tennismylife.org/tennis-match-database](https://stats.tennismylife.org/tennis-match-database)
- Kovalchik, S. (2016). "Searching for the GOAT of tennis win prediction." Journal of Quantitative Analysis in Sports.
- Inspiration: Marantaya's Austin GP F1 prediction using FastF1 API
