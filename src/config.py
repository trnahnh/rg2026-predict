from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_ELO = ROOT / "data" / "elo"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

SACKMANN_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
TML_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"

SACKMANN_YEARS = range(2000, 2025)
TML_YEARS = range(2025, 2027)

TRAIN_END = 2022
VAL_YEARS = (2023, 2024)
TEST_YEAR = 2025

ELO_INITIAL = 1500.0
ELO_K_NEW = 32
ELO_K_ESTABLISHED = 24
ELO_NEW_THRESHOLD = 30

ROLLING_CLAY_MATCHES = 20
ROLLING_MIN_PERIODS = 5
FORM_WINDOW_12M_DAYS = 365
FORM_WINDOW_3M_DAYS = 90
FATIGUE_WINDOW_DAYS = 30
LAPLACE_PRIOR = 1

OPTUNA_N_TRIALS = 100
MC_SIMULATIONS = 10_000
BACKTEST_YEARS = range(2015, 2026)
RANDOM_SEED = 42

ROUND_MAP = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    "RR": 0, "BR": 0, "ER": 0,
}

TML_LEVEL_MAP = {
    "250": "A",
    "500": "A",
    "1000": "M",
    "2000": "G",
    "0": "F",
    "O": "O",
}

CANONICAL_MATCH_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name",
    "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "winner_rank", "winner_rank_points",
    "loser_id", "loser_seed", "loser_entry", "loser_name",
    "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "loser_rank", "loser_rank_points",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]
