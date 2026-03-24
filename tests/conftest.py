import pandas as pd
import pytest


@pytest.fixture()
def sample_matches() -> pd.DataFrame:
    """Small DataFrame of synthetic matches for unit testing."""
    return pd.DataFrame(
        {
            "tourney_id": ["2020-520", "2020-520", "2021-520"],
            "tourney_name": ["Roland Garros"] * 3,
            "surface": ["Clay"] * 3,
            "draw_size": [128] * 3,
            "tourney_level": ["G"] * 3,
            "tourney_date": pd.to_datetime(["2020-09-27", "2020-09-28", "2021-05-30"]),
            "match_num": [1, 2, 1],
            "winner_id": ["100001", "100002", "100001"],
            "winner_seed": [1.0, 3.0, 1.0],
            "winner_entry": [None, None, None],
            "winner_name": ["Player A", "Player B", "Player A"],
            "winner_hand": ["R", "L", "R"],
            "winner_ht": [185.0, 188.0, 185.0],
            "winner_ioc": ["ESP", "SRB", "ESP"],
            "winner_age": [34.0, 33.0, 35.0],
            "winner_rank": [2, 1, 3],
            "winner_rank_points": [9850, 10220, 9600],
            "loser_id": ["100003", "100001", "100002"],
            "loser_seed": [15.0, 1.0, 3.0],
            "loser_entry": [None, None, None],
            "loser_name": ["Player C", "Player A", "Player B"],
            "loser_hand": ["R", "R", "L"],
            "loser_ht": [183.0, 185.0, 188.0],
            "loser_ioc": ["AUT", "ESP", "SRB"],
            "loser_age": [27.0, 34.0, 34.0],
            "loser_rank": [15, 2, 1],
            "loser_rank_points": [2100, 9850, 10000],
            "score": ["6-3 6-2 6-4", "6-4 7-5 6-3", "6-3 6-4 7-6(4)"],
            "best_of": [5] * 3,
            "round": ["R128", "QF", "R64"],
            "minutes": [120.0, 150.0, 165.0],
            "w_ace": [8, 12, 6],
            "w_df": [2, 3, 1],
            "w_svpt": [80, 95, 88],
            "w_1stIn": [52, 60, 55],
            "w_1stWon": [40, 48, 42],
            "w_2ndWon": [15, 18, 17],
            "w_SvGms": [15, 18, 16],
            "w_bpSaved": [5, 7, 4],
            "w_bpFaced": [8, 10, 6],
            "l_ace": [3, 5, 10],
            "l_df": [4, 2, 4],
            "l_svpt": [75, 90, 85],
            "l_1stIn": [45, 55, 50],
            "l_1stWon": [30, 38, 35],
            "l_2ndWon": [12, 14, 15],
            "l_SvGms": [14, 17, 15],
            "l_bpSaved": [3, 6, 5],
            "l_bpFaced": [7, 9, 8],
        }
    )


@pytest.fixture()
def sample_players() -> pd.DataFrame:
    """Small player metadata table."""
    return pd.DataFrame(
        {
            "player_id": ["100001", "100002", "100003"],
            "name_first": ["Player", "Player", "Player"],
            "name_last": ["A", "B", "C"],
            "hand": ["R", "L", "R"],
            "dob": pd.to_datetime(["1986-06-03", "1987-05-22", "1993-09-18"]),
            "ioc": ["ESP", "SRB", "AUT"],
            "height": [185.0, 188.0, 183.0],
        }
    )
