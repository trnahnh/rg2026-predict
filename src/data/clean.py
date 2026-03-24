"""Load, normalize, merge Sackmann + TML data, and output clean parquet files."""

import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.config import (
    CANONICAL_MATCH_COLS,
    DATA_PROCESSED,
    DATA_RAW,
    SACKMANN_YEARS,
    TML_LEVEL_MAP,
    TML_YEARS,
)

STAT_COLS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]

INCOMPLETE_SCORE_PATTERNS = ["W/O", "DEF", "RET", "Walkover", "Default", "ABN", "ABD"]


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents and extra whitespace."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join(name.lower().split())


def load_sackmann_matches(raw_dir: Path, years: range) -> pd.DataFrame:
    """Load and concatenate Sackmann match CSVs."""
    frames = []
    for year in years:
        path = raw_dir / "sackmann" / f"atp_matches_{year}.csv"
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d")
    matches["winner_id"] = matches["winner_id"].astype(str)
    matches["loser_id"] = matches["loser_id"].astype(str)

    for col in STAT_COLS + ["minutes", "winner_ht", "loser_ht", "winner_age", "loser_age"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    for col in ["winner_rank", "loser_rank"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce").astype("Int64")
    for col in ["winner_rank_points", "loser_rank_points"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    for col in ["winner_seed", "loser_seed"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    return matches[CANONICAL_MATCH_COLS]


def load_tml_matches(raw_dir: Path, years: range) -> pd.DataFrame:
    """Load TML match CSVs, keeping indoor column separately."""
    frames = []
    for year in years:
        path = raw_dir / "tml" / f"{year}.csv"
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d")
    matches["winner_id"] = matches["winner_id"].astype(str)
    matches["loser_id"] = matches["loser_id"].astype(str)
    matches["tourney_level"] = (
        matches["tourney_level"].astype(str).map(TML_LEVEL_MAP).fillna(matches["tourney_level"])
    )

    for col in STAT_COLS + ["minutes", "winner_ht", "loser_ht", "winner_age", "loser_age"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")
    for col in ["winner_rank", "loser_rank"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce").astype("Int64")
    for col in ["winner_rank_points", "loser_rank_points"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")
    for col in ["winner_seed", "loser_seed"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    indoor = matches["indoor"].copy() if "indoor" in matches.columns else None

    canonical = matches[[c for c in CANONICAL_MATCH_COLS if c in matches.columns]]
    if indoor is not None:
        canonical = canonical.copy()
        canonical["indoor"] = indoor
    return canonical


def load_sackmann_players(raw_dir: Path) -> pd.DataFrame:
    """Load atp_players.csv with normalized types."""
    df = pd.read_csv(raw_dir / "sackmann" / "atp_players.csv")
    df["player_id"] = df["player_id"].astype(str)
    df["dob"] = pd.to_datetime(df["dob"], format="%Y%m%d", errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["full_name"] = (df["name_first"].fillna("") + " " + df["name_last"].fillna("")).str.strip()
    return df


def load_tml_players(raw_dir: Path) -> pd.DataFrame:
    """Load ATP_Database.csv from TML."""
    df = pd.read_csv(raw_dir / "tml" / "ATP_Database.csv", encoding="latin-1")
    df["id"] = df["id"].astype(str).str.strip('"')
    df["birthdate"] = pd.to_datetime(df["birthdate"], format="%Y%m%d", errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    for col in ["player", "atpname"]:
        df[col] = df[col].astype(str).str.strip('"').str.strip()
    return df


def build_player_id_map(
    sackmann_players: pd.DataFrame,
    tml_players: pd.DataFrame,
    tml_match_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build bidirectional player ID mapping between Sackmann and TML.

    Only maps TML players that appear in match data (if tml_match_ids provided)
    to avoid O(n^2) blowup on the full 7.6K TML player database.

    Strategy:
    1. Vectorized merge on (normalized_name, dob) — fast exact match.
    2. Fuzzy name + same dob for remaining unmatched.
    3. Fuzzy name only for still-unmatched (high threshold, small search space).
    """
    sack = sackmann_players[["player_id", "full_name", "dob"]].copy()
    sack["norm_name"] = sack["full_name"].apply(_normalize_name)
    sack["dob_str"] = sack["dob"].dt.strftime("%Y%m%d").fillna("")

    tml = tml_players[["id", "player", "atpname", "birthdate"]].copy()
    tml["norm_name"] = tml["player"].apply(_normalize_name)
    tml["norm_atpname"] = tml["atpname"].apply(_normalize_name)
    tml["dob_str"] = tml["birthdate"].dt.strftime("%Y%m%d").fillna("")

    if tml_match_ids is not None:
        tml = tml[tml["id"].isin(tml_match_ids)].copy()

    mapped: list[dict] = []
    matched_sack: set[str] = set()
    matched_tml: set[str] = set()

    tml_with_dob = tml[tml["dob_str"] != ""]
    sack_with_dob = sack[sack["dob_str"] != ""]

    merge_name = pd.merge(
        tml_with_dob, sack_with_dob,
        left_on=["norm_name", "dob_str"],
        right_on=["norm_name", "dob_str"],
        how="inner", suffixes=("_tml", "_sack"),
    )
    merge_atpname = pd.merge(
        tml_with_dob, sack_with_dob,
        left_on=["norm_atpname", "dob_str"],
        right_on=["norm_name", "dob_str"],
        how="inner", suffixes=("_tml", "_sack"),
    )

    for df_merge in [merge_name, merge_atpname]:
        for _, row in df_merge.iterrows():
            tid = str(row["id"])
            sid = str(row["player_id"])
            if tid in matched_tml or sid in matched_sack:
                continue
            mapped.append({
                "sackmann_id": sid,
                "tml_id": tid,
                "canonical_name": row["full_name"],
                "dob": row["dob"],
                "match_method": "exact",
            })
            matched_sack.add(sid)
            matched_tml.add(tid)

    unmatched_tml = tml[~tml["id"].isin(matched_tml) & (tml["dob_str"] != "")]
    for _, t_row in unmatched_tml.iterrows():
        candidates = sack[
            (sack["dob_str"] == t_row["dob_str"])
            & (~sack["player_id"].isin(matched_sack))
        ]
        if candidates.empty:
            continue
        best_ratio, best_row = 0.0, None
        for _, s_row in candidates.iterrows():
            ratio = max(
                SequenceMatcher(None, s_row["norm_name"], t_row["norm_name"]).ratio(),
                SequenceMatcher(None, s_row["norm_name"], t_row["norm_atpname"]).ratio(),
            )
            if ratio > best_ratio:
                best_ratio, best_row = ratio, s_row
        if best_ratio > 0.75 and best_row is not None:
            mapped.append({
                "sackmann_id": best_row["player_id"],
                "tml_id": t_row["id"],
                "canonical_name": best_row["full_name"],
                "dob": best_row["dob"],
                "match_method": f"fuzzy_dob_{best_ratio:.2f}",
            })
            matched_sack.add(best_row["player_id"])
            matched_tml.add(t_row["id"])

    # Token index avoids O(n^2) full-scan against 66K Sackmann players
    sack_unmatched = sack[~sack["player_id"].isin(matched_sack)]
    sack_name_index: dict[str, list[int]] = {}
    for idx, row in sack_unmatched.iterrows():
        for token in row["norm_name"].split():
            sack_name_index.setdefault(token, []).append(idx)

    for _, t_row in tml[~tml["id"].isin(matched_tml)].iterrows():
        candidate_idxs: set[int] = set()
        for name_field in [t_row["norm_name"], t_row["norm_atpname"]]:
            for token in name_field.split():
                candidate_idxs.update(sack_name_index.get(token, []))

        best_ratio, best_row = 0.0, None
        for idx in candidate_idxs:
            if sack_unmatched.at[idx, "player_id"] in matched_sack:
                continue
            s_name = sack_unmatched.at[idx, "norm_name"]
            ratio = max(
                SequenceMatcher(None, s_name, t_row["norm_name"]).ratio(),
                SequenceMatcher(None, s_name, t_row["norm_atpname"]).ratio(),
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_row = sack_unmatched.loc[idx]
        if best_ratio > 0.90 and best_row is not None:
            mapped.append({
                "sackmann_id": best_row["player_id"],
                "tml_id": t_row["id"],
                "canonical_name": best_row["full_name"],
                "dob": best_row["dob"],
                "match_method": f"fuzzy_name_{best_ratio:.2f}",
            })
            matched_sack.add(best_row["player_id"])
            matched_tml.add(t_row["id"])

    return pd.DataFrame(mapped)


def reconcile_tml_ids(
    tml_matches: pd.DataFrame,
    id_map: pd.DataFrame,
) -> pd.DataFrame:
    """Replace TML player IDs with Sackmann IDs. New players get synthetic IDs."""
    tml_to_sack = dict(zip(id_map["tml_id"], id_map["sackmann_id"]))

    unmapped_ids = set()
    for col in ["winner_id", "loser_id"]:
        unmapped_ids.update(
            tml_matches.loc[~tml_matches[col].isin(tml_to_sack), col].unique()
        )

    next_synthetic = 900001
    for uid in sorted(unmapped_ids):
        tml_to_sack[uid] = str(next_synthetic)
        next_synthetic += 1

    result = tml_matches.copy()
    result["winner_id"] = result["winner_id"].map(tml_to_sack)
    result["loser_id"] = result["loser_id"].map(tml_to_sack)
    return result


def _is_incomplete(score: str) -> bool:
    """Check if a match score indicates walkover, retirement, etc."""
    if pd.isna(score):
        return True
    score_upper = str(score).upper()
    return any(p in score_upper for p in INCOMPLETE_SCORE_PATTERNS)


def merge_match_data(
    sackmann_matches: pd.DataFrame,
    tml_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Sackmann + TML into single DataFrame, deduplicated."""
    shared_cols = [c for c in CANONICAL_MATCH_COLS if c in tml_matches.columns]
    has_indoor = "indoor" in tml_matches.columns

    sack = sackmann_matches.copy()
    if has_indoor:
        sack["indoor"] = pd.NA

    tml = tml_matches[shared_cols + (["indoor"] if has_indoor else [])].copy()

    merged = pd.concat([sack, tml], ignore_index=True)
    merged = merged.sort_values(
        ["tourney_date", "tourney_id", "match_num"]
    ).reset_index(drop=True)

    merged = merged.drop_duplicates(
        subset=["tourney_date", "winner_id", "loser_id"], keep="first"
    )

    merged = merged[~merged["score"].apply(_is_incomplete)]

    merged = merged.dropna(subset=["winner_id", "loser_id"])

    valid_surfaces = {"Hard", "Clay", "Grass", "Carpet"}
    merged = merged[merged["surface"].isin(valid_surfaces)]

    return merged.reset_index(drop=True)


def load_rankings(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate Sackmann ranking files, filtered to 2000+."""
    suffixes = ["00s", "10s", "20s", "current"]
    frames = []
    for suffix in suffixes:
        path = raw_dir / "sackmann" / f"atp_rankings_{suffix}.csv"
        df = pd.read_csv(path)
        frames.append(df)
    rankings = pd.concat(frames, ignore_index=True)

    rankings.columns = ["ranking_date", "rank", "player_id", "points"]
    rankings["ranking_date"] = pd.to_datetime(rankings["ranking_date"], format="%Y%m%d")
    rankings["player_id"] = rankings["player_id"].astype(str)
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce").astype("Int64")
    rankings["points"] = pd.to_numeric(rankings["points"], errors="coerce")

    rankings = rankings[rankings["ranking_date"].dt.year >= 2000]
    return rankings.sort_values("ranking_date").reset_index(drop=True)


def clean_all() -> None:
    """Run full clean pipeline and output parquet files."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Loading Sackmann matches...")
    sack_matches = load_sackmann_matches(DATA_RAW, SACKMANN_YEARS)
    print(f"  {len(sack_matches):,} Sackmann matches loaded")

    print("Loading TML matches...")
    tml_matches = load_tml_matches(DATA_RAW, TML_YEARS)
    print(f"  {len(tml_matches):,} TML matches loaded")

    print("Loading player databases...")
    sack_players = load_sackmann_players(DATA_RAW)
    tml_players = load_tml_players(DATA_RAW)
    print(f"  {len(sack_players):,} Sackmann players, {len(tml_players):,} TML players")

    tml_unique_players = set(
        tml_matches["winner_id"].unique()
    ) | set(tml_matches["loser_id"].unique())

    print(f"Building player ID map for {len(tml_unique_players)} TML match players...")
    id_map = build_player_id_map(sack_players, tml_players, tml_match_ids=tml_unique_players)
    print(f"  {len(id_map):,} players mapped")
    coverage = len(set(id_map["tml_id"]) & tml_unique_players) / max(len(tml_unique_players), 1)
    print(f"  TML player coverage: {coverage:.1%}")

    if coverage < 0.80:
        print("  WARNING: Low ID coverage. Consider TML-only fallback.")

    print("Reconciling TML player IDs...")
    tml_matches = reconcile_tml_ids(tml_matches, id_map)

    print("Merging match data...")
    matches = merge_match_data(sack_matches, tml_matches)
    print(f"  {len(matches):,} total matches after merge + filter")

    print("Loading rankings...")
    rankings = load_rankings(DATA_RAW)
    print(f"  {len(rankings):,} ranking entries")

    print("Saving parquet files...")
    matches.to_parquet(DATA_PROCESSED / "matches.parquet", index=False)

    players = sack_players[["player_id", "name_first", "name_last", "hand", "dob", "ioc", "height"]]
    players.to_parquet(DATA_PROCESSED / "players.parquet", index=False)

    rankings.to_parquet(DATA_PROCESSED / "rankings.parquet", index=False)

    id_map.to_parquet(DATA_PROCESSED / "player_id_map.parquet", index=False)

    print(f"All files saved to {DATA_PROCESSED}")


if __name__ == "__main__":
    clean_all()
