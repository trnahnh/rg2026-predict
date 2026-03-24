"""Download raw CSV files from Sackmann and TML repositories."""

from pathlib import Path

import requests
from tqdm import tqdm

from src.config import DATA_RAW, SACKMANN_BASE, SACKMANN_YEARS, TML_BASE, TML_YEARS


def _download(url: str, dest: Path) -> Path:
    """Download a file if it doesn't already exist (or is empty)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def fetch_sackmann_matches(years: range, dest: Path) -> list[Path]:
    """Download atp_matches_YYYY.csv for each year."""
    paths = []
    for year in tqdm(years, desc="Sackmann matches"):
        url = f"{SACKMANN_BASE}/atp_matches_{year}.csv"
        paths.append(_download(url, dest / f"atp_matches_{year}.csv"))
    return paths


def fetch_sackmann_players(dest: Path) -> Path:
    """Download atp_players.csv."""
    return _download(f"{SACKMANN_BASE}/atp_players.csv", dest / "atp_players.csv")


def fetch_sackmann_rankings(dest: Path) -> list[Path]:
    """Download ranking files for 2000s, 2010s, 2020s, and current."""
    suffixes = ["00s", "10s", "20s", "current"]
    paths = []
    for suffix in tqdm(suffixes, desc="Sackmann rankings"):
        url = f"{SACKMANN_BASE}/atp_rankings_{suffix}.csv"
        paths.append(_download(url, dest / f"atp_rankings_{suffix}.csv"))
    return paths


def fetch_tml_matches(years: range, dest: Path) -> list[Path]:
    """Download YYYY.csv from TML for each year."""
    paths = []
    for year in tqdm(years, desc="TML matches"):
        url = f"{TML_BASE}/{year}.csv"
        paths.append(_download(url, dest / f"{year}.csv"))
    return paths


def fetch_tml_players(dest: Path) -> Path:
    """Download ATP_Database.csv from TML."""
    return _download(f"{TML_BASE}/ATP_Database.csv", dest / "ATP_Database.csv")


def fetch_all() -> None:
    """Fetch everything from both sources."""
    sackmann_dir = DATA_RAW / "sackmann"
    tml_dir = DATA_RAW / "tml"

    fetch_sackmann_matches(SACKMANN_YEARS, sackmann_dir)
    fetch_sackmann_players(sackmann_dir)
    fetch_sackmann_rankings(sackmann_dir)
    fetch_tml_matches(TML_YEARS, tml_dir)
    fetch_tml_players(tml_dir)

    print(f"All data fetched to {DATA_RAW}")


if __name__ == "__main__":
    fetch_all()
