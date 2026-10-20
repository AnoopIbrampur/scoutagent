"""Pluggable ingestion layer.

Anything that can produce a players DataFrame matching PLAYER_COLUMNS is a
DataSource. The rest of the app (scoring, search, reports) only ever sees the
DuckDB table, so swapping synthetic data for real data is a one-class change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from scout.config import DIVISIONS, PLAYER_COLUMNS, POSITIONS, SEED_PLAYERS_CSV


class DataSourceError(ValueError):
    """Raised when a source returns data that breaks the column contract."""


def validate_players(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and categorical values; return columns in contract order."""
    missing = [c for c in PLAYER_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(f"missing columns: {missing}")
    bad_div = set(df["division"]) - set(DIVISIONS)
    if bad_div:
        raise DataSourceError(f"unknown divisions: {bad_div}")
    bad_pos = set(df["position"]) - set(POSITIONS)
    if bad_pos:
        raise DataSourceError(f"unknown positions: {bad_pos}")
    if df["player_id"].duplicated().any():
        raise DataSourceError("duplicate player_id values")
    return df.loc[:, list(PLAYER_COLUMNS)].reset_index(drop=True)


class DataSource(ABC):
    """Interface for anything that supplies player season data."""

    name: str = "base"

    @abstractmethod
    def fetch_players(self) -> pd.DataFrame:
        """Return one row per player, validated against PLAYER_COLUMNS."""


class SeedDataSource(DataSource):
    """Reads the synthetic CSV written by scripts/generate_seed_data.py."""

    name = "seed"

    def __init__(self, csv_path: Path = SEED_PLAYERS_CSV) -> None:
        self.csv_path = Path(csv_path)

    def fetch_players(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"{self.csv_path} not found. Run: python -m scripts.generate_seed_data"
            )
        return validate_players(pd.read_csv(self.csv_path))


class ApifyDataSource(DataSource):
    """Placeholder for real stats via an Apify actor (Phase 5c).

    TODO: Call the Apify API (apify-client) to run an actor, read its dataset
    items, and map fields onto PLAYER_COLUMNS, e.g.:
        actor "name"/"team"/"pos"  -> name / school / position
        "ht" like "6-2"            -> height_in = 6*12 + 2
        per-game box score fields  -> points_pg, rebounds_pg, ...
        conference lookup table    -> conference, conference_strength
    Attention fields (stars, followers, news) need separate sources.
    Check each source site's terms of service before running any actor.
    """

    name = "apify"

    def fetch_players(self) -> pd.DataFrame:
        raise NotImplementedError(
            "ApifyDataSource is a stub. See the class docstring for the field mapping."
        )
