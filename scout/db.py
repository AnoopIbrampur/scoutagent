"""DuckDB access. One file, a few tables, no ORM.

Tables:
  players       - raw season data (one row per player)
  planted_gems  - ground-truth labels for the synthetic test set. Kept in a
                  separate table so scoring code can never accidentally read them.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from scout.config import DB_PATH


def get_connection(path: Path | str = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB file, creating its folder if needed."""
    path = Path(path)
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def load_players(
    con: duckdb.DuckDBPyConnection,
    players: pd.DataFrame,
    planted: pd.DataFrame | None = None,
) -> int:
    """Replace the players table (and optional planted labels). Returns row count."""
    con.register("players_df", players)
    con.execute("CREATE OR REPLACE TABLE players AS SELECT * FROM players_df")
    con.unregister("players_df")
    if planted is not None:
        con.register("planted_df", planted)
        con.execute("CREATE OR REPLACE TABLE planted_gems AS SELECT * FROM planted_df")
        con.unregister("planted_df")
    return con.execute("SELECT count(*) FROM players").fetchone()[0]


def fetch_players(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return the full players table as a DataFrame."""
    return con.execute("SELECT * FROM players ORDER BY player_id").df()


def fetch_planted(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return planted labels (evaluation only; never used for scoring)."""
    return con.execute("SELECT * FROM planted_gems ORDER BY player_id").df()


def load_synergy(con: duckdb.DuckDBPyConnection, offense: pd.DataFrame, defense: pd.DataFrame,
                 prefix: str = "synergy") -> None:
    """Replace <prefix>_offense and <prefix>_defense (Synergy situational tables)."""
    for side, df in (("offense", offense), ("defense", defense)):
        con.register("syn_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {prefix}_{side} AS SELECT * FROM syn_df")
        con.unregister("syn_df")


def fetch_synergy(con: duckdb.DuckDBPyConnection, prefix: str = "synergy") -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return (offense, defense) Synergy tables, or (None, None) if they don't exist yet."""
    try:
        return (con.execute(f"SELECT * FROM {prefix}_offense").df(),
                con.execute(f"SELECT * FROM {prefix}_defense").df())
    except duckdb.CatalogException:
        return None, None
