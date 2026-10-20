"""Search over scored players behind a swappable interface.

DuckDBSearchBackend loads players once, scores them in memory, and answers
filter queries with SQL against that scored table. An Elasticsearch backend
(Phase 5) only has to implement the same two methods.

Ranking:
  base       = Underrated or Translation score (per filters.sort_by)
  emphasis   = mean D1-percentile of the stats behind the requested skills
  rank_score = base                          if no emphasis requested
               0.7 * base + 0.3 * emphasis   otherwise
When sorting by Underrated, players below SCORING_CONFIG["min_translation_to_rank"]
are excluded, and the response says how many were hidden.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from scout.config import DB_PATH, SCORING_CONFIG
from scout.db import fetch_players, get_connection
from scout.query_parser import SearchFilters, inches_to_str
from scout.scoring import explain_player, score_players

# Which percentile columns each skill tag maps to. tov_per40 is inverted.
EMPHASIS_STATS: dict[str, list[str]] = {
    "passing": ["ast_per40", "ast_to"],
    "shooting": ["three_pct_adj", "ts_pct", "ft_pct"],
    "scoring": ["pts_per40", "ts_pct"],
    "rebounding": ["reb_per40"],
    "defense": ["stl_per40", "blk_per40"],
    "rim_protection": ["blk_per40", "reb_per40"],
    "efficiency": ["ts_pct", "ast_to"],
    "ball_security": ["ast_to", "tov_per40"],
}
EMPHASIS_WEIGHT = 0.3


def to_python(value: Any) -> Any:
    """numpy/pandas scalars -> plain Python (JSON-safe); NaN -> None."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def emphasis_score(df: pd.DataFrame, emphasis: list[str]) -> pd.Series:
    """Mean percentile across the stats behind the requested skills (0-100)."""
    cols = []
    for tag in emphasis:
        for stat in EMPHASIS_STATS[tag]:
            pct = df[f"pct_{stat}"]
            cols.append(100 - pct if stat == "tov_per40" else pct)
    return pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(float("nan"), index=df.index)


def player_summary(row: pd.Series) -> dict[str, Any]:
    """Compact JSON-safe view of a scored player for result lists."""
    keys = ["player_id", "name", "school", "conference", "division", "position",
            "height_in", "class_year", "translation_score", "attention_score", "underrated_score"]
    out = {k: to_python(row[k]) for k in keys}
    for k in ("translation_score", "attention_score", "underrated_score"):
        out[k] = round(out[k], 1)
    out["height"] = inches_to_str(int(row["height_in"]))
    return out


class SearchBackend(ABC):
    """Interface every search implementation must provide."""

    @abstractmethod
    def search(self, filters: SearchFilters, limit: int = 20) -> dict[str, Any]:
        """Return {"results": [...], "total_matches", "excluded_by_gate", "where_sql"}."""

    @abstractmethod
    def get_player(self, player_id: str) -> pd.Series | None:
        """Return the full scored row for one player, or None."""


class DuckDBSearchBackend(SearchBackend):
    """Scores all players on load, then filters with DuckDB SQL over the scored table."""

    def __init__(self, db_path: Path | str = DB_PATH, players: pd.DataFrame | None = None) -> None:
        if players is None:
            with get_connection(db_path, read_only=True) as con:
                players = fetch_players(con)
        self.scored = score_players(players)
        self._con = duckdb.connect()  # in-memory; avoids file locks with other processes
        self._con.register("player_scores", self.scored)

    def _where(self, f: SearchFilters) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if f.division:
            clauses.append(f"division IN ({', '.join('?' * len(f.division))})")
            params += f.division
        if f.position:
            clauses.append(f"position IN ({', '.join('?' * len(f.position))})")
            params += f.position
        if f.height_min_in:
            clauses.append("height_in >= ?")
            params.append(f.height_min_in)
        if f.height_max_in:
            clauses.append("height_in <= ?")
            params.append(f.height_max_in)
        if f.text_match:
            clauses.append("(name ILIKE ? OR school ILIKE ?)")
            params += [f"%{f.text_match}%"] * 2
        return (" AND ".join(clauses) or "TRUE"), params

    def search(self, filters: SearchFilters, limit: int = 20) -> dict[str, Any]:
        where, params = self._where(filters)
        matches = self._con.execute(f"SELECT * FROM player_scores WHERE {where}", params).df()
        total = len(matches)

        gate = SCORING_CONFIG.get("min_translation_to_rank")
        excluded = 0
        if filters.sort_by == "underrated" and gate is not None:
            keep = matches["translation_score"] >= gate
            excluded = int((~keep).sum())
            matches = matches[keep]

        base = matches["underrated_score" if filters.sort_by == "underrated" else "translation_score"]
        if filters.stat_emphasis:
            emph = emphasis_score(matches, filters.stat_emphasis)
            rank = (1 - EMPHASIS_WEIGHT) * base + EMPHASIS_WEIGHT * emph
        else:
            emph, rank = pd.Series(float("nan"), index=matches.index), base
        matches = matches.assign(emphasis_score=emph, rank_score=rank).sort_values("rank_score", ascending=False)

        results = []
        for _, row in matches.head(limit).iterrows():
            exp = explain_player(row)
            item = player_summary(row)
            item.update(
                rank_score=round(float(row["rank_score"]), 1),
                emphasis_score=None if pd.isna(row["emphasis_score"]) else round(float(row["emphasis_score"]), 1),
                top_factor=exp["top_factors"][0]["text"],
                explanation=exp,
            )
            results.append(item)
        return {"results": results, "total_matches": total, "excluded_by_gate": excluded,
                "gate": gate if filters.sort_by == "underrated" else None, "where_sql": where}

    @property
    def schools(self) -> list[str]:
        """All school names (lets the keyword parser recognize them)."""
        return sorted(self.scored["school"].unique())

    def get_player(self, player_id: str) -> pd.Series | None:
        rows = self.scored[self.scored["player_id"] == player_id]
        return None if rows.empty else rows.iloc[0]
