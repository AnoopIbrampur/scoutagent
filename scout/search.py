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

import os

from scout.config import DB_PATH, SCORING_CONFIG
from scout.db import fetch_playtypes, fetch_players, fetch_real_players, fetch_synergy, get_connection
from scout.query_parser import SearchFilters, inches_to_str
from scout.scoring import explain_player, score_players
from scout.synergy import STAT_KEYS, attach_synergy, top_playtypes

# Which percentile columns each skill tag maps to (all percentiles are higher = better).
EMPHASIS_STATS: dict[str, list[str]] = {
    "passing": ["ast_per40", "ast_to"],
    "shooting": ["three_pct_adj", "ts_pct", "ft_pct", "ssm"],
    "scoring": ["pts_per40", "ts_pct", "off_ppp"],
    "rebounding": ["reb_per40"],
    "defense": ["stl_per40", "blk_per40", "def_ppp"],
    "rim_protection": ["blk_per40", "reb_per40"],
    "efficiency": ["ts_pct", "ast_to", "off_ppp"],
    "ball_security": ["ast_to", "tov_per40", "to_pct"],
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
            cols.append(df[f"pct_{stat}"])  # NaN (no Synergy data) is skipped by mean()
    return pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(float("nan"), index=df.index)


def player_summary(row: pd.Series) -> dict[str, Any]:
    """Compact JSON-safe view of a scored player for result lists."""
    keys = ["player_id", "name", "school", "conference", "division", "position",
            "height_in", "class_year", "translation_score", "attention_score", "underrated_score"]
    out = {k: to_python(row[k]) for k in keys}
    for k in ("translation_score", "attention_score", "underrated_score"):
        out[k] = None if out[k] is None else round(out[k], 1)  # None = unknown (no attention data)
    out["height"] = inches_to_str(int(row["height_in"]))
    return out


class SearchBackend(ABC):
    """Interface every search implementation must provide."""

    @abstractmethod
    def search(self, filters: SearchFilters, limit: int = 20, apply_gate: bool = True) -> dict[str, Any]:
        """Return {"results": [...], "total_matches", "excluded_by_gate", "where_sql"}."""

    @abstractmethod
    def get_synergy(self, player_id: str) -> dict[str, list[dict[str, Any]]] | None:
        """A player's Synergy situational rows (offense and defense), JSON-safe, or None."""
        if self.synergy is None:
            return None
        tables = dict(zip(("offense", "defense"), self.synergy)) | self.playtypes
        out = {}
        for name, table in tables.items():
            rows = table[table["player_id"] == player_id]
            out[name] = [{"situation": r["situation"], **{k: to_python(r[k]) for k in STAT_KEYS}}
                         for _, r in rows.iterrows()]
        return out if out.get("offense") or out.get("defense") else None

    def playtype_summary(self, player_id: str, n: int = 3) -> list[dict[str, Any]]:
        """His n most-used offensive play types (10+ possessions), formatted for report facts."""
        table = self.playtypes.get("playtypes_offense")
        if table is None or table.empty:
            return []
        return [{
            "play_type": r["situation"], "share_of_possessions": f"{100 * r['pct_time']:.1f}%",
            "possessions": int(r["poss"]), "points_per_possession": round(float(r["ppp"]), 3),
            "rating": r["rating"] if isinstance(r["rating"], str) else None,
        } for _, r in top_playtypes(table, player_id, n).iterrows()]

    def get_player(self, player_id: str) -> pd.Series | None:
        """Return the full scored row for one player, or None."""


class DuckDBSearchBackend(SearchBackend):
    """Scores all players on load, then filters with DuckDB SQL over the scored table."""

    def __init__(self, db_path: Path | str = DB_PATH, players: pd.DataFrame | None = None,
                 synergy: tuple[pd.DataFrame, pd.DataFrame] | None = None,
                 playtypes: dict[str, pd.DataFrame] | None = None, dataset: str | None = None) -> None:
        """Load from DuckDB, or use the given players (+ optional Synergy tables).

        dataset: "synthetic" (default) or "real" (env SCOUT_DATASET). Real mode serves the
        players loaded by scripts.load_real_players and compares them against the
        synthetic D1 pool, because percentiles need a D1 baseline.
        """
        self.dataset = dataset or os.getenv("SCOUT_DATASET", "synthetic")
        reference, baseline = None, None
        if players is None:
            with get_connection(db_path, read_only=True) as con:
                if self.dataset == "real":
                    try:
                        players = fetch_real_players(con)
                    except duckdb.CatalogException as e:
                        raise RuntimeError("No real players loaded. Run: python -m scripts.load_real_players") from e
                    synergy, playtypes = fetch_synergy(con, "synergy_real"), fetch_playtypes(con, "synergy_real")
                    synth = fetch_players(con)
                    reference = attach_synergy(synth[synth["division"] == "D1"], *fetch_synergy(con))
                    baseline = "synthetic D1"
                else:
                    players, synergy, playtypes = fetch_players(con), fetch_synergy(con), fetch_playtypes(con)
        self.synergy = synergy if synergy and synergy[0] is not None else None
        self.playtypes = playtypes or {}
        self.scored = score_players(attach_synergy(players, *(self.synergy or (None, None))),
                                    reference=reference, baseline_label=baseline)
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

    def search(self, filters: SearchFilters, limit: int = 20, apply_gate: bool = True) -> dict[str, Any]:
        where, params = self._where(filters)
        matches = self._con.execute(f"SELECT * FROM player_scores WHERE {where}", params).df()
        total = len(matches)

        gate = SCORING_CONFIG.get("min_translation_to_rank")
        excluded = 0
        if apply_gate and filters.sort_by == "underrated" and gate is not None:
            keep = matches["translation_score"] >= gate
            excluded = int((~keep).sum())
            matches = matches[keep]

        base = matches["underrated_score" if filters.sort_by == "underrated" else "translation_score"]
        if filters.stat_emphasis:
            emph = emphasis_score(matches, filters.stat_emphasis)
            rank = (1 - EMPHASIS_WEIGHT) * base + EMPHASIS_WEIGHT * emph
        else:
            emph, rank = pd.Series(float("nan"), index=matches.index), base
        matches = matches.assign(emphasis_score=emph, rank_score=rank).sort_values(
            "rank_score", ascending=False, na_position="last")

        results = []
        for _, row in matches.head(limit).iterrows():
            exp = explain_player(row)
            item = player_summary(row)
            item.update(
                rank_score=to_python(round(float(row["rank_score"]), 1)),
                emphasis_score=None if pd.isna(row["emphasis_score"]) else round(float(row["emphasis_score"]), 1),
                top_factor=exp["top_factors"][0]["text"],
                explanation=exp,
            )
            results.append(item)
        return {"results": results, "total_matches": total, "excluded_by_gate": excluded,
                "gate": gate if (apply_gate and filters.sort_by == "underrated") else None, "where_sql": where}

    @property
    def schools(self) -> list[str]:
        """All school names (lets the keyword parser recognize them)."""
        return sorted(self.scored["school"].unique())

    def get_synergy(self, player_id: str) -> dict[str, list[dict[str, Any]]] | None:
        """A player's Synergy situational rows (offense and defense), JSON-safe, or None."""
        if self.synergy is None:
            return None
        tables = dict(zip(("offense", "defense"), self.synergy)) | self.playtypes
        out = {}
        for name, table in tables.items():
            rows = table[table["player_id"] == player_id]
            out[name] = [{"situation": r["situation"], **{k: to_python(r[k]) for k in STAT_KEYS}}
                         for _, r in rows.iterrows()]
        return out if out.get("offense") or out.get("defense") else None

    def playtype_summary(self, player_id: str, n: int = 3) -> list[dict[str, Any]]:
        """His n most-used offensive play types (10+ possessions), formatted for report facts."""
        table = self.playtypes.get("playtypes_offense")
        if table is None or table.empty:
            return []
        return [{
            "play_type": r["situation"], "share_of_possessions": f"{100 * r['pct_time']:.1f}%",
            "possessions": int(r["poss"]), "points_per_possession": round(float(r["ppp"]), 3),
            "rating": r["rating"] if isinstance(r["rating"], str) else None,
        } for _, r in top_playtypes(table, player_id, n).iterrows()]

    def get_player(self, player_id: str) -> pd.Series | None:
        rows = self.scored[self.scored["player_id"] == player_id]
        return None if rows.empty else rows.iloc[0]
