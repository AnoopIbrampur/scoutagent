"""The app's operations in one place: search, player detail, reports.

Both front doors use this: the FastAPI service (api/main.py) wraps it in HTTP,
and the Streamlit UI calls it directly when it runs as a single process (the
default, and how the hosted demo runs). Keeping one implementation means the
two can't drift apart.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import duckdb

from scout.config import DB_PATH, POSITION_WEIGHTS, SCOUT_MODEL, has_api_key
from scout.query_parser import parse_query
from scout.report import generate_report
from scout.scoring import STAT_LABELS, explain_player
from scout.search import DuckDBSearchBackend, SearchBackend, player_summary, to_python

log = logging.getLogger(__name__)

RAW_FIELDS = ["games", "minutes_pg", "points_pg", "rebounds_pg", "assists_pg", "steals_pg", "blocks_pg",
              "turnovers_pg", "fg_pct", "three_pct", "three_att_pg", "ft_pct", "ft_att_pg", "weight_lb",
              "conference_strength", "recruiting_stars", "social_followers", "news_mentions_90d"]
DERIVED_FIELDS = ["ts_pct", "ast_to", "pts_per40", "reb_per40", "ast_per40", "stl_per40", "blk_per40",
                  "tov_per40", "three_pct_adj", "fga_pg", "season_minutes", "level_mult"]
ADJUSTED_FIELDS = ["pts_per40", "reb_per40", "ast_per40", "stl_per40", "blk_per40", "tov_per40",
                   "ts_pct", "three_pct_adj", "ft_pct", "ast_to"]


class UnknownPlayer(KeyError):
    """No player with that id."""


def ensure_database(db_path: Path | str = DB_PATH) -> bool:
    """Build the synthetic dataset if the database is missing or empty. Returns True if it built.

    The database file is gitignored, so a fresh clone (or a hosted deploy) has none.
    Building takes about a second and is deterministic, so every visitor sees the same players.
    """
    db_path = Path(db_path)
    if db_path.exists():
        try:
            with duckdb.connect(str(db_path), read_only=True) as con:
                if con.execute("SELECT count(*) FROM players").fetchone()[0] > 0:
                    return False
        except duckdb.Error:
            pass  # missing table or unreadable file: rebuild below
    from scripts.generate_seed_data import build_database  # local import: scripts depend on scout
    log.info("No dataset at %s; building the synthetic one", db_path)
    build_database(db_path)
    return True


class ScoutService:
    """Search, player detail, and reports over one loaded backend."""

    def __init__(self, backend: SearchBackend | None = None, db_path: Path | str = DB_PATH) -> None:
        self.db_path = db_path
        if backend is None:
            ensure_database(db_path)
            backend = DuckDBSearchBackend(db_path)
        self.backend = backend
        self._report_lock = threading.Lock()  # report cache writes go to one DuckDB file

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "players": len(self.backend.scored),
                "dataset": getattr(self.backend, "dataset", "synthetic"),
                "llm_enabled": has_api_key(), "model": SCOUT_MODEL}

    def search(self, query: str, limit: int = 20, include_gated: bool = False) -> dict[str, Any]:
        be = self.backend
        parsed = parse_query(query, known_schools=getattr(be, "schools", None))
        res = be.search(parsed.filters, limit=limit, apply_gate=not include_gated)
        return {"parsed": {"filters": parsed.filters.model_dump(), "method": parsed.method,
                           "notes": parsed.notes, "chips": parsed.chips()}, **res}

    def _row(self, player_id: str):
        row = self.backend.get_player(player_id)
        if row is None:
            raise UnknownPlayer(player_id)
        return row

    def player(self, player_id: str) -> dict[str, Any]:
        row = self._row(player_id)
        chart = [{"stat": s, "label": STAT_LABELS[s], "percentile": round(float(row[f"pct_{s}"]), 1),
                  "weight": w} for s, w in POSITION_WEIGHTS[row["position"]].items()]
        chart.append({"stat": "tov_per40", "label": "Turnover avoidance",
                      "percentile": round(float(row["pct_tov_per40"]), 1), "weight": 0.0})
        chart = [c for c in chart if c["percentile"] == c["percentile"]]  # drop NaN (no Synergy data)
        return {
            "summary": player_summary(row),
            "raw": {k: to_python(row[k]) for k in RAW_FIELDS},
            "derived": {k: to_python(row[k]) for k in DERIVED_FIELDS},
            "adjusted": {k: to_python(row[f"adj_{k}"]) for k in ADJUSTED_FIELDS},
            "percentiles_vs_d1_position": chart,
            "explanation": explain_player(row),
            "synergy": self.backend.get_synergy(player_id),
        }

    def report(self, player_id: str, force: bool = False) -> dict[str, Any]:
        row = self._row(player_id)
        playtypes = self.backend.playtype_summary(player_id)
        with self._report_lock:
            return generate_report(row, force=force, db_path=self.db_path, playtypes=playtypes).model_dump()
