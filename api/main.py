"""FastAPI service: search, player detail, and scouting reports.

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs (auto-generated, handy for demos)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from scout.config import DB_PATH, POSITION_WEIGHTS, SCOUT_MODEL, has_api_key
from scout.query_parser import parse_query
from scout.report import ReportResult, generate_report
from scout.scoring import STAT_LABELS, explain_player
from scout.search import DuckDBSearchBackend, SearchBackend, player_summary, to_python

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("scout.api")

RAW_FIELDS = ["games", "minutes_pg", "points_pg", "rebounds_pg", "assists_pg", "steals_pg", "blocks_pg",
              "turnovers_pg", "fg_pct", "three_pct", "three_att_pg", "ft_pct", "ft_att_pg", "weight_lb",
              "conference_strength", "recruiting_stars", "social_followers", "news_mentions_90d"]
DERIVED_FIELDS = ["ts_pct", "ast_to", "pts_per40", "reb_per40", "ast_per40", "stl_per40", "blk_per40",
                  "tov_per40", "three_pct_adj", "fga_pg", "season_minutes", "level_mult"]


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500, examples=["D2 point guard under 6'3\" who can pass"])
    limit: int = Field(20, ge=1, le=100)
    include_gated: bool = Field(False, description="Include players below the Translation ranking gate")


def create_app(backend: SearchBackend | None = None, db_path: Path | str = DB_PATH) -> FastAPI:
    """App factory so tests can inject a backend and a temp DB."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if backend is None:
            try:
                app.state.backend = DuckDBSearchBackend(db_path)
            except Exception as e:  # missing/empty DB is the common first-run mistake
                raise RuntimeError(f"Could not load players from {db_path}. "
                                   "Run: python -m scripts.generate_seed_data") from e
        else:
            app.state.backend = backend
        app.state.db_path = db_path
        log.info("Loaded %d players; LLM %s", len(app.state.backend.scored),
                 "enabled" if has_api_key() else "disabled (no key)")
        yield

    app = FastAPI(title="ScoutAgent API", version="0.1.0", lifespan=lifespan)

    def _player_or_404(request: Request, player_id: str):
        row = request.app.state.backend.get_player(player_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Unknown player_id {player_id}")
        return row

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "players": len(request.app.state.backend.scored),
                "dataset": getattr(request.app.state.backend, "dataset", "synthetic"),
                "llm_enabled": has_api_key(), "model": SCOUT_MODEL}

    @app.post("/search")
    def search(req: SearchRequest, request: Request) -> dict[str, Any]:
        be = request.app.state.backend
        parsed = parse_query(req.query, known_schools=getattr(be, "schools", None))
        res = be.search(parsed.filters, limit=req.limit, apply_gate=not req.include_gated)
        return {"parsed": {"filters": parsed.filters.model_dump(), "method": parsed.method,
                           "notes": parsed.notes, "chips": parsed.chips()}, **res}

    @app.get("/players/{player_id}")
    def get_player(player_id: str, request: Request) -> dict[str, Any]:
        row = _player_or_404(request, player_id)
        pos = row["position"]
        chart = [{"stat": s, "label": STAT_LABELS[s], "percentile": round(float(row[f"pct_{s}"]), 1),
                  "weight": w} for s, w in POSITION_WEIGHTS[pos].items()]
        chart.append({"stat": "tov_per40", "label": "Turnover avoidance",
                      "percentile": round(float(row["pct_tov_per40"]), 1), "weight": 0.0})
        chart = [c for c in chart if c["percentile"] == c["percentile"]]  # drop NaN (no Synergy data)
        return {
            "summary": player_summary(row),
            "raw": {k: to_python(row[k]) for k in RAW_FIELDS},
            "derived": {k: to_python(row[k]) for k in DERIVED_FIELDS},
            "adjusted": {k: to_python(row[f"adj_{k}"]) for k in
                         ["pts_per40", "reb_per40", "ast_per40", "stl_per40", "blk_per40", "tov_per40",
                          "ts_pct", "three_pct_adj", "ft_pct", "ast_to"]},
            "percentiles_vs_d1_position": chart,
            "explanation": explain_player(row),
            "synergy": request.app.state.backend.get_synergy(player_id),
        }

    @app.post("/players/{player_id}/report")
    def report(player_id: str, request: Request, force: bool = False) -> ReportResult:
        row = _player_or_404(request, player_id)
        playtypes = request.app.state.backend.playtype_summary(player_id)
        return generate_report(row, force=force, db_path=request.app.state.db_path, playtypes=playtypes)

    return app


app = create_app()
