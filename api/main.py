"""FastAPI service: a thin HTTP wrapper over scout.service.

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs (auto-generated, handy for demos)

The Streamlit UI doesn't need this by default (it calls scout.service directly).
Run the API when another program needs HTTP access, or set SCOUT_MODE=api for the UI.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from scout.config import DB_PATH, has_api_key
from scout.report import ReportResult
from scout.search import SearchBackend
from scout.service import ScoutService, UnknownPlayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("scout.api")


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500, examples=["D2 point guard under 6'3\" who can pass"])
    limit: int = Field(20, ge=1, le=100)
    include_gated: bool = Field(False, description="Include players below the Translation ranking gate")


def create_app(backend: SearchBackend | None = None, db_path: Path | str = DB_PATH) -> FastAPI:
    """App factory so tests can inject a backend and a temp DB."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = ScoutService(backend, db_path)
        log.info("Loaded %d players; LLM %s", len(app.state.service.backend.scored),
                 "enabled" if has_api_key() else "disabled (no key)")
        yield

    app = FastAPI(title="ScoutAgent API", version="0.1.0", lifespan=lifespan)

    def svc(request: Request) -> ScoutService:
        return request.app.state.service

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return svc(request).health()

    @app.post("/search")
    def search(req: SearchRequest, request: Request) -> dict[str, Any]:
        return svc(request).search(req.query, limit=req.limit, include_gated=req.include_gated)

    @app.get("/players/{player_id}")
    def get_player(player_id: str, request: Request) -> dict[str, Any]:
        try:
            return svc(request).player(player_id)
        except UnknownPlayer:
            raise HTTPException(status_code=404, detail=f"Unknown player_id {player_id}")

    @app.post("/players/{player_id}/report")
    def report(player_id: str, request: Request, force: bool = False) -> ReportResult:
        try:
            return ReportResult(**svc(request).report(player_id, force=force))
        except UnknownPlayer:
            raise HTTPException(status_code=404, detail=f"Unknown player_id {player_id}")

    return app


app = create_app()
